import logging
import dash
import base64
from io import BytesIO, StringIO
from pathlib import Path
import time
import feffery_antd_components as fac
import feffery_utils_components as fuc
from itertools import cycle
import numpy as np
import pandas as pd
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from ..pca import SciPyPCA as PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import colors as plotly_colors
from ..duckdb_manager import duckdb_connection, create_pivot
from ..plugin_interface import PluginInterface
import plotly.express as px
from scipy.stats import ttest_ind, f_oneway
from ..sample_metadata import GROUP_COLUMNS, GROUP_LABELS
from .scalir import (
    intersect_peaks,
    fit_estimator,
    build_concentration_table,
    training_plot_frame,
    plot_standard_curve,
    slugify_label,
)

_label = "Analysis"

logger = logging.getLogger(__name__)


def load_persisted_scalir_results(wdir: str) -> dict | None:
    """
    Load persisted SCALiR results from workspace if they exist.
    
    Returns:
        Dictionary with results data if files exist, None otherwise
    """
    output_dir = Path(wdir) / "results" / "scalir"
    train_frame_path = output_dir / "train_frame.csv"
    params_path = output_dir / "standard_curve_parameters.csv"
    concentrations_path = output_dir / "concentrations.csv"
    units_path = output_dir / "units.csv"
    plots_dir = output_dir / "plots"
    
    # Check if essential files exist
    if not train_frame_path.exists() or not params_path.exists():
        return None
    
    try:
        train_frame = pd.read_csv(train_frame_path)
        params = pd.read_csv(params_path)
        units = pd.read_csv(units_path) if units_path.exists() else None
        
        # Get list of metabolites from params
        common = params['peak_label'].tolist() if 'peak_label' in params.columns else []
        
        return {
            "train_frame": train_frame.to_json(orient="split"),
            "units": units.to_json(orient="split") if units is not None else None,
            "params": params.to_json(orient="split"),
            "plot_dir": str(plots_dir),
            "common": common,
            "generated_all_plots": plots_dir.exists() and any(plots_dir.iterdir()) if plots_dir.exists() else False,
            "concentrations_path": str(concentrations_path),
            "params_path": str(params_path),
        }
    except Exception as e:
        logger.warning(f"Failed to load persisted SCALiR results: {e}")
        return None

PCA_COMPONENT_OPTIONS = [
    {'label': f'PC{i}', 'value': f'PC{i}'}
    for i in range(1, 6)
]
NORM_OPTIONS = [
    {'label': 'None (raw)', 'value': 'none'},
    {'label': 'Z-score', 'value': 'zscore'},
    {'label': 'Rocke-Durbin', 'value': 'durbin'},
    {'label': 'Z-score + Rocke-Durbin', 'value': 'zscore_durbin'},
]
TAB_DEFAULT_NORM = {
    'clustermap': 'zscore',
    'pca': 'durbin',
    'raincloud': 'durbin',
}
GROUPING_FIELDS = ['sample_type'] + GROUP_COLUMNS
GROUP_SELECT_OPTIONS = [
    {'label': GROUP_LABELS.get(field, field.replace('_', ' ').title()), 'value': field}
    for field in GROUPING_FIELDS
]

# High-resolution export configuration for Plotly graphs
# Scale of 4 provides ~300 DPI (default is 72 DPI)
PLOTLY_HIGH_RES_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'scale': 4,  # 4x scale ≈ 300 DPI
        'height': None,  # maintains aspect ratio
        'width': None,   # maintains aspect ratio
    },
    'displayModeBar': True,
    'displaylogo': False,
}


class AnalysisPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 9
        logger.info(f'Initiated {_label} plugin')

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)

    def outputs(self):
        return _outputs


def rocke_durbin(df: pd.DataFrame, c: float) -> pd.DataFrame:
    # df: samples x features (metabolites)
    z = df.to_numpy(dtype=float)
    ef = np.log((z + np.sqrt(z ** 2 + c ** 2)) / 2.0)
    return pd.DataFrame(ef, index=df.index, columns=df.columns)


def run_pca_samples_in_cols(df: pd.DataFrame, n_components=None, random_state=0):
    """Run PCA with samples in rows and targets in columns."""

    X = df.to_numpy(dtype=float)  # samples x targets

    pca = PCA(n_components=n_components, random_state=random_state)
    scores = pca.fit_transform(X)  # samples x components

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    pc_labels = [f"PC{i + 1}" for i in range(pca.n_components_)]

    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_), # https://stats.stackexchange.com/questions/143905/loadings-vs-eigenvectors-in-pca-when-to-use-one-or-another
        columns=pc_labels,
        index=df.columns,
    )
    scores_df = pd.DataFrame(
        scores,
        index=df.index,
        columns=pc_labels,
    )

    return {
        "pca": pca,
        "scores": scores_df,
        "loadings": loadings,
        "explained_variance_ratio": pd.Series(explained, index=pc_labels),
        "cumulative_variance_ratio": pd.Series(cumulative, index=pc_labels),
    }


clustermap_tab = html.Div([
    fac.AntdSpace([
        html.Span("X-axis Font:", style={'fontWeight': 500, 'fontSize': 12}),
        fac.AntdSlider(
            id='clustermap-fontsize-x-slider',
            min=0,
            max=20,
            step=1,
            value=5,
            marks={0: '0', 10: '10', 20: '20'},
            style={'width': 200, 'fontSize': 10},
        ),
        html.Span("Y-axis Font:", style={'fontWeight': 500, 'fontSize': 12, 'marginLeft': '24px'}),
        fac.AntdSlider(
            id='clustermap-fontsize-y-slider',
            min=0,
            max=20,
            step=1,
            value=5,
            marks={0: '0', 10: '10', 20: '20'},
            style={'width': 200, 'fontSize': 10},
        ),
        fac.AntdTooltip(
            fac.AntdButton(
                "Regenerate",
                id='clustermap-regenerate-btn',
                type='default',
                size='small',
                style={'marginLeft': '24px'},
            ),
            title="Click to regenerate heatmap with new font sizes",
            placement='right',
        ),
    ], size=8, style={'padding': '8px 20px'}),
    html.Div(
        fuc.FefferyResizable(
            fac.AntdSpin(
                fac.AntdCenter(
                    html.Img(id='bar-graph-matplotlib', style={
                        'width': '100%',
                        'height': '100%',
                        'object-fit': 'cover',
                        'border': '0px solid #dee2e6'
                    }),
                    style={
                        'height': '100%',
                    },
                ),
                id='clustermap-spinner',
                spinning=True,
                text='Loading clustermap...',
                style={'minHeight': '20vh', 'width': '100%'},
            ),
            minWidth=100,
            minHeight=100,
        ),
        style={'overflow': 'auto', 'height': 'calc(100vh - 200px)'}
    )
], style={'height': 'calc(100vh - 156px)'})
pca_tab = html.Div(
    [
        fac.AntdSpace(
            [
                html.Span("X axis:"),
                fac.AntdSelect(
                    id='pca-x-comp',
                    options=PCA_COMPONENT_OPTIONS,
                    value='PC1',
                    allowClear=False,
                    style={'width': 140},
                ),
                html.Span("Y axis:"),
                fac.AntdSelect(
                    id='pca-y-comp',
                    options=PCA_COMPONENT_OPTIONS,
                    value='PC2',
                    allowClear=False,
                    style={'width': 140},
                ),
            ],
            style={'marginBottom': 10},
        ),
        fac.AntdSpin(
            dcc.Graph(
                id='pca-graph',
                config=PLOTLY_HIGH_RES_CONFIG,
                # Start with invisible figure to show only spinner during loading
                figure={
                    'data': [],
                    'layout': {
                        'xaxis': {'visible': False},
                        'yaxis': {'visible': False},
                        'paper_bgcolor': 'white',
                        'plot_bgcolor': 'white',
                        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                        'height': 700,  # Match final PCA plot height for proper spinner area
                    }
                },
            ),
            text='Loading PCA...',
            style={'minHeight': '20vh', 'width': '100%'},
        ),
    ]
)

# SCALiR tab has been moved to Processing plugin as a modal workflow

_layout = html.Div(
    [
        fac.AntdFlex(
            [
                fac.AntdFlex(
                    [
                        fac.AntdTitle(
                            'Analysis', level=4, style={'margin': '0'}
                        ),
                        fac.AntdIcon(
                            id='analysis-tour-icon',
                            icon='pi-info',
                            style={"cursor": "pointer", 'paddingLeft': '10px'},
                        ),
                    ],
                    align='center',
                ),
                # fac.AntdDropdown(
                #     id='processing-options',
                #     title='Options',
                #     buttonMode=True,
                #     arrow=True,
                #     menuItems=[
                #         {'title': fac.AntdText('Download', strong=True), 'key': 'processing-download'},
                #         {'isDivider': True},
                #         {'title': fac.AntdText('Delete selected', strong=True, type='warning'),
                #          'key': 'processing-delete-selected'},
                #         {'title': fac.AntdText('Clear table', strong=True, type='danger'),
                #          'key': 'processing-delete-all'},
                #     ],
                #     buttonProps={'style': {'textTransform': 'uppercase'}},
                # ),
            ],
            justify="space-between",
            align="center",
            gap="middle",
        ),
        fac.AntdTabs(
            id='analysis-tabs',
            items=[
                {'key': 'pca', 'label': 'PCA', 'children': pca_tab},
                {
                    'key': 'raincloud',
                    'label': 'Violin',
                    'children': html.Div(
                        [
                            fac.AntdFlex(
                                [
                                    fac.AntdText('Target to display', strong=True),
                                    fac.AntdSelect(
                                        id='violin-comp-checks',
                                        # mode='multiple',  # Changed to single select
                                        options=[],
                                        value=None,
                                        allowClear=False,
                                        # maxTagCount=4,
                                        optionFilterProp='label',
                                        optionFilterMode='case-insensitive',
                                        style={'width': '320px'},
                                    ),
                                    fac.AntdText(
                                        'Click on individual samples to show the chromatogram.',
                                        type='secondary',
                                    ),
                                ],
                                align='center',
                                gap='small',
                                wrap=True,
                                style={'paddingBottom': '0.75rem'},
                            ),
                            fac.AntdSpin(
                                html.Div(
                                    id='violin-graphs',
                                    style={
                                        'display': 'flex',
                                        'flexDirection': 'column',
                                        'gap': '24px',
                                    },
                                ),
                                text='Loading Violin...',
                                style={'minHeight': '20vh', 'width': '100%'},
                            ),
                            html.Div(
                                [
                                    fac.AntdDivider("Chromatogram", style={'margin': '12px 0 12px 0'}),
                                    fac.AntdSpin(
                                        dcc.Graph(
                                            id='violin-chromatogram',
                                            config=PLOTLY_HIGH_RES_CONFIG,
                                            style={'height': '400px'},
                                        ),
                                        text='Loading Chromatogram...',
                                    ),
                                ],
                                id='violin-chromatogram-container',
                                style={'display': 'none'}
                            ),
                        ]
                    ),
                },
                {'key': 'clustermap', 'label': 'Clustermap', 'children': clustermap_tab},
            ],
            centered=True,
            defaultActiveKey='pca',
            style={'margin': '12px 0 0 0'},
            tabBarLeftExtraContent=fac.AntdSpace(
                [
                    fac.AntdSpace(
                        [
                            fac.AntdText("Metric:", style={'fontWeight': 500}),
                            fac.AntdSelect(
                                id='analysis-metric-select',
                                options=[
                                    {'label': 'Peak Area', 'value': 'peak_area'},
                                    {'label': 'Peak Area (Top 3)', 'value': 'peak_area_top3'},
                                    {'label': 'Peak Max', 'value': 'peak_max'},
                                    {'label': 'Peak Mean', 'value': 'peak_mean'},
                                    {'label': 'Peak Median', 'value': 'peak_median'},
                                    {'label': 'Concentration', 'value': 'scalir_conc'},
                                ],
                                value='peak_area',
                                optionFilterProp='label',
                                optionFilterMode='case-insensitive',
                                allowClear=False,
                                style={'width': 200},
                            ),
                        ],
                        align='center',
                        size='small',
                    ),
                    fac.AntdSpace(
                        [
                            fac.AntdText("Transformations:", style={'fontWeight': 500}),
                            fac.AntdSelect(
                                id='analysis-normalization-select',
                                options=NORM_OPTIONS,
                                value='zscore',
                                optionFilterProp='label',
                                optionFilterMode='case-insensitive',
                                allowClear=False,
                                style={'width': 180},
                            ),
                        ],
                        align='center',
                        size='small',
                    ),
                    fac.AntdSpace(
                        [
                            fac.AntdText("Group by:", style={'fontWeight': 500}),
                            fac.AntdSelect(
                                id='analysis-grouping-select',
                                options=GROUP_SELECT_OPTIONS,
                                value='sample_type',
                                allowClear=False,
                                style={'width': 180},
                            ),
                        ],
                        align='center',
                        size='small',
                    ),
                ],
                size='small',
                id='analysis-metric-container',
            ),
        )
        , fac.AntdTour(
            locale='en-us',
            steps=[],
            id='analysis-tour',
            open=False,
            current=0,
        ),
        html.Div(id="analysis-notifications-container")
    ]
)

_outputs = None


def layout():
    return _layout


allowed_metrics = {
    'peak_area',
    'peak_area_top3',
    'peak_max',
    'peak_mean',
    'peak_median',
    'scalir_conc',
}


def _parse_uploaded_standards(contents, filename):
    if not contents or not filename:
        raise ValueError("No standards file provided.")
    if len(contents) > 15_000_000:
        raise ValueError("Standards file is too large (limit: 15MB).")
    content_type, content_string = contents.split(",", 1)
    if content_type and "csv" not in content_type and "excel" not in content_type:
        raise ValueError("Unsupported file type. Use CSV or Excel.")
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(BytesIO(decoded))
    try:
        return pd.read_csv(StringIO(decoded.decode("utf-8")))
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(decoded))

def _plot_curve_fig(frame: pd.DataFrame, peak_label: str, units: pd.DataFrame = None, params_df: pd.DataFrame = None):
    subset = frame[frame.peak_label == peak_label]
    if subset.empty:
        return go.Figure()

    unit = None
    if units is not None and "unit" in units.columns:
        match = units[units.peak_label == peak_label]
        if not match.empty:
            unit = match.unit.iloc[0]

    in_range = subset[subset.in_range == 1]
    out_range = subset[subset.in_range != 1]

    fig = go.Figure()

    if not out_range.empty:
        fig.add_trace(
            go.Scatter(
                x=out_range.true_conc,
                y=out_range.value,
                mode="markers",
                marker=dict(color="#888888", size=10),
                name="Outside range",
            )
        )
    if not in_range.empty:
        fig.add_trace(
            go.Scatter(
                x=in_range.true_conc,
                y=in_range.value,
                mode="markers",
                marker=dict(color="#111111", size=10),
                name="In range",
            )
        )
        line_data = in_range.sort_values("pred_conc")
        fig.add_trace(
            go.Scatter(
                x=line_data.pred_conc,
                y=line_data.value,
                mode="lines",
                line=dict(color="#111111", width=2),
                name="Fit",
            )
        )

    xlabel = f"{peak_label} concentration"
    if unit:
        xlabel = f"{xlabel} ({unit})"
    params_text = None
    if params_df is not None and not params_df.empty:
        try:
            row = params_df[params_df.peak_label == peak_label].iloc[0]
            slope = row.get('slope', None)
            intercept = row.get('intercept', None)
            lloq = row.get('LLOQ', None)
            uloq = row.get('ULOQ', None)
            params_text = (
                f"slope={slope:.1f}<br>"
                f"intercept={intercept:.0f}<br>"
                f"LLOQ={lloq:.2g}<br>"
                f"ULOQ={uloq:.2g}"
            )
        except Exception:
            params_text = None
    fig.update_layout(
        title=dict(text=peak_label, x=0.5, xanchor='center'),
        xaxis=dict(
            title=xlabel,
            type="log",
            ticks="outside",
        ),
        yaxis=dict(
            title=f"{peak_label} intensity (AU)",
            type="log",
            tickformat="~s",
            tickmode="auto",
            nticks=6,
            ticks="outside",
        ),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=60, r=20, t=90, b=60),
        template="plotly_white",
    )
    if params_text:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.03,
            y=0.97,
            xanchor="left",
            yanchor="top",
            text=params_text,
            showarrow=False,
            font=dict(size=12, color="#444"),
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderpad=4,
        )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig

def _analysis_tour_steps(active_tab: str):
    return [
        {
            'title': 'Analysis overview',
            'description': 'Use the controls above to choose a metric, transformation, and grouping.',
        },
        {
            'title': 'Metric',
            'description': 'Pick the quantitative metric to visualize.',
            'targetSelector': '#analysis-metric-select',
        },
        {
            'title': 'Transformations',
            'description': 'Normalize or transform the values before plotting.',
            'targetSelector': '#analysis-normalization-select',
        },
        {
            'title': 'Group by',
            'description': 'Group samples for coloring and summaries.',
            'targetSelector': '#analysis-grouping-select',
        },
        {
            'title': 'Tabs',
            'description': 'Switch between PCA, Violin, and Clustermap views.',
            'targetSelector': '#analysis-tabs',
        },
    ]


def _build_color_map(color_df: pd.DataFrame, group_col: str) -> dict:
    if not group_col or group_col not in color_df.columns or color_df.empty:
        return {}
    working = color_df[[group_col, 'color']].copy()
    working = working[working[group_col].notna()]
    working['color'] = working['color'].apply(
        lambda c: c if isinstance(c, str) and c.strip() and c.strip() != '#bbbbbb' else None
    )
    color_map = (
        working.dropna(subset=['color'])
        .drop_duplicates(subset=[group_col])
        .set_index(group_col)['color']
        .to_dict()
    )
    missing = [val for val in working[group_col].dropna().unique() if val not in color_map]
    if missing:
        palette = plotly_colors.qualitative.Plotly
        for val, color in zip(missing, cycle(palette)):
            color_map[val] = color
    return color_map

def _clean_numeric(numeric_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = numeric_df.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if cleaned.isna().any().any():
        cleaned = cleaned.fillna(0)
    return cleaned



def _create_pivot_custom(conn, value='peak_area', table='results'):
    """
    Local implementation of create_pivot to ensure table parameter is respected.
    """
    # Get ordered peak labels
    ordered_pl = [row[0] for row in conn.execute(f"""
        SELECT DISTINCT r.peak_label
        FROM {table} r
        JOIN targets t ON r.peak_label = t.peak_label
        ORDER BY t.ms_type
    """).fetchall()]

    group_cols_sql = ",\n                ".join([f"s.{col}" for col in GROUP_COLUMNS])

    query = f"""
        PIVOT (
            SELECT
                s.ms_type,
                s.sample_type,
                {group_cols_sql},
                r.ms_file_label,
                r.peak_label,
                r.{value}
            FROM {table} r
            JOIN samples s ON s.ms_file_label = r.ms_file_label
            WHERE s.use_for_analysis = TRUE
            ORDER BY s.ms_type, r.peak_label
        )
        ON peak_label
        USING FIRST({value})
        ORDER BY ms_type
    """
    df = conn.execute(query).df()
    meta_cols = ['ms_type', 'sample_type', *GROUP_COLUMNS, 'ms_file_label']
    keep_cols = [col for col in meta_cols if col in df.columns] + ordered_pl
    return df[keep_cols]

def show_tab_content(section_context, tab_key, x_comp, y_comp, violin_comp_checks, metric_value, norm_value,
                    group_by, regen_clicks, fontsize_x, fontsize_y, wdir):
    if not section_context or section_context.get('page') != 'Analysis':
        raise PreventUpdate
    # Prevent double-firing when switching tabs forces a normalization update.
    # If the tab change triggered this, and the current norm isn't the tab's default,
    # we skip this and wait for the normalization update to trigger the callback again.
    from dash import callback_context
    triggered_props = [t['prop_id'] for t in callback_context.triggered] if callback_context.triggered else []
    if 'analysis-tabs.activeKey' in triggered_props:
        default_norm = TAB_DEFAULT_NORM.get(tab_key)
        if default_norm and norm_value != default_norm:
            raise PreventUpdate
    if not wdir:
        raise PreventUpdate
    # Early guard: if there are no results yet, return empty placeholders instead of erroring.
    # Create an invisible figure for PCA when there's no data
    invisible_fig = go.Figure()
    invisible_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
        height=10,  # Minimum height allowed by Plotly
    )
    from dash import callback_context
    ctx = callback_context
    triggered_prop = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    grouping_fields = GROUPING_FIELDS
    selected_group = group_by if group_by in grouping_fields else GROUPING_FIELDS[0]
    with duckdb_connection(wdir) as conn:
        if conn is None:
            return None, invisible_fig, [], [], []
        results_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        if results_count == 0:
            return None, invisible_fig, [], [], []
        # Robust metric selection
        metric = 'peak_area'
        if metric_value == 'scalir_conc' or metric_value in allowed_metrics:
            metric = metric_value
        
        logger.info(f"DEBUG: metric_value={metric_value}, metric={metric}")
        
        target_table = 'results'
        if metric == 'scalir_conc':
            scalir_path = Path(wdir) / "results" / "scalir" / "concentrations.csv"
            if not scalir_path.exists():
                return None, invisible_fig, [], [], []
            try:
                conn.execute(f"CREATE OR REPLACE TEMP VIEW scalir_temp_conc AS SELECT * FROM read_csv_auto('{scalir_path}')")
                conn.execute("""
                    CREATE OR REPLACE TEMP VIEW scalir_results_view AS 
                    SELECT 
                        r.ms_file_label, 
                        r.peak_label, 
                        s.pred_conc AS scalir_conc 
                    FROM results r 
                    LEFT JOIN scalir_temp_conc s 
                    ON r.ms_file_label = CAST(s.ms_file AS VARCHAR) AND r.peak_label = s.peak_label
                """)
                target_table = 'scalir_results_view'
            except Exception as e:
                logger.error(f"Error preparing SCALiR data: {e}")
                return None, invisible_fig, [], [], []

        df = _create_pivot_custom(conn, value=metric, table=target_table)
        df.set_index('ms_file_label', inplace=True)
        group_field = selected_group if selected_group in df.columns else (
            'sample_type' if 'sample_type' in df.columns else None
        )
        group_label = GROUP_LABELS.get(group_field, 'Group')
        missing_group_label = f"{group_label} (unset)"
        metadata_cols = [col for col in ['ms_type'] + grouping_fields if col in df.columns]
        order_df = conn.execute(
            "SELECT ms_file_label FROM samples ORDER BY ms_file_label"
        ).df()["ms_file_label"].tolist()
        ordered_labels = [lbl for lbl in order_df if lbl in df.index]
        leftover_labels = [lbl for lbl in df.index if lbl not in ordered_labels]
        df = df.loc[ordered_labels + leftover_labels]
        # Sort samples by group, then alphabetically within each group for clustermap display
        if group_field and group_field in df.columns:
            df = df.sort_values(by=[group_field, df.index.name or 'ms_file_label'], 
                                key=lambda x: x.str.lower() if x.dtype == 'object' else x)
            # Re-sort preserving group order but sorting index alphabetically within groups
            df['_sort_group'] = df[group_field].fillna('')
            df['_sort_index'] = df.index.str.lower()
            df = df.sort_values(by=['_sort_group', '_sort_index'])
            df = df.drop(columns=['_sort_group', '_sort_index'])
        group_series = df[group_field] if group_field else pd.Series(df.index, index=df.index, name='group')
        if isinstance(group_series, pd.Series):
            group_series = group_series.replace("", pd.NA)
        group_series.name = group_field or 'group'
        group_series = group_series.fillna(missing_group_label)
        colors_df = conn.execute(
            f"SELECT ms_file_label, color, sample_type, {', '.join(GROUP_COLUMNS)} FROM samples"
        ).df()
        color_map = _build_color_map(colors_df, group_field)
        if not color_map and group_field != 'sample_type':
            color_map = _build_color_map(colors_df, 'sample_type')
        if missing_group_label in group_series.values:
            if color_map is None:
                color_map = {}
            color_map.setdefault(missing_group_label, '#bbbbbb')
        raw_df = df.copy()
        df = df.drop(columns=[c for c in metadata_cols if c in df.columns], axis=1)
        compound_options = sorted(
            [
                {'label': c, 'value': c}
                for c in raw_df.columns
                if c not in metadata_cols
            ],
            key=lambda o: o['label'].lower(),
        )
        # Guard against NaN/inf and empty matrices (numeric only) before downstream plots
        df = _clean_numeric(df)
        raw_numeric_cols = [c for c in raw_df.columns if c not in metadata_cols]
        raw_numeric = _clean_numeric(raw_df[raw_numeric_cols])
        raw_df[raw_numeric_cols] = raw_numeric
        color_labels = group_series.reindex(df.index).fillna(missing_group_label)
        if df.empty or raw_numeric.empty:
            return None, invisible_fig, [], [], []
        from ..pca import StandardScaler
        scaler = StandardScaler()
        provided_norm = norm_value  # keep the user-provided value (None on first layout pass)
        norm_value = norm_value or TAB_DEFAULT_NORM.get(tab_key, 'zscore')
        if norm_value == 'zscore':
            zdf = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            ndf = zdf
        elif norm_value == 'durbin':
            ndf = rocke_durbin(df, c=10)
            ndf = _clean_numeric(ndf)
            zdf = ndf
        elif norm_value == 'zscore_durbin':
            z_tmp = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            ndf_tmp = rocke_durbin(z_tmp, c=10)
            ndf = _clean_numeric(ndf_tmp)
            zdf = ndf
        else:
            # raw/none: use df directly
            ndf = df
            zdf = df
        if ndf.empty or zdf.empty:
            raise PreventUpdate
        # Choose matrix for violin based on normalization selection
        if norm_value == 'zscore':
            violin_matrix = zdf
        elif norm_value in ('durbin', 'zscore_durbin'):
            violin_matrix = ndf
        else:
            violin_matrix = df
        if violin_matrix.empty:
            return dash.no_update, invisible_fig, [], [], []
    if tab_key == 'clustermap':
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.patches as mpatches
        matplotlib.use('Agg')
        # Use base font_scale for other elements, apply specific sizes to tick labels
        sns.set_theme(style='white', font_scale=0.5)
        sample_colors = None
        if color_map:
            sample_colors = [color_map.get(lbl, '#bbbbbb') for lbl in color_labels]
        
        norm_label = next((o['label'] for o in NORM_OPTIONS if o['value'] == norm_value), norm_value)
        
        vmin = -2.00 if norm_value == 'zscore' else None
        vmax = 2.05 if norm_value == 'zscore' else None
        fig = sns.clustermap(
                             zdf.T,
                             method='ward', metric='euclidean', 
                             cmap='vlag', center=0, vmin=vmin, vmax=vmax,
                             standard_scale=None,
                             col_cluster=False, 
                             dendrogram_ratio=0.1,
                             figsize=(8, 8),
                             cbar_kws={"orientation": "horizontal"},
                             cbar_pos=(0.01, 0.95, 0.075, 0.01),
                             col_colors=sample_colors,
                             row_colors=['#ffffff'] * len(zdf.T.index),
                             colors_ratio=(0.0015, 0.015)
                            )
        # Ensure white backgrounds across panels
        fig.fig.patch.set_facecolor('white')
        fig.ax_heatmap.set_facecolor('white')
        fig.ax_col_dendrogram.set_facecolor('white')
        fig.ax_row_dendrogram.set_facecolor('white')
        # Apply custom font sizes to x and y tick labels
        x_fontsize = fontsize_x if fontsize_x else 5
        y_fontsize = fontsize_y if fontsize_y else 5
        fig.ax_heatmap.tick_params(axis='x', labelsize=x_fontsize, length=0, rotation=90)
        fig.ax_heatmap.tick_params(axis='y', labelsize=y_fontsize, length=0)
        
        fig.ax_heatmap.set_xlabel('Samples', fontsize=7, labelpad=8)
        fig.ax_cbar.tick_params(which='both', axis='both', width=0.3, length=2, labelsize=4)
        fig.ax_cbar.set_title(norm_label, fontsize=6, pad=4)
        # Legend for grouping colors (top right)
        if color_map:
            used_types = [lbl for lbl in color_labels if lbl in color_map]
            handles = [
                mpatches.Patch(color=color_map[stype], label=stype)
                for stype in dict.fromkeys(used_types)  # preserve order, unique
                if stype in color_map
            ]
            if handles:
                fig.ax_heatmap.legend(
                    handles=handles,
                    title=group_label,
                    bbox_to_anchor=(1.00, 1.085),
                    ncol=len(handles),
                    frameon=False,
                    alignment='right',
                    borderpad=0,
                    fontsize=5,
                    title_fontsize=5,
                )
        from io import BytesIO
        buf = BytesIO()
        # Save a high-resolution copy to disk for durability/exports
        try:
            # Avoid double-saving when the norm dropdown hasn't populated yet (provided_norm is None)
            # Also skip the immediate tab-change trigger; wait for the follow-up with the final norm value.
            if wdir and provided_norm is not None and triggered_prop != 'analysis-tabs':
                cm_dir = Path(wdir) / "analysis" / "clustermap"
                cm_dir.mkdir(parents=True, exist_ok=True)
                safe_metric = slugify_label(metric)
                file_name = f"{safe_metric}_{norm_value}_clustermap.png"
                save_path = cm_dir / file_name
                should_save = True
                if save_path.exists():
                    last_write = save_path.stat().st_mtime
                    should_save = (time.time() - last_write) > 30
                if should_save:
                    fig.savefig(save_path, format="png", dpi=600)
                    logger.info(f"Saved high-res Clustermap: {save_path}")
        except Exception:
            logger.error("Failed to save Clustermap image.", exc_info=True)
            pass
        fig.savefig(buf, format="png", dpi=300)
        # Avoid accumulating open figures across callbacks
        plt.close(fig.fig)
        # fig.savefig('test.png', format="png")
        import base64
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
        return fig_bar_matplotlib, dash.no_update, dash.no_update, compound_options, dash.no_update
    elif tab_key == 'pca':
        logger.info(f"Generating PCA ({x_comp} vs {y_comp})...")
        # n_components should be <= min(n_samples, n_features)
        results = run_pca_samples_in_cols(ndf, n_components=min(ndf.shape[0], ndf.shape[1], 5))
        results['scores']['color_group'] = color_labels
        results['scores']['sample_label'] = results['scores'].index
        x_axis = x_comp or 'PC1'
        y_axis = y_comp or 'PC2'
        component_id = x_axis
        loading_bar = None
        loading_category_order = None
        loadings_df = results.get('loadings')
        has_component = isinstance(loadings_df, pd.DataFrame) and component_id in loadings_df.columns
        if has_component:
            ordered = loadings_df[component_id].abs().sort_values(ascending=False).head(10)
            loading_category_order = list(reversed(ordered.index.tolist()))
            loading_bar = go.Bar(
                x=ordered.loc[loading_category_order],
                y=loading_category_order,
                orientation='h',
                name=component_id,
                marker=dict(color='#bbbbbb'),
                showlegend=False,
                text=loading_category_order,
                textposition='outside',
                cliponaxis=False,
                hovertemplate='<b>%{y}</b><br>|Loading|=%{x:.4f}<extra></extra>',
            )
        variance_bar = go.Bar(
            x=results['explained_variance_ratio'].index,
            y=results['explained_variance_ratio'].values,
            width=0.5,
            showlegend=False,
            marker=dict(color='#bbbbbb'),
        )
        variance_line = go.Scatter(
            x=results['cumulative_variance_ratio'].index,
            y=results['cumulative_variance_ratio'].values,
            mode='lines+markers',
            marker=dict(color='#999999'),
            line=dict(color='#999999'),
            showlegend=False,
        )
        base_scatter = px.scatter(
            results['scores'],
            x=x_axis,
            y=y_axis,
            color='color_group',
            color_discrete_map=color_map if color_map else None,
            hover_data={'sample_label': True},
            title=f'PCA ({x_axis} vs {y_axis})'
        )
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{'rowspan': 2}, {}],
                   [None, {}]],
            column_widths=[0.55, 0.45],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            subplot_titles=(
                f"PCA ({x_axis} vs {y_axis})",
                "Cummulative Variance",
                f"|Loadings| ({component_id})" if has_component else "Loadings",
            ),
        )
        for t in base_scatter.data:
            fig.add_trace(t, row=1, col=1)
        fig.add_trace(variance_bar, row=1, col=2)
        fig.add_trace(variance_line, row=1, col=2)
        # Label PCA scatter axes
        fig.update_xaxes(title_text=x_axis, row=1, col=1)
        fig.update_yaxes(title_text=y_axis, row=1, col=1)
        if loading_bar:
            fig.add_trace(loading_bar, row=2, col=2)
            fig.update_yaxes(
                categoryorder='array',
                categoryarray=loading_category_order,
                showticklabels=False,
                row=2,
                col=2,
            )
            # Nudge the subplot title away from the bars for a bit more breathing room
            for ann in fig['layout']['annotations']:
                if isinstance(ann.text, str) and ann.text.startswith("Loadings"):
                    ann.y = ann.y + 0.02
        fig.update_layout(
            height=700,
            margin=dict(l=160, r=200, t=70, b=60),
            legend_title_text=group_label,
            legend=dict(
                x=-0.05,
                y=1.04,
                xanchor="right",
                yanchor="top",
                orientation="v",
                title=dict(text=f"{group_label}<br>", font=dict(size=14)),
                font=dict(size=12),
            ),
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            xaxis_tickfont=dict(size=12),
            yaxis_tickfont=dict(size=12),
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
        return dash.no_update, fig, dash.no_update, compound_options, dash.no_update
    elif tab_key == 'raincloud':
        logger.info("Generating Violin/Raincloud plots...")
        # Build options list; sort by absolute PC1 loading if available so the most
        # influential metabolites surface first.
        loadings_for_sort = None
        violin_options = compound_options
        # Default selection: highest absolute loading on PC1 (per current metric/norm).
        default_violin = None
        if violin_options:
            try:
                pca_results = run_pca_samples_in_cols(
                    violin_matrix,
                    n_components=min(violin_matrix.shape[0], violin_matrix.shape[1], 5)
                )
                loadings = pca_results.get('loadings')
                if loadings is not None and 'PC1' in loadings.columns:
                    loadings_for_sort = loadings
                    default_violin = loadings['PC1'].abs().idxmax()
            except Exception:
                if violin_options:
                    default_violin = violin_options[0]['value']
        
        if not default_violin and violin_options:
             default_violin = violin_options[0]['value']

        if loadings_for_sort is not None and 'PC1' in loadings_for_sort.columns:
            pc1_sorted = loadings_for_sort['PC1'].abs().sort_values(ascending=False)
            option_map = {opt['value']: opt for opt in compound_options}
            violin_options = [option_map[val] for val in pc1_sorted.index if val in option_map]
        
        from dash import callback_context
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        user_selected = triggered == 'violin-comp-checks'
        
        selected_compound = None
        if user_selected and violin_comp_checks:
             # Handle potential legacy list value or single value
             if isinstance(violin_comp_checks, list):
                  selected_compound = violin_comp_checks[0] if violin_comp_checks else default_violin
             else:
                  selected_compound = violin_comp_checks
        else:
             # Use current selection if valid and not triggering a reset, otherwise default
             if violin_comp_checks and not isinstance(violin_comp_checks, list) and violin_comp_checks in violin_matrix.columns:
                 selected_compound = violin_comp_checks
             else:
                 selected_compound = default_violin
        
        graphs = []
        if selected_compound and selected_compound in violin_matrix.columns:
            selected = selected_compound
            melt_df = violin_matrix[[selected]].join(group_series).reset_index().rename(columns={
                'ms_file_label': 'Sample',
                group_series.name: group_label,
                selected: 'Intensity',
            })
            metric_label = 'Concentration' if metric == 'scalir_conc' else 'Intensity'
            if norm_value == 'none':
                melt_df['PlotValue'] = melt_df['Intensity']
                y_label = metric_label
            else:
                melt_df['PlotValue'] = melt_df['Intensity']
                y_label = metric_label
            fig = px.violin(
                melt_df,
                x=group_label,
                y='PlotValue',
                color=group_label,
                color_discrete_map=color_map if color_map else None,
                box=False,
                points='all',
                hover_data=['Sample', group_label, 'Intensity', 'PlotValue'],
            )
            fig.update_traces(jitter=0.25, meanline_visible=False, pointpos=-0.5, selector=dict(type='violin'))
            # Clamp KDE tails with spanmode='hard', similar to seaborn cut; use 1st-99th percentiles
            low, high = (
                melt_df['PlotValue'].quantile(0.01),
                melt_df['PlotValue'].quantile(0.99),
            )
            fig.update_traces(spanmode='hard', span=[low, high], side='positive', scalemode='width',
                              selector=dict(type='violin'))
            # Simple significance test: t-test for 2 groups, ANOVA for >2
            groups = [
                g['PlotValue'].dropna().to_numpy()
                for _, g in melt_df.groupby(group_label)
            ]
            groups = [g for g in groups if len(g) >= 2]
            method = None
            p_val = None
            if len(groups) == 2:
                method = "t-test"
                _, p_val = ttest_ind(groups[0], groups[1], equal_var=False, nan_policy='omit')
            elif len(groups) > 2:
                method = "ANOVA"
                _, p_val = f_oneway(*groups)
            if method and p_val is not None and np.isfinite(p_val):
                display_p = f"{p_val:.3e}"
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=1.0,
                    y=1.08,
                    xanchor="right",
                    yanchor="top",
                    text=f"{method}, p={display_p}",
                    showarrow=False,
                    font=dict(size=12, color="#444"),
                )
            fig.update_layout(
                title=f"{selected}",
                title_font=dict(size=16),
                yaxis_title=y_label,
                xaxis_title=group_label,
                yaxis=dict(range=[0, None] if norm_value == 'none' else [None, None], fixedrange=False),
                margin=dict(l=60, r=20, t=50, b=60),
                legend=dict(
                        title=dict(text=f"{group_label}<br>", font=dict(size=14)),
                        font=dict(size=12),
                    ),
                xaxis_title_font=dict(size=16),
                yaxis_title_font=dict(size=16),
                xaxis_tickfont=dict(size=12),
                yaxis_tickfont=dict(size=12),
                template='plotly_white',
                paper_bgcolor='white',
                plot_bgcolor='white',
                clickmode='event+select'
            )
            graphs.append(dcc.Graph(
                id='violin-plot-main',
                figure=fig, 
                style={'marginBottom': 20, 'width': '100%'},
                config=PLOTLY_HIGH_RES_CONFIG
            ))
        return dash.no_update, dash.no_update, graphs, violin_options, selected_compound
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

def callbacks(app, fsc, cache):
    @app.callback(
        Output("analysis-notifications-container", "children"),
        Input('section-context', 'data'),
        Input("wdir", "data"),
        prevent_initial_call=True,
    )
    def warn_missing_workspace(section_context, wdir):
        if not section_context or section_context.get('page') != 'Analysis':
            return dash.no_update
        if not wdir:
            return fac.AntdNotification(
                message="Activate a workspace",
                description="Please select or create a workspace first.",
                type="warning",
                duration=4,
                placement='bottom',
                showProgress=True,
                stack=True,
            )
        
        # Check if results table is empty
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return []
            results_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            if results_count == 0:
                return fac.AntdNotification(
                    message="No data available",
                    description="Please run MINT processing first by navigating to the Processing tab and clicking Run MINT.",
                    type="info",
                    duration=6,
                    placement='bottom',
                    showProgress=True,
                    stack=True,
                )
        
        return []




    @app.callback(
        Output('analysis-metric-container', 'style'),
        Input('analysis-tabs', 'activeKey'),
        prevent_initial_call=False,
    )
    def toggle_metric_visibility(active_tab):
        # SCALiR tab has been moved to Processing plugin
        return {}


    @app.callback(
        Output('analysis-normalization-select', 'value', allow_duplicate=True),
        Input('analysis-tabs', 'activeKey'),
        prevent_initial_call=True,
    )
    def set_norm_default_for_tab(active_tab):
        return TAB_DEFAULT_NORM.get(active_tab, 'zscore')


    @app.callback(
        Output('analysis-tour', 'current'),
        Output('analysis-tour', 'open'),
        Output('analysis-tour', 'steps'),
        Input('analysis-tabs', 'activeKey'),
        Input('analysis-tour-icon', 'nClicks'),
        State('analysis-tour', 'open'),
        prevent_initial_call=False,
    )
    def analysis_tour_open(active_tab, n_clicks, is_open):
        ctx = dash.callback_context
        steps = _analysis_tour_steps(active_tab)
        if not ctx.triggered:
            return 0, False, steps
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'analysis-tour-icon':
            return 0, True, steps
        if trigger == 'analysis-tabs' and is_open:
            return 0, True, steps
        return dash.no_update, dash.no_update, steps

    app.callback(
        Output('bar-graph-matplotlib', 'src'),
        Output('pca-graph', 'figure'),
        Output('violin-graphs', 'children'),
        Output('violin-comp-checks', 'options'),
        Output('violin-comp-checks', 'value'),

        Input('section-context', 'data'),
        Input('analysis-tabs', 'activeKey'),
        Input('pca-x-comp', 'value'),
        Input('pca-y-comp', 'value'),
        Input('violin-comp-checks', 'value'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('clustermap-regenerate-btn', 'nClicks'),
        State('clustermap-fontsize-x-slider', 'value'),
        State('clustermap-fontsize-y-slider', 'value'),
        State("wdir", "data"),
        prevent_initial_call=True,

    )(show_tab_content)


    @app.callback(
        Output('clustermap-spinner', 'spinning'),
        Input('analysis-tabs', 'activeKey'),
        Input('bar-graph-matplotlib', 'src'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        prevent_initial_call=True,
    )
    def toggle_clustermap_spinner(active_tab, bar_src, metric_value, norm_value):
        from dash import callback_context

        if active_tab != 'clustermap':
            return False

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        # When user switches to clustermap tab or changes metric/normalization, force spinner on even if previous image exists
        if trigger in ('analysis-tabs', 'analysis-metric-select', 'analysis-normalization-select'):
            return True

        # Otherwise, keep spinning until image src is set
        return bar_src is None

    @app.callback(
        Output('violin-chromatogram', 'figure'),
        Output('violin-chromatogram-container', 'style'),
        Input('violin-plot-main', 'clickData'),
        Input('violin-comp-checks', 'value'),
        State('analysis-grouping-select', 'value'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def update_chromatogram_on_click(clickData, peak_label, group_by_col, wdir):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # If the compound selection changed, hide the chromatogram
        if trigger_id == 'violin-comp-checks':
            return go.Figure(), {'display': 'none'}

        if not clickData or not wdir or not peak_label:
            return dash.no_update, dash.no_update

        try:
            ms_file_label = clickData['points'][0]['customdata'][0]
        except (KeyError, IndexError, TypeError):
            return dash.no_update, dash.no_update

        # Fetch data
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update, dash.no_update
            
            # 1. Get RT span info for the target
            rt_info = conn.execute("SELECT rt_min, rt_max FROM targets WHERE peak_label = ?", [peak_label]).fetchone()
            rt_min, rt_max = rt_info if rt_info else (None, None)

            # 2. Identify neighbors if a grouping column is selected
            neighbor_files = []
            if group_by_col:
                # Get the group value for the clicked sample
                # Use query formatting for column name (safe as it comes from app options)
                try:
                    group_val_query = f'SELECT "{group_by_col}" FROM samples WHERE ms_file_label = ?'
                    group_val = conn.execute(group_val_query, [ms_file_label]).fetchone()
                    
                    if group_val:
                        group_val = group_val[0]
                        # Fetch up to 10 random other samples from the same group
                        neighbors_query = f"""
                            SELECT ms_file_label, color 
                            FROM samples 
                            WHERE "{group_by_col}" = ? AND ms_file_label != ? 
                            ORDER BY random() 
                            LIMIT 10
                        """
                        neighbor_files = conn.execute(neighbors_query, [group_val, ms_file_label]).fetchall()
                except Exception as e:
                    logger.warning(f"Failed to fetch neighbors: {e}")
                    pass

            # 3. Fetch chromatograms
            # We need the clicked sample + neighbors
            files_to_fetch = [ms_file_label] + [n[0] for n in neighbor_files]
            placeholders = ','.join(['?'] * len(files_to_fetch))
            
            chrom_query = f"""
                SELECT c.ms_file_label, c.scan_time, c.intensity, s.color
                FROM chromatograms c
                JOIN samples s ON c.ms_file_label = s.ms_file_label
                WHERE c.peak_label = ? AND c.ms_file_label IN ({placeholders})
            """
            
            chrom_data = conn.execute(chrom_query, [peak_label] + files_to_fetch).fetchall()
            
            if not chrom_data:
                fig = go.Figure()
                fig.add_annotation(text="No chromatogram data found", showarrow=False)
                return fig, {'display': 'block'}

            # Organize data
            # chrom_data: [(ms_file_label, scan_time, intensity, color), ...]
            data_map = {row[0]: row for row in chrom_data}
            
            fig = go.Figure()

            # Plot neighbors first (background)
            for n_label, n_color in neighbor_files:
                if n_label in data_map:
                    _, scan_times, intensities, _ = data_map[n_label]
                    fig.add_trace(go.Scatter(
                        x=scan_times,
                        y=intensities,
                        mode='lines',
                        name=n_label,
                        line=dict(width=1, color=n_color),
                        opacity=0.4,
                        hoverinfo='skip' 
                    ))

            # Plot clicked sample (foreground)
            if ms_file_label in data_map:
                _, scan_times, intensities, main_color = data_map[ms_file_label]
                fig.add_trace(go.Scatter(
                    x=scan_times,
                    y=intensities,
                    mode='lines',
                    name=ms_file_label,
                    line=dict(width=2, color=main_color),
                    fill='tozeroy',
                    opacity=1.0
                ))
            
            # Add RT span
            if rt_min is not None and rt_max is not None:
                fig.add_vrect(
                    x0=rt_min, x1=rt_max,
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0,
                )
            
            fig.update_layout(
                title=f"{peak_label} | {ms_file_label}",
                xaxis_title="Scan Time (s)",
                yaxis_title="Intensity",
                template="plotly_white",
                margin=dict(l=60, r=20, t=50, b=50),
                height=400,
                showlegend=False
            )
            return fig, {'display': 'block'}
