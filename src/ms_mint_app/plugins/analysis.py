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
from sklearn.decomposition import PCA
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
            dcc.Graph(id='pca-graph'),
            text='Loading PCA...',
        ),
    ]
)

scalir_tab = dcc.Loading(
    children=html.Div(
    [
        fac.AntdAlert(
            message="SCALiR calibration",
            description=html.Div(
                [
                    html.Div("Load a standards table and fit calibration curves using current workspace results."),
                ]
            ),
            type="info",
            showIcon=True,
            style={'marginBottom': 12},
        ),
        fac.AntdSpace(
            [
                fac.AntdText("Params:", strong=True, style={'marginRight': 8}),
                fac.AntdSelect(
                    id='scalir-intensity',
                    options=[
                        {'label': 'Peak Area', 'value': 'peak_area'},
                        {'label': 'Peak Max', 'value': 'peak_max'},
                    ],
                    value='peak_area',
                    style={'width': 160},
                ),
                fac.AntdSelect(
                    id='scalir-slope-mode',
                    options=[
                        {'label': 'Fixed slope', 'value': 'fixed'},
                        {'label': 'Constrained slope', 'value': 'interval'},
                        {'label': 'Free slope', 'value': 'wide'},
                    ],
                    value='fixed',
                    style={'width': 180},
                ),
                fac.AntdInputNumber(
                    id='scalir-slope-low',
                    value=0.85,
                    min=0.1,
                    max=5,
                    step=0.01,
                    placeholder='Slope low',
                    style={'width': 110},
                ),
                fac.AntdInputNumber(
                    id='scalir-slope-high',
                    value=1.15,
                    min=0.1,
                    max=5,
                    step=0.01,
                    placeholder='Slope high',
                    style={'width': 110},
                ),
                fac.AntdSwitch(
                    id='scalir-generate-plots',
                    checked=False,
                    checkedChildren='Save plots',
                    unCheckedChildren='Skip plots',
                ),
                fac.AntdButton(
                    'Run SCALiR',
                    id='scalir-run-btn',
                    type='primary',
                    style={'minWidth': 110},
                ),
                fac.AntdButton(
                    'Clear',
                    id='scalir-reset-btn',
                    type='default',
                    danger=False,
                ),
            ],
            wrap=True,
            size='small',
            style={'marginBottom': 12},
        ),
        html.Div(
            [
                fac.AntdSpace(
                    [
                        fac.AntdText(
                            'Standards file (CSV):',
                            strong=True,
                            style={'margin': 0, 'lineHeight': '32px'},
                        ),
                        html.Div(
                            [
                                dcc.Upload(
                                    id='scalir-standards-upload',
                                    children=fac.AntdButton(
                                        'Select standards file',
                                        icon=fac.AntdIcon(icon='antd-upload'),
                                        type='default',
                                        block=True,
                                        style={
                                            'width': '180px',
                                            'height': '36px',
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'center',
                                            'gap': 6,
                                        },
                                    ),
                                    style={'padding': 0},
                                    multiple=False,
                                ),
                                fac.AntdText(
                                    id='scalir-standards-note',
                                    style={'color': '#555', 'fontSize': 12, 'lineHeight': '32px', 'marginLeft': 8},
                                ),
                            ],
                            style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'},
                        ),
                    ],
                    align='center',
                    size='small',
                    style={'width': '100%', 'marginTop': 4},
                ),
                html.Div(id='scalir-status-text', style={'marginTop': 6}),
                fac.AntdSpace(
                    [
                        fac.AntdText(id='scalir-conc-path'),
                        fac.AntdText(id='scalir-params-path'),
                    ],
                    direction='vertical',
                    size='small',
                    style={'marginTop': 8},
                ),
            ],
            style={'marginBottom': 12, 'padding': '0 0px'},
        ),
        fac.AntdSpace(
            [
                fac.AntdSelect(
                    id='scalir-metabolite-select',
                    options=[],
                    value=[],
                    mode='multiple',
                    allowClear=False,
                    maxTagCount=3,
                    placeholder='Select metabolite(s)',
                    style={'width': 320},
                ),
                fac.AntdText(id='scalir-plot-path'),
            ],
            align='center',
            style={'marginBottom': 12},
        ),
        html.Div(
            id='scalir-plot-graphs',
            style={
                'display': 'none',
                'flexWrap': 'wrap',
                'gap': '16px',
                'paddingTop': '8px',
                'justifyContent': 'flex-start',
            },
        ),
        dcc.Store(id='scalir-results-store'),
    ],
    style={'overflow': 'auto', 'height': 'calc(100vh - 156px)'}
    ),
    type='default',
    id='scalir-loading',
)

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
                {'key': 'clustermap', 'label': 'Clustermap', 'children': clustermap_tab},
                {'key': 'pca', 'label': 'PCA', 'children': pca_tab},
                {
                    'key': 'raincloud',
                    'label': 'Violin',
                    'children': html.Div(
                        [
                            fac.AntdSelect(
                                id='violin-comp-checks',
                                mode='multiple',
                                options=[],
                                value=[],
                                allowClear=False,
                                maxTagCount=4,
                                optionFilterProp='label',
                                optionFilterMode='case-insensitive',
                                style={'width': 360, 'marginBottom': 12},
                            ),
                            html.Div(
                                id='violin-graphs',
                                style={
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'gap': '24px',
                                },
                            ),
                        ]
                    ),
                },
                {'key': 'scalir', 'label': 'SCALiR', 'children': scalir_tab},
            ],
            centered=True,
            defaultActiveKey='clustermap',
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
        margin=dict(l=60, r=20, t=60, b=60),
        template="plotly_white",
    )
    if params_text:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
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
    if active_tab == 'scalir':
        return [
            {
                'title': 'SCALiR overview',
                'description': 'Calibrate concentrations from standards and workspace results.',
            },
            {
                'title': 'Tabs',
                'description': 'Switch between Clustermap, PCA, Violin, and SCALiR views.',
                'targetSelector': '#analysis-tabs',
            },
            {
                'title': 'Params',
                'description': 'Choose the intensity metric and slope strategy.',
                'targetSelector': '#scalir-intensity',
            },
            {
                'title': 'Standards file',
                'description': 'Upload your standards table (CSV).',
                'targetSelector': '#scalir-standards-upload',
            },
            {
                'title': 'Run',
                'description': 'Run SCALiR to fit calibration curves.',
                'targetSelector': '#scalir-run-btn',
            },
            {
                'title': 'Inspect results',
                'description': 'Select metabolites to view calibration plots.',
                'targetSelector': '#scalir-metabolite-select',
            },
        ]
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
            'description': 'Switch between Clustermap, PCA, Violin, and SCALiR views.',
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
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No results available",
        template="plotly_white",
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    from dash import callback_context
    ctx = callback_context
    triggered_prop = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    grouping_fields = GROUPING_FIELDS
    selected_group = group_by if group_by in grouping_fields else GROUPING_FIELDS[0]
    with duckdb_connection(wdir) as conn:
        if conn is None:
            return None, empty_fig, [], [], []
        results_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        if results_count == 0:
            return None, empty_fig, [], [], []
        metric = metric_value if metric_value in allowed_metrics else 'peak_area'
        df = create_pivot(conn, value=metric)
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
            return None, empty_fig, [], [], []
        from sklearn.preprocessing import StandardScaler
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
            return dash.no_update, empty_fig, go.Figure(), [], [], []
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
                    bbox_to_anchor=(1.00, 1.075),
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
        default_violin = []
        if violin_options:
            try:
                pca_results = run_pca_samples_in_cols(
                    violin_matrix,
                    n_components=min(violin_matrix.shape[0], violin_matrix.shape[1], 5)
                )
                loadings = pca_results.get('loadings')
                if loadings is not None and 'PC1' in loadings.columns:
                    loadings_for_sort = loadings
                    default_violin = [loadings['PC1'].abs().idxmax()]
            except Exception:
                default_violin = [violin_options[0]['value']]
        if loadings_for_sort is not None and 'PC1' in loadings_for_sort.columns:
            pc1_sorted = loadings_for_sort['PC1'].abs().sort_values(ascending=False)
            option_map = {opt['value']: opt for opt in compound_options}
            violin_options = [option_map[val] for val in pc1_sorted.index if val in option_map]
        from dash import callback_context
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        user_selected = triggered == 'violin-comp-checks'
        if user_selected:
            # Respect manual picks, but still fall back if empty.
            selected_list = violin_comp_checks or default_violin or ([violin_options[0]['value']] if violin_options else [])
        else:
            # Metric/norm/tab changes refresh to the current PC1 leader, ignoring stale state.
            selected_list = default_violin or ([violin_options[0]['value']] if violin_options else [])
        graphs = []
        for selected in selected_list:
            if selected not in violin_matrix.columns:
                continue
            melt_df = violin_matrix[[selected]].join(group_series).reset_index().rename(columns={
                'ms_file_label': 'Sample',
                group_series.name: group_label,
                selected: 'Intensity',
            })
            if norm_value == 'none':
                melt_df['PlotValue'] = melt_df['Intensity']
                y_label = 'Intensity'
            else:
                melt_df['PlotValue'] = melt_df['Intensity']
                y_label = 'Intensity'
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
            )
            graphs.append(dcc.Graph(figure=fig, style={'marginBottom': 20, 'width': '100%'}))
        return dash.no_update, dash.no_update, graphs, violin_options, selected_list
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

def run_scalir(n_clicks, standards_contents, standards_filename, intensity, slope_mode, slope_low, slope_high,
               generate_plots, wdir, active_tab, section_context):
    if not n_clicks or not section_context or section_context.get('page') != 'Analysis' or active_tab != 'scalir':
        raise PreventUpdate
    hidden_style = {
        'display': 'none',
        'flexWrap': 'wrap',
        'gap': '16px',
        'paddingTop': '8px',
        'justifyContent': 'flex-start',
    }
    if not wdir:
        return ("No active workspace.", "", "", [], [], [], hidden_style, None)
    try:
        logger.info(f"SCALiR: Parsing standards file {standards_filename}...")
        standards_df = _parse_uploaded_standards(standards_contents, standards_filename)
    except Exception as exc:
        logger.error(f"SCALiR: Failed to parse standards file: {exc}")
        return (f"Upload a standards table (CSV). Error: {exc}", "", "", [], [], [], hidden_style, None)
    with duckdb_connection(wdir) as conn:
        if conn is None:
            logger.error("SCALiR: Failed to connect to database.")
            return ("Database connection failed.", "", "", [], [], [], hidden_style, None)
        if intensity not in allowed_metrics:
            intensity = 'peak_area'
        try:
            mint_df = conn.execute(f"""
                SELECT ms_file_label AS ms_file, peak_label, {intensity}
                FROM results
                WHERE {intensity} IS NOT NULL
            """).df()
        except Exception as exc:
            logger.error(f"SCALiR: Could not load results from database: {exc}")
            return (f"Could not load results: {exc}", "", "", [], [], [], hidden_style, None)
    if mint_df.empty:
        return ("No results found for calibration.", "", "", [], [], [], hidden_style, None)
    units_df = None
    if "unit" in standards_df.columns:
        units_df = standards_df[["peak_label", "unit"]].copy()
        standards_df = standards_df.drop(columns=["unit"])
    try:
        mint_filtered, standards_filtered, units_filtered, common = intersect_peaks(
            mint_df, standards_df, units_df
        )
    except Exception as exc:
        logger.error(f"SCALiR: Alignment failed: {exc}", exc_info=True)
        return (f"Could not align standards with results: {exc}", "", "", [], [], [], hidden_style, None)
    if not common:
        return ("No overlapping peak_label values between results and standards.", "", "", [], [], [], hidden_style, None)
    low = slope_low or 0.85
    high = slope_high or 1.15
    slope_interval = (min(low, high), max(low, high))
    try:
        estimator, std_results, x_train, y_train, params = fit_estimator(
            mint_filtered, standards_filtered, intensity, slope_mode or "fixed", slope_interval
        )
        logger.info(f"SCALiR: Fitting completed. Metabolites: {len(common)}")
        concentrations = build_concentration_table(
            estimator, mint_filtered, intensity, units_filtered
        )
    except Exception as exc:
        logger.error(f"SCALiR: Fitting failed: {exc}", exc_info=True)
        return (f"Error fitting calibration: {exc}", "", "", [], [], [], hidden_style, None)
    output_dir = Path(wdir) / "analysis" / "scalir"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    if generate_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
    concentrations_path = output_dir / "concentrations.csv"
    params_path = output_dir / "standard_curve_parameters.csv"
    concentrations.to_csv(concentrations_path, index=False)
    params.to_csv(params_path, index=False)
    try:
        train_frame = training_plot_frame(estimator, x_train, y_train, params)
    except Exception:
        train_frame = pd.DataFrame()
    if generate_plots and not train_frame.empty:
        for label in common:
            plot_standard_curve(train_frame, label, units_filtered, plots_dir)
    metabolite_options = [{'label': label, 'value': label} for label in common]
    first_label = common[0] if common else None
    initial_selection = [first_label] if first_label else []
    plots = []
    for lbl in initial_selection:
        plots.append(
            dcc.Graph(
                figure=_plot_curve_fig(train_frame, lbl, units_filtered),
                style={
                    'flex': '0 0 calc(33.333% - 12px)',
                    'minWidth': '320px',
                    'maxWidth': '520px',
                    'minHeight': '320px',
                },
                config={'displaylogo': False},
            )
        )
    plot_style = {
        'display': 'flex' if plots else 'none',
        'flexWrap': 'wrap',
        'gap': '16px',
        'paddingTop': '8px',
        'justifyContent': 'flex-start',
    }
    store_data = {
        "train_frame": train_frame.to_json(orient="split") if not train_frame.empty else None,
        "units": units_filtered.to_json(orient="split") if units_filtered is not None else None,
        "params": params.to_json(orient="split") if not params.empty else None,
        "plot_dir": str(plots_dir),
        "common": common,
        "generated_all_plots": bool(generate_plots),
    }
    status_text = f"Fitted {len(common)} metabolites with intensity '{intensity}'."
    return (
        status_text,
        f"Concentrations: {concentrations_path}",
        f"Curve parameters: {params_path}",
        metabolite_options,
        initial_selection,
        plots,
        plot_style,
        store_data,
    )

def callbacks(app, fsc, cache):
    @app.callback(
        Output("analysis-notifications-container", "children"),
        Input('section-context', 'data'),
        Input("wdir", "data"),
    )
    def warn_missing_workspace(section_context, wdir):
        if not section_context or section_context.get('page') != 'Analysis':
            return dash.no_update
        if wdir:
            return []
        return fac.AntdNotification(
            message="Activate a workspace",
            description="Please select or create a workspace first.",
            type="warning",
            duration=4,
            placement='bottom',
            showProgress=True,
            stack=True,
        )




    @app.callback(
        Output('analysis-metric-container', 'style'),
        Input('analysis-tabs', 'activeKey'),
        prevent_initial_call=False,
    )
    def toggle_metric_visibility(active_tab):
        if active_tab == 'scalir':
            return {'display': 'none'}
        return {}

    @app.callback(
        Output('scalir-standards-note', 'children'),
        Input('scalir-standards-upload', 'filename'),
        prevent_initial_call=False,
    )
    def show_standards_filename(filename):
        if filename:
            return f"Selected: {filename}"
        return "No standards file selected."

    @app.callback(
        Output('analysis-normalization-select', 'value', allow_duplicate=True),
        Input('analysis-tabs', 'activeKey'),
        prevent_initial_call=True,
    )
    def set_norm_default_for_tab(active_tab):
        return TAB_DEFAULT_NORM.get(active_tab, 'zscore')

    @app.callback(
        Output('scalir-status-text', 'children', allow_duplicate=True),
        Output('scalir-conc-path', 'children', allow_duplicate=True),
        Output('scalir-params-path', 'children', allow_duplicate=True),
        Output('scalir-metabolite-select', 'options', allow_duplicate=True),
        Output('scalir-metabolite-select', 'value', allow_duplicate=True),
        Output('scalir-plot-graphs', 'children', allow_duplicate=True),
        Output('scalir-plot-graphs', 'style', allow_duplicate=True),
        Output('scalir-results-store', 'data', allow_duplicate=True),
        Output('scalir-standards-upload', 'contents', allow_duplicate=True),
        Output('scalir-standards-upload', 'filename', allow_duplicate=True),
        Output('scalir-standards-note', 'children', allow_duplicate=True),
        Input('scalir-reset-btn', 'nClicks'),
        prevent_initial_call=True,
    )
    def reset_scalir(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return (
            "",
            "",
            "",
            [],
            [],
            [],
            {
                'display': 'none',
                'flexWrap': 'wrap',
                'gap': '16px',
                'paddingTop': '8px',
                'justifyContent': 'flex-start',
            },
            None,
            None,
            None,
            "No standards file selected.",
        )



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

    app.callback(
        Output('scalir-status-text', 'children'),
        Output('scalir-conc-path', 'children'),
        Output('scalir-params-path', 'children'),
        Output('scalir-metabolite-select', 'options'),
        Output('scalir-metabolite-select', 'value'),
        Output('scalir-plot-graphs', 'children'),
        Output('scalir-plot-graphs', 'style'),
        Output('scalir-results-store', 'data'),
        Input('scalir-run-btn', 'nClicks'),
        State('scalir-standards-upload', 'contents'),
        State('scalir-standards-upload', 'filename'),
        State('scalir-intensity', 'value'),
        State('scalir-slope-mode', 'value'),
        State('scalir-slope-low', 'value'),
        State('scalir-slope-high', 'value'),
        State('scalir-generate-plots', 'checked'),
        State('wdir', 'data'),
        State('analysis-tabs', 'activeKey'),
        State('section-context', 'data'),
        prevent_initial_call=True,

    )(run_scalir)

    @app.callback(
        Output('scalir-plot-graphs', 'children', allow_duplicate=True),
        Output('scalir-plot-graphs', 'style', allow_duplicate=True),
        Output('scalir-plot-path', 'children'),
        Input('scalir-metabolite-select', 'value'),
        State('scalir-results-store', 'data'),
        prevent_initial_call=True,
    )
    def update_scalir_plot(selected_labels, store_data):
        if not store_data or not selected_labels:
            raise PreventUpdate

        train_frame_json = store_data.get("train_frame")
        if not train_frame_json:
            raise PreventUpdate

        train_frame = pd.read_json(StringIO(train_frame_json), orient="split")
        units_json = store_data.get("units")
        units_df = pd.read_json(StringIO(units_json), orient="split") if units_json else None
        params_json = store_data.get("params")
        params_df = pd.read_json(StringIO(params_json), orient="split") if params_json else None

        labels = selected_labels if isinstance(selected_labels, list) else [selected_labels]
        plots = []
        for lbl in labels:
            plots.append(
                dcc.Graph(
                    figure=_plot_curve_fig(train_frame, lbl, units_df, params_df),
                    style={
                        'flex': '0 0 calc(33.333% - 12px)',
                        'minWidth': '320px',
                        'maxWidth': '520px',
                        'minHeight': '320px',
                    },
                    config={'displaylogo': False},
                )
            )

        plot_dir = Path(store_data.get("plot_dir", ""))
        plot_path = ""
        if labels and store_data.get("generated_all_plots") and plot_dir:
            candidate = plot_dir / f"{slugify_label(labels[0])}_curve.png"
            if candidate.exists():
                plot_path = f"Plot saved at: {candidate}"

        return plots, {
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '16px',
            'paddingTop': '8px',
            'justifyContent': 'flex-start',
        }, plot_path

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
