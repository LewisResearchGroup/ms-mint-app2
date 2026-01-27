"""Analysis plugin orchestrator."""

import base64
from pathlib import Path
from io import BytesIO, StringIO
import logging
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
import feffery_antd_components as fac
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from ...plugin_interface import PluginInterface
from ...duckdb_manager import duckdb_connection, get_physical_cores
from ._shared import (
    METRIC_OPTIONS, NORM_OPTIONS, GROUP_SELECT_OPTIONS, TAB_DEFAULT_NORM,
    GROUPING_FIELDS, GROUP_LABELS, GROUP_COLUMNS,
    rocke_durbin, _build_color_map, _clean_numeric, _create_pivot_custom,
    create_invisible_figure
)
from . import qc, pca, tsne, violin, bar, clustermap

logger = logging.getLogger(__name__)

# Analysis menu items for sidebar
ANALYSIS_MENU_ITEMS = [
    {'component': 'Item', 'props': {'key': 'qc', 'title': 'QC', 'icon': 'antd-check-circle'}},
    {'component': 'Item', 'props': {'key': 'pca', 'title': 'PCA', 'icon': 'antd-dot-chart'}},
    {'component': 'Item', 'props': {'key': 'tsne', 'title': 't-SNE', 'icon': 'antd-deployment-unit'}},
    {'component': 'Item', 'props': {'key': 'raincloud', 'title': 'Violin', 'icon': 'antd-control'}},
    {'component': 'Item', 'props': {'key': 'bar', 'title': 'Bar', 'icon': 'antd-bar-chart'}},
    {'component': 'Item', 'props': {'key': 'clustermap', 'title': 'Clustermap', 'icon': 'antd-build'}},
]

_layout = fac.AntdLayout(
    [
        fac.AntdHeader(
            [
                fac.AntdFlex(
                    [
                        fac.AntdTitle(
                            'Analysis', level=4, style={'margin': '0', 'whiteSpace': 'nowrap'}
                        ),
                        fac.AntdTooltip(
                            fac.AntdIcon(
                                id='analysis-tour-icon',
                                icon='pi-info',
                                style={"cursor": "pointer", 'paddingLeft': '10px'},
                                **{'aria-label': 'Show tutorial'},
                            ),
                            title='Show tutorial'
                        ),
                    ],
                    align='center',
                ),
            ],
            style={'background': 'white', 'padding': '0px', 'lineHeight': '32px', 'height': '32px'}
        ),
        fac.AntdLayout(
            [
                fac.AntdSider(
                    [
                        fac.AntdFlex(
                            [
                                fac.AntdMenu(
                                    id='analysis-sidebar-menu',
                                    menuItems=ANALYSIS_MENU_ITEMS,
                                    mode='inline',
                                    defaultSelectedKey='qc',
                                    style={'border': 'none'},
                                ),
                            ],
                            vertical=True,
                            style={'height': '100%', 'paddingTop': '8px'}
                        ),
                        html.Div(
                            fac.AntdTooltip(
                                fac.AntdButton(
                                    id='analysis-sidebar-collapse',
                                    type='text',
                                    icon=fac.AntdIcon(
                                        id='analysis-sidebar-collapse-icon',
                                        icon='antd-left',
                                        style={'fontSize': '14px'},
                                    ),
                                    shape='default',
                                    **{'aria-label': 'Collapse/Expand Analysis Sidebar'},
                                ),
                                title='Collapse/Expand Analysis Sidebar'
                            ),
                            style={
                                'position': 'absolute',
                                'zIndex': 1,
                                'right': -8,
                                'bottom': 16,
                                'boxShadow': '2px 2px 5px 1px rgba(0,0,0,0.5)',
                                'background': 'white',
                                'borderRadius': '4px',
                            },
                        ),
                    ],
                    id='analysis-sidebar',
                    collapsible=True,
                    collapsed=False,
                    collapsedWidth=60,
                    width=180,
                    trigger=None,
                    style={'height': '100%', 'background': 'white'},
                    className="sidebar-mint"
                ),
                html.Div(
                    [
                        # Shared controls header - visible for all views
                        fac.AntdFlex(
                            [
                                html.Div(
                                    fac.AntdSpace(
                                        [
                                            fac.AntdText("Metric:", style={'fontWeight': 500, 'marginLeft': '15px'}),
                                            fac.AntdSelect(
                                                id='analysis-metric-select',
                                                options=METRIC_OPTIONS,
                                                value='peak_area',
                                                optionFilterProp='label',
                                                optionFilterMode='case-insensitive',
                                                allowClear=False,
                                                style={'width': 160},
                                            ),
                                        ],
                                        align='center',
                                        size='small',
                                    ),
                                    id='analysis-metric-wrapper',
                                    style={'display': 'flex'},
                                ),
                                html.Div(
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
                                                style={'width': 160},
                                            ),
                                        ],
                                        align='center',
                                        size='small',
                                    ),
                                    id='analysis-normalization-wrapper',
                                    style={'display': 'flex'},
                                ),
                                fac.AntdSpace(
                                    [
                                        fac.AntdText("Group by:", style={'fontWeight': 500}),
                                        fac.AntdSelect(
                                            id='analysis-grouping-select',
                                            options=GROUP_SELECT_OPTIONS,
                                            value='sample_type',
                                            allowClear=False,
                                            style={'width': 160},
                                        ),
                                    ],
                                    align='center',
                                    size='small',
                                ),
                                html.Div(
                                    fac.AntdSpace(
                                        [
                                            fac.AntdText("Target:", style={'fontWeight': 500}),
                                            fac.AntdSelect(
                                                id='qc-target-select',
                                                options=[],
                                                value=None,
                                                allowClear=False,
                                                optionFilterProp='label',
                                                optionFilterMode='case-insensitive',
                                                style={'width': 350},
                                            ),
                                        ],
                                        align='center',
                                        size='small',
                                    ),
                                    id='qc-target-wrapper',
                                    style={'display': 'flex'},
                                ),
                            ],
                            wrap=True,
                            gap='middle',
                            align='center',
                            id='analysis-metric-container',
                            style={'padding': '12px 16px', 'borderBottom': '1px solid #f0f0f0'},
                        ),
                        # QC content
                        html.Div(
                            qc.create_layout(),
                            id='analysis-qc-container',
                            style={'display': 'block', 'padding': '16px'}
                        ),
                        # PCA content
                        html.Div(
                            pca.create_layout(),
                            id='analysis-pca-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # t-SNE content
                        html.Div(
                            tsne.create_layout(),
                            id='analysis-tsne-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # Violin content
                        html.Div(
                            violin.create_layout(),
                            id='analysis-violin-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # Bar content
                        html.Div(
                            bar.create_layout(),
                            id='analysis-bar-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # Clustermap content
                        html.Div(
                            clustermap.create_layout(),
                            id='analysis-clustermap-container',
                            style={'display': 'none', 'padding': '16px', 'height': 'calc(100vh - 150px)'}
                        ),
                    ],
                    className='ant-layout-content css-1v28nim',
                    style={'background': 'white', 'overflowY': 'auto', 'height': 'calc(100vh - 80px)'}
                ),
            ],
            style={'height': 'calc(100vh - 100px)', 'background': 'white'},
        ),
        # Hidden store for maintaining tab state compatibility with callbacks
        dcc.Store(id='analysis-tabs', data={'activeKey': 'qc'}),
        fac.AntdTour(
            locale='en-us',
            steps=[],
            id='analysis-tour',
            open=False,
            current=0,
        ),
        html.Div(id="analysis-notifications-container"),
    ]
)

class AnalysisPlugin(PluginInterface):
    def __init__(self):
        self._layout = _layout

    def layout(self):
        return self._layout

    def callbacks(self, app, fsc=None, cache=None):
        callbacks(app, fsc, cache)

    def outputs(self):
        return None

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

def _analysis_tour_steps(active_tab: str):
    # Common steps for all views
    common_steps = [
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
            'title': 'Analysis Types',
            'description': 'Switch between PCA, Violin, and Clustermap views.',
            'targetSelector': '#analysis-sidebar-menu',
        },
    ]

    # QC-specific override for common steps
    if active_tab == 'qc':
        common_steps = [
            {
                'title': 'Analysis overview',
                'description': 'Use the controls above to group samples and select targets for QC.',
            },
            {
                'title': 'Group by',
                'description': 'Group samples for coloring in the plots.',
                'targetSelector': '#analysis-grouping-select',
            },
            {
                'title': 'Select Target',
                'description': 'Choose the specific target compound to visualize in the QC plots.',
                'targetSelector': '[id="qc-target-select"]', # Use attribute selector for component IDs
            },
            {
                'title': 'Analysis Types',
                'description': 'Switch between different analysis views.',
                'targetSelector': '#analysis-sidebar-menu',
            },
        ]
    
    # View-specific steps
    if active_tab == 'qc':
        view_steps = [
            {
                'title': 'QC Scatter Plots',
                'description': 'View Retention Time (RT) and Peak Area drifts over acquisition order. Useful for identifying run order effects.',
                'targetSelector': '#qc-spinner',
            },
            {
                'title': 'Chromatogram',
                'description': 'See the raw chromatogram for the selected sample. Click on a point in the scatter plots to select a sample.',
                'targetSelector': '#qc-chromatogram-container',
            },
        ]
    elif active_tab == 'pca':
        view_steps = [
            {
                'title': 'PCA Axes',
                'description': 'Select which principal components to display on the X and Y axes.',
                'targetSelector': '#pca-x-comp',
            },
            {
                'title': 'PCA Plot',
                'description': 'Interactive scatter plot showing samples in PCA space. Hover for details, use the legend to filter groups.',
                'targetSelector': '#pca-graph',
            },

        ]
    elif active_tab == 'tsne':
        view_steps = [
            {
                'title': 't-SNE Axes',
                'description': 'Select which t-SNE dimensions to display.',
                'targetSelector': '#tsne-x-comp',
            },
            {
                'title': 't-SNE Plot',
                'description': 'Interactive scatter plot showing samples in t-SNE space.',
                'targetSelector': '#tsne-graph',
            },
        ]
    elif active_tab == 'raincloud':
        view_steps = [
            {
                'title': 'Target Selection',
                'description': 'Choose which target compound to display in the violin plot.',
                'targetSelector': '#violin-comp-checks',
            },
            {
                'title': 'Violin Plot',
                'description': 'Shows the distribution of values for each group. Click on individual points to see the chromatogram.',
                'targetSelector': '#violin-graphs',
            },
            {
                'title': 'Chromatogram',
                'description': 'View the chromatogram for the selected sample.',
                'targetSelector': '#violin-chromatogram-container',
            },
        ]
    elif active_tab == 'bar':
        view_steps = [
            {
                'title': 'Target Selection',
                'description': 'Choose which target compound to display in the bar plot.',
                'targetSelector': '#bar-comp-checks',
            },
            {
                'title': 'Bar Plot',
                'description': 'Compare mean values across groups with error bars. Click on a bar to select a sample from that group.',
                'targetSelector': '#bar-graphs',
            },
            {
                'title': 'Chromatogram',
                'description': 'View the chromatogram for the selected sample.',
                'targetSelector': '#bar-chromatogram-container',
            },
        ]
    elif active_tab == 'clustermap':
        view_steps = [
            {
                'title': 'Font Size Controls',
                'description': 'Adjust the font size for row and column labels.',
                'targetSelector': '#clustermap-fontsize-x-slider',
            },
            {
                'title': 'Regenerate',
                'description': 'Click to regenerate the clustermap with current settings.',
                'targetSelector': '#clustermap-regenerate-btn',
            },
            {
                'title': 'Clustermap',
                'description': 'Heatmap with hierarchical clustering of samples and targets.',
                'targetSelector': '#clustermap-image',
            },
        ]
    else:
        view_steps = []
    
    return common_steps + view_steps


def callbacks(app, fsc=None, cache=None):

    # Register tab-specific callbacks
    qc.register_callbacks(app)
    pca.register_callbacks(app)
    tsne.register_callbacks(app)
    violin.register_callbacks(app)
    bar.register_callbacks(app)
    clustermap.register_callbacks(app)

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
            try:
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
            except Exception:
                pass
        
        return []

    # Sidebar collapse toggle callback
    @app.callback(
        Output('analysis-sidebar', 'collapsed'),
        Output('analysis-sidebar-collapse-icon', 'icon'),
        Input('analysis-sidebar-collapse', 'nClicks'),
        State('analysis-sidebar', 'collapsed'),
        prevent_initial_call=True,
    )
    def toggle_analysis_sidebar(n_clicks, is_collapsed):
        if n_clicks is None:
            raise PreventUpdate
        new_collapsed = not is_collapsed
        icon = 'antd-right' if new_collapsed else 'antd-left'
        return new_collapsed, icon

    # Menu selection -> content visibility + sync to analysis-tabs store
    @app.callback(
        Output('analysis-qc-container', 'style'),
        Output('analysis-pca-container', 'style'),
        Output('analysis-tsne-container', 'style'),
        Output('analysis-violin-container', 'style'),
        Output('analysis-bar-container', 'style'),
        Output('analysis-clustermap-container', 'style'),
        Output('analysis-tabs', 'data'),
        Input('analysis-sidebar-menu', 'currentKey'),
        prevent_initial_call=False,
    )
    def update_analysis_content_visibility(current_key):
        # Default to 'qc' if no key selected (QC is now the first tab)
        active_key = current_key or 'qc'
        
        qc_style = {'display': 'block', 'padding': '16px'} if active_key == 'qc' else {'display': 'none', 'padding': '16px'}
        pca_style = {'display': 'block', 'padding': '16px'} if active_key == 'pca' else {'display': 'none', 'padding': '16px'}
        tsne_style = {'display': 'block', 'padding': '16px'} if active_key == 'tsne' else {'display': 'none', 'padding': '16px'}
        violin_style = {'display': 'block', 'padding': '16px'} if active_key == 'raincloud' else {'display': 'none', 'padding': '16px'}
        bar_style = {'display': 'block', 'padding': '16px'} if active_key == 'bar' else {'display': 'none', 'padding': '16px'}
        # Clustermap needs explicit height for spinner centering to work on first access
        clustermap_style = {
            'display': 'block' if active_key == 'clustermap' else 'none',
            'padding': '16px',
            'height': 'calc(100vh - 150px)',  # Explicit height for flexbox centering
        }
        
        # Sync to legacy analysis-tabs store for backward compatibility with other callbacks
        tabs_data = {'activeKey': active_key}
        
        return qc_style, pca_style, tsne_style, violin_style, bar_style, clustermap_style, tabs_data

    @app.callback(
        Output('analysis-metric-wrapper', 'style'),
        Output('analysis-normalization-wrapper', 'style'),
        Input('analysis-sidebar-menu', 'currentKey'),
        prevent_initial_call=False,
    )
    def toggle_metric_visibility(active_tab):
        # For QC tab, hide Metric and Transformations (show only Group by)
        if active_tab == 'qc':
            hidden_style = {'display': 'none'}
            return hidden_style, hidden_style
        # For other tabs, show all controls
        visible_style = {'display': 'flex'}
        return visible_style, visible_style

    @app.callback(
        Output('analysis-normalization-select', 'value', allow_duplicate=True),
        Input('analysis-sidebar-menu', 'currentKey'),
        prevent_initial_call=True,
    )
    def set_norm_default_for_tab(active_tab):
        return TAB_DEFAULT_NORM.get(active_tab, 'zscore')

    @app.callback(
        Output('analysis-tour', 'current'),
        Output('analysis-tour', 'open'),
        Output('analysis-tour', 'steps'),
        Input('analysis-sidebar-menu', 'currentKey'),
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
        if trigger == 'analysis-sidebar-menu' and is_open:
            return 0, True, steps
        return dash.no_update, dash.no_update, steps

    @app.callback(
        Output('clustermap-image', 'src'),
        Output('pca-graph', 'figure'),
        Output('tsne-graph', 'figure'),
        Output('violin-graphs', 'children'),
        Output('violin-comp-checks', 'options'),
        Output('violin-comp-checks', 'value'),
        Output('bar-graphs', 'children'),
        Output('bar-comp-checks', 'options'),
        Output('bar-comp-checks', 'value'),

        Input('section-context', 'data'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('pca-x-comp', 'value'),
        Input('pca-y-comp', 'value'),
        Input('violin-comp-checks', 'value'),
        Input('bar-comp-checks', 'value'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('clustermap-regenerate-btn', 'nClicks'),
        Input('tsne-regenerate-btn', 'nClicks'),
        Input('clustermap-cluster-rows', 'checked'),
        Input('clustermap-cluster-cols', 'checked'),
        Input('clustermap-fontsize-x-slider', 'value'),
        Input('clustermap-fontsize-y-slider', 'value'),
        Input('wdir', 'data'),
        Input('tsne-x-comp', 'value'),
        Input('tsne-y-comp', 'value'),
        Input('tsne-perplexity-slider', 'value'),
        prevent_initial_call=False,
    )
    def update_content(section_context, tab_key, x_comp, y_comp, violin_comp_checks, bar_comp_checks, metric_value, norm_value,
                        group_by, regen_clicks, tsne_regen_clicks, cluster_rows, cluster_cols, fontsize_x, fontsize_y, wdir,
                        tsne_x_comp, tsne_y_comp, tsne_perplexity):
        
        if not section_context or section_context.get('page') != 'Analysis':
            raise PreventUpdate
        
        # Prevent double-firing when switching tabs forces a normalization update.
        from dash import callback_context
        triggered_props = [t['prop_id'] for t in callback_context.triggered] if callback_context.triggered else []
        if 'analysis-sidebar-menu.currentKey' in triggered_props:
            default_norm = TAB_DEFAULT_NORM.get(tab_key)
            if default_norm and norm_value != default_norm:
                raise PreventUpdate
        
        if not wdir:
            raise PreventUpdate
        
        invisible_fig = create_invisible_figure()
        
        from dash import callback_context
        ctx = callback_context

        grouping_fields = GROUPING_FIELDS
        selected_group = group_by if group_by in grouping_fields else GROUPING_FIELDS[0]
        
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return None, invisible_fig, invisible_fig, [], [], [], [], [], []
            try:
                results_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
                if results_count == 0:
                    return None, invisible_fig, invisible_fig, [], [], [], [], [], []
            except Exception:
                return None, invisible_fig, invisible_fig, [], [], [], [], [], []
            
            # Robust metric selection
            metric = 'peak_area'
            if metric_value == 'scalir_conc' or metric_value in allowed_metrics:
                metric = metric_value
            
            target_table = 'results'
            if metric == 'scalir_conc':
                scalir_path = Path(wdir) / "results" / "scalir" / "concentrations.csv"
                if not scalir_path.exists():
                    return None, invisible_fig, invisible_fig, [], [], [], [], [], []
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
                    return None, invisible_fig, invisible_fig, [], [], [], [], [], []

            df = _create_pivot_custom(conn, value=metric, table=target_table)
            df.set_index('ms_file_label', inplace=True)
            
            group_field = selected_group if selected_group in df.columns else (
                'sample_type' if 'sample_type' in df.columns else None
            )
            group_label = GROUP_LABELS.get(group_field, 'Group')
            missing_group_label = f"{group_label} (unset)"
            metadata_cols = [col for col in ['ms_type'] + grouping_fields if col in df.columns]
            
            order_df_raw = conn.execute(
                "SELECT ms_file_label FROM samples ORDER BY ms_file_label"
            ).df()
            if order_df_raw.empty:
                return None, invisible_fig, invisible_fig, [], [], [], [], [], []
            
            order_df = order_df_raw["ms_file_label"].tolist()
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
                return None, invisible_fig, invisible_fig, [], [], [], [], [], []
            
            from ...pca import StandardScaler
            scaler = StandardScaler()
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
            
            # Choose matrix for violin/bar based on normalization selection
            if norm_value == 'zscore':
                violin_matrix = zdf
            elif norm_value in ('durbin', 'zscore_durbin'):
                violin_matrix = ndf
            else:
                violin_matrix = df
            
            if violin_matrix.empty:
                return dash.no_update, invisible_fig, invisible_fig, [], [], [], [], [], []

        # Route logic to submodules
        if tab_key == 'clustermap':
             triggered_prop = triggered_props[0].split('.')[0] if triggered_props else None
             src = clustermap.generate_clustermap(zdf, color_labels, color_map, group_label, norm_value, cluster_rows, cluster_cols, fontsize_x, fontsize_y, wdir, metric, triggered_prop, norm_value)
             return src, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        elif tab_key == 'pca':
             fig = pca.generate_pca_figure(ndf, color_labels, color_map, group_label, x_comp, y_comp)
             return dash.no_update, fig, dash.no_update, dash.no_update, compound_options, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        elif tab_key == 'tsne':
             fig = tsne.generate_tsne_figure(ndf, color_labels, color_map, group_label, tsne_x_comp, tsne_y_comp, tsne_perplexity)
             return dash.no_update, dash.no_update, fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        elif tab_key == 'raincloud':
             graphs, options, val = violin.generate_violin_plots(violin_matrix, group_series, color_map, group_label, metric, norm_value, violin_comp_checks, compound_options)
             return dash.no_update, dash.no_update, dash.no_update, graphs, options, val, dash.no_update, dash.no_update, dash.no_update

        elif tab_key == 'bar':
             graphs, options, val = bar.generate_bar_plots(violin_matrix, group_series, color_map, group_label, metric, norm_value, bar_comp_checks, compound_options)
             return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, graphs, options, val

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
