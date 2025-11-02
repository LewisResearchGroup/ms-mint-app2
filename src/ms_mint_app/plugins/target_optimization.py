import json
from os import cpu_count

import dash
import feffery_antd_components as fac
import math
import numpy as np
import plotly.graph_objects as go
import psutil
import time
from dash import html, dcc, Patch
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from ..duckdb_manager import duckdb_connection, compute_and_insert_chromatograms_from_ms_data
from ..plugin_interface import PluginInterface

_label = "Optimization"


class TargetOptimizationPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 6
        print(f'Initiated {_label} plugin')

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)

    def outputs(self):
        return None


def downsample_for_preview(scan_time, intensity, max_points=100):
    """Reduce puntos manteniendo la forma general"""
    if len(scan_time) <= max_points:
        return scan_time, intensity

    indices = np.linspace(0, len(scan_time) - 1, max_points, dtype=int)
    return scan_time[indices], intensity[indices]


_layout = fac.AntdLayout(
    [
        fac.AntdHeader(
            [
                fac.AntdFlex(
                    [
                        fac.AntdTitle(
                            'Optimization', level=4, style={'margin': '0'}
                        ),
                        fac.AntdIcon(
                            id='ms-files-tour-icon',
                            icon='pi-info',
                            style={"cursor": "pointer", 'paddingLeft': '10px'},
                        ),
                        fac.AntdSpace(
                            [
                                fac.AntdButton(
                                    'Compute Chromatograms',
                                    id='compute-chromatograms-btn',
                                    style={'textTransform': 'uppercase'},
                                ),
                            ],
                            addSplitLine=True,
                            size="small",
                            style={"margin": "0 50px"},
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
                        fac.AntdButton(
                            id='optimization-sidebar-collapse',
                            type='text',
                            icon=fac.AntdIcon(
                                id='optimization-sidebar-collapse-icon',
                                icon='antd-left',
                                style={'fontSize': '14px'}, ),
                            shape='default',
                            style={
                                'position': 'absolute',
                                'zIndex': 1,
                                # 'top': 0,
                                'right': -8,
                                'boxShadow': '2px 2px 5px 1px rgba(0,0,0,0.5)',
                                'background': 'white',
                            },

                        ),
                        fac.AntdFlex(
                            [
                                fac.AntdFlex(
                                    [
                                        fac.AntdTitle(
                                            'Sample Type',
                                            level=5,
                                            style={'margin': '0'}
                                        ),
                                        fac.AntdCompact(
                                            [
                                                fac.AntdTooltip(
                                                    fac.AntdIcon(
                                                        icon='pi-crosshair',
                                                        className='expand-icon',
                                                        id='mark-tree-action'
                                                    ),
                                                    title='Mark all Sample Types'
                                                ),
                                                fac.AntdTooltip(
                                                    fac.AntdIcon(
                                                        icon='antd-up',
                                                        className='expand-icon',
                                                        id='collapse-tree-action'
                                                    ),
                                                    title='Collapse Sample Type Tree'
                                                ),
                                                fac.AntdTooltip(
                                                    fac.AntdIcon(
                                                        icon='antd-down',
                                                        className='expand-icon',
                                                        id='expand-tree-action'
                                                    ),
                                                    title='Expand Sample Type Tree'
                                                ),
                                            ],
                                        )
                                    ],
                                    justify='space-between',
                                    align='center',
                                    style={'marginRight': 30, 'height': 32, 'overflow': 'hidden'}
                                ),
                                html.Div(
                                    [
                                        fac.AntdTree(
                                            id='sample-type-tree',
                                            treeData=[],
                                            multiple=True,
                                            checkable=True,
                                            defaultExpandAll=True,
                                            showIcon=True,
                                            style={'display': 'none'}
                                        ),
                                        fac.AntdEmpty(
                                            id='sample-type-tree-empty',
                                            description='No samples marked for optimization',
                                            locale='en-us',
                                            image='simple',
                                            styles={'root': {'height': '100%', 'alignContent': 'center'}}
                                        )
                                    ],
                                    style={
                                        'flex': '1',
                                        'overflow': 'auto',
                                        'minHeight': '0'
                                    }
                                ),

                                html.Div(
                                    [
                                        fac.AntdDivider(
                                            'Options',
                                            size='small'
                                        ),
                                        fac.AntdForm(
                                            [
                                                fac.AntdFormItem(
                                                    fac.AntdSelect(
                                                        id='chromatogram-preview-filter-ms-type',
                                                        options=['All', 'ms1', 'ms2'],
                                                        value='All',
                                                        placeholder='Select ms_type',
                                                        style={'width': '100%'},
                                                        allowClear=False,
                                                        locale="en-us",
                                                    ),
                                                    label='MS-Type:',
                                                    tooltip='Filter chromatograms by ms_type',
                                                    style={'marginBottom': '1rem'}
                                                ),
                                                fac.AntdFormItem(
                                                    fac.AntdSelect(
                                                        id='chromatogram-preview-filter-bookmark',
                                                        options=['All', 'Bookmarked', 'Unmarked'],
                                                        value='All',
                                                        placeholder='Select filter',
                                                        style={'width': '100%'},
                                                        allowClear=False,
                                                        locale="en-us",
                                                    ),
                                                    label='Selection:',
                                                    tooltip='Filter chromatograms by bookmark status',
                                                    style={'marginBottom': '1rem'}
                                                ),
                                                fac.AntdFormItem(
                                                    fac.AntdSelect(
                                                        id='chromatogram-preview-order',
                                                        options=[{'label': 'By Peak Label', 'value': 'peak_label'},
                                                                 {'label': 'By MZ-Mean', 'value': 'mz_mean'}],
                                                        value='peak_label',
                                                        placeholder='Select filter',
                                                        style={'width': '100%'},
                                                        allowClear=False,
                                                        locale="en-us",
                                                    ),
                                                    label='Order by:',
                                                    tooltip='Ascended order chromatograms by peak label or mz mean',
                                                    style={'marginBottom': '1rem'}
                                                ),
                                                fac.AntdFormItem(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-preview-log-y',
                                                        checked=False
                                                    ),
                                                    label='Intensity Log Scale',
                                                    tooltip='Apply log scale to intensity axis',
                                                    style={'marginBottom': '0'}
                                                )
                                            ],
                                            style={'padding': 10}
                                        ),
                                        fac.AntdForm(
                                            [
                                                fac.AntdFormItem(
                                                    fac.AntdCompact(
                                                        [
                                                            fac.AntdInputNumber(
                                                                id='chromatogram-graph-width',
                                                                defaultValue=250,
                                                                min=180,
                                                                max=1400
                                                            ),
                                                            fac.AntdInputNumber(
                                                                id='chromatogram-graph-height',
                                                                defaultValue=180,
                                                                min=100,
                                                                max=700
                                                            ),
                                                        ],
                                                        style={'width': '160px'}
                                                    ),
                                                    label='WxH:',
                                                    tooltip='Set preview plot width and height'
                                                ),
                                                fac.AntdFormItem(
                                                    fac.AntdButton(
                                                        # 'Apply',
                                                        id='chromatogram-graph-button',
                                                        icon=fac.AntdIcon(icon='pi-broom', style={'fontSize': 20}),
                                                        # type='primary'
                                                    ),
                                                    style={"marginInlineEnd": 0}
                                                )
                                            ],
                                            layout='inline',
                                            style={'padding': 10, 'justifyContent': 'center'}
                                        )
                                    ],
                                    style={'overflow': 'hidden'}
                                ),
                            ],
                            vertical=True,
                            justify='space-between',
                            style={'height': '100%'}
                        )
                    ],
                    id='optimization-sidebar',
                    collapsible=True,
                    collapsedWidth=0,
                    width=300,
                    trigger=None,
                    style={'height': '100%'},
                    className="sidebar-mint"
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                fac.AntdSpin(
                                    [
                                        html.Div(
                                            [
                                                fac.AntdSpace(
                                                    [
                                                        fac.AntdCard(
                                                            id={'type': 'target-card-preview', 'index': i},
                                                            style={'cursor': 'pointer'},
                                                            styles={'header': {'display': 'none'},
                                                                    'body': {'padding': '5px'},
                                                                    'actions': {'height': 30}},
                                                            hoverable=True,
                                                            children=[
                                                                dcc.Graph(
                                                                    id={'type': 'graph', 'index': i},
                                                                    figure=go.Figure(
                                                                        layout=dict(
                                                                            xaxis_title="Retention Time [s]",
                                                                            yaxis_title="Intensity",
                                                                            showlegend=False,
                                                                            margin=dict(l=40, r=5, t=30, b=30),
                                                                            hovermode=False,
                                                                            dragmode=False,
                                                                        )
                                                                    ),
                                                                    style={
                                                                        'height': '180px', 'width': '250px',
                                                                        'margin': '0px',
                                                                    },
                                                                    config={
                                                                        'displayModeBar': False,
                                                                        'staticPlot': True,
                                                                        'doubleClick': False,
                                                                        'showTips': False,
                                                                        'responsive': False
                                                                    },
                                                                ),
                                                                fac.AntdRate(
                                                                    id={'type': 'bookmark-target-card', 'index': i},
                                                                    count=1,
                                                                    defaultValue=0,
                                                                    value=0,
                                                                    allowHalf=False,
                                                                    tooltips=['Bookmark this target'],
                                                                    style={'position': 'absolute', 'top': '8px',
                                                                           'right': '8px', 'zIndex': 20},
                                                                )
                                                            ],
                                                            **{'data-target': None}
                                                        ) for i in range(20)  # Only 20 pre-configures cards
                                                    ],
                                                    id='chromatogram-preview',
                                                    wrap=True,
                                                    align='center',
                                                    style={'height': 'calc(100vh - 64px - 4rem)', 'overflowY': 'auto',
                                                           'width': '100%', 'padding': '0 10px 10px 10px',
                                                           'justifyContent': 'center'}
                                                ),
                                                fac.AntdPagination(
                                                    id='chromatogram-preview-pagination',
                                                    defaultPageSize=20,
                                                    showSizeChanger=True,
                                                    pageSizeOptions=[4, 6, 8, 10, 12, 16, 20],
                                                    locale='en-us',
                                                    align='center',
                                                    showTotalSuffix='targets',
                                                ),
                                                html.Div(
                                                    id='chromatograms-dummy-output',
                                                    style={'display': 'none'}
                                                )
                                            ])
                                    ],
                                    text='Loading plots...',
                                ),
                            ],
                            id='chromatogram-preview-container'
                        ),
                        fac.AntdEmpty(
                            id='chromatogram-preview-empty',
                            description="No chromatograms to preview.",
                            locale='en-us',
                            style={"display": "none"}
                        ),
                    ],
                    className='ant-layout-content css-1v28nim',
                    style={'background': 'white',
                           # 'height': 'calc(100vh - 64px - 4rem)', 'overflowY': 'auto'
                           'alignContent': 'center'
                           }
                ),
            ],
            style={'padding': '1rem 0', 'background': 'white'},
        ),
        fac.AntdModal(
            [
                fac.AntdFlex(
                    [
                        fac.AntdDivider('Recompute Chromatograms'),
                        fac.AntdForm(
                            [
                                fac.AntdFormItem(
                                    fac.AntdCheckbox(
                                        id='chromatograms-recompute-ms1',
                                        label='Recompute MS1'
                                    ),

                                ),
                                fac.AntdFormItem(
                                    fac.AntdCheckbox(
                                        id='chromatograms-recompute-ms2',
                                        label='Recompute MS2'
                                    ),
                                ),
                                # fac.AntdFormItem(
                                #     fac.AntdSelect(
                                #         id='compute-chromatogram-targets-select',
                                #         options=['All', 'Preselected for processing'],
                                #         allowClear=False,
                                #         placeholder='Select targets',
                                #         defaultValue='Preselected for processing',
                                #         style={'width': '100%'},
                                #     ),
                                #     label='Select targets',
                                # ),
                                # fac.AntdFormItem(
                                #     fac.AntdSelect(
                                #         id='compute-chromatogram-samples-select',
                                #         options=['All', 'Use for Optimization'],
                                #         allowClear=False,
                                #         placeholder='Select samples',
                                #         defaultValue='Use for Optimization',
                                #         style={'width': '100%'},
                                #     ),
                                #     label='Select samples',
                                # ),
                            ],
                            layout='inline'
                        ),
                        fac.AntdDivider('Configuration'),
                        fac.AntdForm(
                            [
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='chromatogram-compute-cpu',
                                        defaultValue=cpu_count() - 2,
                                        min=1,
                                        max=cpu_count() - 2,
                                    ),
                                    label='CPU:',
                                    hasFeedback=True,
                                    help=f"Selected {cpu_count() - 2} / {cpu_count()} cpus"

                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='chromatogram-compute-ram',
                                        value=round(psutil.virtual_memory().available * 0.9 / (1024 ** 3), 1),
                                        min=1,
                                        precision=1,
                                        step=0.1,
                                        suffix='GB'
                                    ),
                                    label='RAM:',
                                    hasFeedback=True,
                                    id='chromatogram-compute-ram-item',
                                    help=f"Selected "
                                         f"{round(psutil.virtual_memory().available * 0.9 / (1024 ** 3), 1)}GB / "
                                         f"{round(psutil.virtual_memory().available / (1024 ** 3), 1)}GB available RAM"
                                ),
                            ],
                            layout='inline'
                        ),

                        fac.AntdDivider(),
                        fac.AntdAlert(
                            message='There are already computed chromatograms',
                            type='warning',
                            showIcon=True,
                            id='chromatogram-warning',
                            style={'display': 'none'},
                        )
                    ],
                    id='chromatogram-compute-options-container',
                    vertical=True
                ),

                html.Div(
                    [
                        html.H4("Generating Chromatograms..."),
                        fac.AntdProgress(
                            id='chromatogram-processing-progress',
                            percent=0,
                        ),
                        fac.AntdButton(
                            'Cancel',
                            id='cancel-chromatogram-processing',
                            style={
                                'alignText': 'center',
                            },
                        ),
                    ],
                    id='chromatogram-processing-progress-container',
                    style={'display': 'none'},
                ),
            ],
            id='compute-chromatogram-modal',
            title='Compute chromatograms',
            width=700,
            renderFooter=True,
            locale='en-us',
            confirmAutoSpin=True,
            loadingOkText='Generating Chromatograms...',
            okClickClose=False,
            closable=False,
            maskClosable=False,
            destroyOnClose=True,
            okText="Generate",
            centered=True,
            styles={'body': {'height': "50vh"}},
        ),
        fac.AntdModal(
            id="chromatogram-view-modal",
            width="100vw",
            centered=True,
            destroyOnClose=True,
            closable=False,
            maskClosable=False,
            children=[
                html.Div(
                    [
                        dcc.Graph(
                            id='chromatogram-view-plot',
                            figure=go.Figure(
                                layout=dict(
                                    xaxis_title="Retention Time [s]",
                                    yaxis_title="Intensity",
                                    showlegend=True,
                                    margin=dict(l=40, r=10, t=50, b=80),
                                )
                            ),
                            config={'displayModeBar': True},
                            style={'width': '100%', 'height': '600px'}
                        ),
                        html.Div(
                            [
                                dcc.RangeSlider(
                                    id='rt-range-slider',
                                    step=1,
                                    allowCross=False,
                                    updatemode="drag",
                                ),
                                fac.AntdFlex(
                                    id="rt-values-span",
                                    justify='space-between',
                                    align='center'
                                ),
                            ],
                            id='rslider',
                        ),

                        fac.AntdDrawer(
                            [
                                fac.AntdForm(
                                    [
                                        fac.AntdFormItem(
                                            fac.AntdSwitch(
                                                id='chromatogram-view-log-y',
                                                checked=False,
                                            ),
                                            label='Intensity Log Scale:',
                                        ),
                                        fac.AntdFormItem(
                                            fac.AntdSwitch(
                                                id='chromatogram-view-groupclick',
                                                checked=False,
                                                checkedChildren='Group',
                                                unCheckedChildren='Single'
                                            ),
                                            label='Legend behavior:',
                                        ),
                                    ]
                                )
                            ],
                            id='chromatogram-view-options-drawer',
                            title='Options',
                            containerId='chromatogram-view-container',
                            placement='right',
                            styles={
                                'mask': {'background': 'rgba(0, 0, 0, 0)'}
                            },
                            closable=False
                        ),
                    ],
                    id='chromatogram-view-container',
                    style={
                        'position': 'relative',
                        'overflowX': 'hidden',
                    },
                ),
                fac.AntdDivider(size='small'),
                fac.AntdFlex(
                    [
                        fac.AntdSpace(
                            [
                                fac.AntdAlert(
                                    message="RT values have been changed. Save or reset the changes.",
                                    type="warning",
                                    showIcon=True,
                                ),
                                fac.AntdSpace(
                                    [
                                        fac.AntdButton(
                                            "Reset",
                                            id="reset-btn",
                                        ),
                                        fac.AntdButton(
                                            "Save",
                                            id="save-btn",
                                            type="primary",
                                        ),
                                    ],
                                    addSplitLine=True,
                                    size='small'
                                )
                            ],
                            align='center',
                            size=60,
                            id='action-buttons-container',
                            style={
                                "visibility": "hidden",
                                'opacity': '0',
                                'transition': 'opacity 0.3s ease-in-out',
                            }
                        ),
                        fac.AntdSpace(
                            [
                                fac.AntdButton(
                                    id='chromatogram-view-options',
                                    icon=fac.AntdIcon(icon='antd-setting', style={'fontSize': 20}),
                                    shape="circle"
                                ),
                                fac.AntdButton(
                                    "Close",
                                    id="chromatogram-view-close",
                                ),
                            ],
                            size=20,
                            addSplitLine=True,
                            style={
                                'marginLeft': '60px',
                            },
                        ),

                    ],
                    justify='space-between',
                    align='center',
                ),
            ]
        ),
        fac.AntdModal(
            id="delete-targets-modal",
            title="Delete target",
            width=400,
            renderFooter=True,
            okText="Delete",
            okButtonProps={"danger": True},
            cancelText="Cancel",
            locale='en-us',
        ),
        fac.AntdModal(
            "Are you sure you want to close this window without saving your changes?",
            id="confirm-unsave-modal",
            title="Confirm close without saving",
            width=400,
            okButtonProps={'danger': True},
            renderFooter=True,
            locale='en-us'
        ),

        dcc.Store(id='slider-data'),
        dcc.Store(id='slider-reference-data'),
        dcc.Store(id='target-preview-clicked'),

        dcc.Store(id='chromatograms', data={}),
        dcc.Store(id='drop-chromatogram'),
        dcc.Store(id="delete-target-clicked"),
        dcc.Store(id='chromatogram-view-plot-max')
    ],
    style={'height': '100%'},
)


def layout():
    return _layout


def callbacks(app, fsc, cache, cpu=None):
    app.clientside_callback(
        """(nClicks, collapsed) => {
            return [!collapsed, collapsed ? 'antd-left' : 'antd-right'];
        }""",
        Output('optimization-sidebar', 'collapsed'),
        Output('optimization-sidebar-collapse-icon', 'icon'),

        Input('optimization-sidebar-collapse', 'nClicks'),
        State('optimization-sidebar', 'collapsed'),
        prevent_initial_call=True,
    )

    # clientside callback for set width rangeslider == plot bglayer
    app.clientside_callback(
        """
        function(relayoutData, visible) {
            if (visible !== true) {
                return window.dash_clientside.no_update;
            }
            return new Promise((resolve) => {
                setTimeout(() => {
                    const root = document.getElementById("chromatogram-view-plot");
                    const bg = root?.querySelector("div > div > div svg > g.draglayer > g.xy > rect");
                    
                    if (bg) {
                        const pl = bg.x.baseVal.value - 25;
                        const w = bg.width.baseVal.value + 50;
                        
                        if (isFinite(w) && w > 0) {
                            resolve({"marginLeft": pl + "px", "width": w + "px"});
                            return;
                        }
                    }
                    resolve(window.dash_clientside.no_update);
                }, 150);  // 150ms suele ser suficiente
            });
        }
        """,
        Output("rslider", "style"),
        Input("chromatogram-view-plot", "relayoutData"),
        Input("chromatogram-view-modal", 'visible'),
        prevent_initial_call=True
    )

    ############# TREE BEGIN #####################################
    @app.callback(
        Output('sample-type-tree', 'treeData'),
        Output('sample-type-tree', 'checkedKeys'),
        Output('sample-type-tree', 'expandedKeys'),
        Output('sample-type-tree', 'style'),
        Output('sample-type-tree-empty', 'style'),

        Input('section-context', 'data'),
        Input('mark-tree-action', 'nClicks'),
        Input('expand-tree-action', 'nClicks'),
        Input('collapse-tree-action', 'nClicks'),
        Input('chromatogram-preview-filter-ms-type', 'value'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def update_sample_type_tree(section_context, mark_action, expand_action, collapse_action, selection_ms_type, wdir):

        ctx = dash.callback_context
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if section_context['page'] != 'Optimization':
            raise PreventUpdate

        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update, dash.no_update, dash.no_update
            df = conn.execute("""
                              SELECT sample_type,
                                     list({'title': label, 'key': label})                                as children,
                                     (SELECT list(label) FROM samples WHERE use_for_optimization = TRUE) as checked_keys
                              FROM samples
                              WHERE use_for_optimization = TRUE
                                AND CASE
                                        WHEN ? = 'ms1' THEN ms_type = 'ms1'
                                        WHEN ? = 'ms2' THEN ms_type = 'ms2'
                                        ELSE TRUE -- 'all' case
                                  END
                              GROUP BY sample_type
                              ORDER BY sample_type
                              """, [selection_ms_type, selection_ms_type]).df()

            if df.empty:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

            if prop_id in ['mark-tree-action', 'section-context']:
                checked_keys = df['checked_keys'].iloc[0]  # Es el mismo en todas las filas
            else:
                checked_keys = dash.no_update

            if prop_id in ['section-context', 'chromatogram-preview-filter-ms-type']:
                tree_data = [
                    {
                        'title': row['sample_type'],
                        'key': row['sample_type'],
                        'children': row['children']
                    }
                    for _, row in df.iterrows()
                ]
            else:
                tree_data = dash.no_update

            if prop_id in ['expand-tree-action', 'section-context']:
                expanded_keys = df['sample_type'].tolist()
            elif prop_id == 'collapse-tree-action':
                expanded_keys = []
            else:
                expanded_keys = dash.no_update
        return tree_data, checked_keys, expanded_keys, {'display': 'flex'}, {'display': 'none'}

    ############# TREE END #######################################

    ############# GRAPH OPTIONS BEGIN #####################################
    @app.callback(
        Output({'type': 'graph', 'index': ALL}, 'style'),
        Input('chromatogram-graph-button', 'nClicks'),
        State('chromatogram-graph-width', 'value'),
        State('chromatogram-graph-height', 'value'),
        prevent_initial_call=True
    )
    def set_chromatogram_graph_size(nClicks, width, height):

        if not nClicks:
            raise PreventUpdate
        return [{
            'width': width, 'height': height,
            'margin': '0px',
        } for _ in range(20)]

    ############# GRAPH OPTIONS END #######################################

    ############# PREVIEW BEGIN #####################################
    @app.callback(
        Output({'type': 'target-card-preview', 'index': ALL}, 'data-target'),
        Output({'type': 'graph', 'index': ALL}, 'figure'),
        Output({'type': 'bookmark-target-card', 'index': ALL}, 'value'),
        Output('chromatogram-preview-pagination', 'total'),
        Output('chromatograms-dummy-output', 'children'),

        Input('chromatograms', 'data'),
        Input('chromatogram-preview-pagination', 'current'),
        Input('chromatogram-preview-pagination', 'pageSize'),
        Input('sample-type-tree', 'checkedKeys'),
        Input('chromatogram-preview-log-y', 'checked'),
        Input('chromatogram-preview-filter-bookmark', 'value'),
        Input('chromatogram-preview-filter-ms-type', 'value'),
        Input('chromatogram-preview-order', 'value'),
        Input('drop-chromatogram', 'data'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def chromatograms_preview(chromatograms, current_page, page_size, checkedkeys, log_scale, selection_bookmark,
                              selection_ms_type, targets_order, dropped_target, wdir):
        import polars as pl
        start_idx = (current_page - 1) * page_size
        t1 = time.perf_counter()

        with duckdb_connection(wdir) as conn:
            all_targets = conn.execute("""
                                       SELECT *
                                       from targets
                                       WHERE CASE
                                                 WHEN ? = 'ms1' THEN ms_type = 'ms1'
                                                 WHEN ? = 'ms2' THEN ms_type = 'ms2'
                                                 ELSE TRUE
                                           END
                                         AND CASE
                                                 WHEN ? = 'Bookmarked' THEN bookmark = TRUE
                                                 WHEN ? = 'Unmarked' THEN bookmark = FALSE
                                                 ELSE TRUE -- 'all' case
                                           END
                                       """, [selection_ms_type, selection_ms_type,
                                             selection_bookmark, selection_bookmark, ]).pl()
            query = f"""
                        WITH picked_samples AS (SELECT ms_file_label, color, label
                                                FROM samples
                                                WHERE use_for_optimization = TRUE
                                                  AND ms_file_label IN (SELECT unnest(?::VARCHAR[]))),
                             picked_targets AS (SELECT peak_label,
                                                       rt,
                                                       rt_min,
                                                       rt_max,
                                                       rt_min - (rt_max - rt) AS scan_time_min,
                                                       rt_max + (rt - rt_min) AS scan_time_max,
                                                       mz_mean,
                                                       bookmark,
                                                       intensity_threshold
                                                FROM targets
                                                WHERE CASE
                                                          WHEN ? = 'ms1' THEN ms_type = 'ms1'
                                                          WHEN ? = 'ms2' THEN ms_type = 'ms2'
                                                          ELSE TRUE
                                                    END
                                                  AND CASE
                                                          WHEN ? = 'Bookmarked' THEN bookmark = TRUE
                                                          WHEN ? = 'Unmarked' THEN bookmark = FALSE
                                                          ELSE TRUE -- 'all' case
                                                          END
                                                LIMIT ? -- 1) limit
                                                    OFFSET ? -- 2) offset
                             ),
                             base AS (SELECT c.*,
                                             s.color,
                                             s.label,
                                             t.scan_time_min,
                                             t.scan_time_max,
                                             t.intensity_threshold,
                                             t.mz_mean
                                      FROM chromatograms c
                                               JOIN picked_samples s USING (ms_file_label)
                                               JOIN picked_targets t USING (peak_label)),
                             -- Emparejamos (scan_time[i], intensity[i]) en una lista de structs
                             zipped AS (SELECT peak_label,
                                               ms_file_label,
                                               color,
                                               label,
                                               scan_time_min,
                                               scan_time_max,
                                               intensity_threshold,
                                               mz_mean,
                                               list_transform(
                                                       range(1, len(scan_time) + 1),
                                                       i -> struct_pack(
                                                               t := list_extract(scan_time, i),
                                                               i := list_extract(intensity, i)
                                                            )
                                               ) AS pairs
                                        FROM base),
                             -- Filtramos por el rango de tiempo Y por intensity_threshold
                             sliced AS (SELECT peak_label,
                                               ms_file_label,
                                               color,
                                               label,
                                               mz_mean,
                                               pairs,
                                               list_filter(pairs, p -> p.t >= scan_time_min AND p.t <= scan_time_max AND
                                                                  p.i >= COALESCE(intensity_threshold, 0)) AS pairs_in
                                        FROM zipped),
                             -- Calculamos min/max de TODO el cromatograma (pairs completo) PERO por peak_label
                             -- Tomamos el máximo de todos los ms_file_label para ese peak_label
                             global_stats AS (SELECT peak_label,
                                                     MAX(list_max(list_transform(pairs, p -> p.i))) * 1.10 AS intensity_max_global,
                                                     MIN(list_min(list_transform(pairs, p -> p.i)))        AS intensity_min_global
                                              FROM zipped
                                              GROUP BY peak_label),
                             -- Reconstruimos listas y unimos con las estadísticas globales
                             final AS (SELECT s.peak_label,
                                              s.ms_file_label,
                                              s.color,
                                              s.label,
                                              s.mz_mean,
                                              list_transform(pairs_in, p -> p.t) AS scan_time_sliced,
                                              list_transform(pairs_in, p -> p.i) AS intensity_sliced,
                                              g.intensity_max_global             AS intensity_max_in_range,
                                              g.intensity_min_global             AS intensity_min_in_range
                                       FROM sliced s
                                                JOIN global_stats g USING (peak_label))
                        SELECT *
                        FROM final
                        ORDER BY {targets_order}, ms_file_label;
                        """
            df = conn.execute(query, [checkedkeys, selection_ms_type, selection_ms_type,
                                      selection_bookmark, selection_bookmark, page_size, start_idx]
                              ).pl()

        titles = []
        figures = []
        bookmarks = []
        for target_data, g in df.group_by(['peak_label', 'intensity_min_in_range', 'intensity_max_in_range'],
                                          maintain_order=True):
            peak_label, intensity_min_in_range, intensity_max_in_range = target_data

            titles.append(peak_label)
            target_dict = all_targets.filter(pl.col('peak_label') == peak_label).head(1).rows(named=True)[0]
            bookmarks.append(int(target_dict['bookmark']))  # convert bool to int

            fig = Patch()
            traces = []
            for i, row in enumerate(g.iter_rows(named=True)):
                traces.append({
                    'type': 'scatter',
                    'mode': 'lines',
                    'x': row['scan_time_sliced'],
                    'y': row['intensity_sliced'],
                    'name': row['label'] or row['ms_file_label'],
                    'line': {'color': row['color']},
                })

            fig['data'] = traces

            y_min = intensity_min_in_range
            y_max = intensity_max_in_range

            fig['layout']['shapes'] = [
                {
                    'line': {'color': 'black', 'width': 1.5, 'dash': 'dashdot'},
                    'type': 'line',
                    'x0': target_dict['rt'],
                    'x1': target_dict['rt'],
                    'xref': 'x',
                    'y0': 0,
                    'y1': 1,
                    'yref': 'y domain'
                },
                {
                    'fillcolor': 'green',
                    'line': {'width': 0},
                    'opacity': 0.1,
                    'type': 'rect',
                    'x0': target_dict['rt_min'],
                    'x1': target_dict['rt_max'],
                    'xref': 'x',
                    'y0': 0,
                    'y1': 1,
                    'yref': 'y domain'
                }
            ]
            fig['layout']['template'] = 'plotly_white'

            filter_type = (f"mz_mean = {target_dict['mz_mean']}"
                           if target_dict['ms_type'] == 'ms1'
                           else f"{target_dict['filterLine']}")
            fig['layout']['title'] = dict(text=peak_label, font={'size': 14},
                                          subtitle=dict(text=f"{filter_type}"))

            x_min = target_dict['rt_min'] - (target_dict['rt_max'] - target_dict['rt'])
            x_max = target_dict['rt_max'] + (target_dict['rt'] - target_dict['rt_min'])

            fig['layout']['xaxis']['title'] = dict(text="Retention Time [s]", font={'size': 10})
            fig['layout']['xaxis']['autorange'] = False
            fig['layout']['xaxis']['fixedrange'] = True
            fig['layout']['xaxis']['range'] = [x_min, x_max]

            fig['layout']['yaxis']['title'] = dict(text="Intensity", font={'size': 10})
            fig['layout']['yaxis']['autorange'] = False

            if log_scale:
                fig['layout']['yaxis']['type'] = 'log'
                log_y_min = math.log10(y_min) if y_min > 0 else y_min
                log_y_max = math.log10(y_max) if y_max > 0 else y_max

                fig['layout']['yaxis']['range'] = [log_y_min, log_y_max]
            else:
                fig['layout']['yaxis']['type'] = 'linear'
                fig['layout']['yaxis']['range'] = [y_min, y_max]

            fig["layout"]["showlegend"] = False
            fig['layout']['margin'] = dict(l=40, r=5, t=50, b=30)
            # fig['layout']['uirevision'] = f"xr_{peak_label}"
            figures.append(fig)

        titles.extend([None for _ in range(20 - len(figures))])
        figures.extend([{} for _ in range(20 - len(figures))])
        bookmarks.extend([0 for _ in range(20 - len(bookmarks))])

        print(f"{time.perf_counter() - t1 = }")
        return titles, figures, bookmarks, len(all_targets), []

    @app.callback(
        Output({'type': 'target-card-preview', 'index': ALL}, 'style'),
        Output('chromatogram-preview-container', 'style'),
        Output('chromatogram-preview-empty', 'style'),

        Input({'type': 'graph', 'index': ALL}, 'figure'),
        prevent_initial_call=True
    )
    def toggle_card_visibility(figures):
        visible_fig = 0
        cards_style = []
        for figure in figures:
            if figure:
                cards_style.append({'display': 'block'})
                visible_fig += 1
            else:
                cards_style.append({'display': 'none'})

        show_empty = {'display': 'block'} if visible_fig == 0 else {'display': 'none'}
        show_space = {'display': 'none'} if visible_fig == 0 else {'display': 'block'}
        return cards_style, show_space, show_empty

    ############# PREVIEW END #######################################

    ############# VIEW MODAL BEGIN #####################################
    @app.callback(
        Output('target-preview-clicked', 'data'),

        [Input({'type': 'target-card-preview', 'index': ALL}, 'nClicks')],
        [Input({'type': 'bookmark-target-card', 'index': ALL}, 'value')],
        State({'type': 'target-card-preview', 'index': ALL}, 'data-target'),
        prevent_initial_call=True
    )
    def open_chromatogram_view_modal(card_preview_clicks, bookmark_target_clicks, data_target):
        if not card_preview_clicks:
            raise PreventUpdate

        ctx = dash.callback_context
        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_type = ctx_trigger['type']

        if len(ctx.triggered) > 1 or trigger_type != 'target-card-preview':
            raise PreventUpdate

        ctx = dash.callback_context
        prop_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        return data_target[prop_id['index']]

    @app.callback(
        Output('chromatogram-view-modal', 'visible'),
        Output('slider-reference-data', 'data', allow_duplicate=True),

        Input('target-preview-clicked', 'data'),
        Input('chromatogram-view-close', 'nClicks'),
        Input('confirm-unsave-modal', 'okCounts'),
        State('slider-reference-data', 'data'),
        State('slider-data', 'data'),
        prevent_initial_call=True
    )
    def handle_modal_open_close(target_clicked, close_clicks, close_without_save_clicks, slider_ref, slider_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'target-preview-clicked':
            return True, dash.no_update
            # if not has_changes, close it
        elif trigger_id == 'chromatogram-view-close':
            if slider_ref and slider_data and slider_ref['value'] == slider_data['value']:
                return False, None
            # if it has_changes, don't close it
            return dash.no_update, dash.no_update
        elif trigger_id == 'confirm-unsave-modal':
            # Close modal without saving changes
            if close_without_save_clicks:
                return False, None

        return dash.no_update, dash.no_update

    @app.callback(
        Output('confirm-unsave-modal', 'visible'),

        Input('chromatogram-view-close', 'nClicks'),
        State('slider-reference-data', 'data'),
        State('slider-data', 'data'),
        prevent_initial_call=True
    )
    def show_confirm_modal(close_clicks, reference_data, slider_data):
        has_changes = slider_data['value'] != reference_data['value']
        return bool(close_clicks and has_changes)

    ############# VIEW MODAL END #######################################

    ############# VIEW OPTIONS BEGIN #######################################

    @app.callback(
        Output('chromatogram-view-options-drawer', 'visible'),

        Input('chromatogram-view-options', 'nClicks'),
        State('chromatogram-view-options-drawer', 'visible'),
        prevent_initial_call=True
    )
    def show_options_drawer(nClicks, drawer_visible):
        return not drawer_visible

    ############# VIEW OPTIONS END #######################################

    ############# VIEW BEGIN #######################################
    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),

        Input('chromatogram-view-log-y', 'checked'),
        State('chromatogram-view-plot', 'figure'),
        State('chromatogram-view-plot-max', 'data'),
        prevent_initial_call=True
    )
    def chromatogram_view_y_scale(log_scale, figure, max_y):

        y_min, y_max = max_y
        fig = Patch()
        if log_scale:
            if figure['layout']['yaxis']['type'] == 'log':
                raise PreventUpdate
            fig['layout']['yaxis']['type'] = 'log'
            log_y_min = math.log10(y_min) if y_min > 0 else y_min
            log_y_max = math.log10(y_max) if y_max > 0 else y_max

            fig['layout']['yaxis']['range'] = [log_y_min, log_y_max]
        else:
            if figure['layout']['yaxis']['type'] == 'linear':
                raise PreventUpdate
            fig['layout']['yaxis']['type'] = 'linear'
            linear_y_min = y_min
            linear_y_max = y_max

            fig['layout']['yaxis']['range'] = [linear_y_min, linear_y_max]
        return fig

    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Input('chromatogram-view-groupclick', 'checked'),
        prevent_initial_call=True
    )
    def chromatogram_view_legend_group(groupclick):
        fig = Patch()
        if groupclick:
            fig['layout']['legend']['groupclick'] = 'togglegroup'
        else:
            fig['layout']['legend']['groupclick'] = 'toggleitem'
        return fig

    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Output('chromatogram-view-modal', 'title'),
        Output('chromatogram-view-modal', 'loading'),
        Output('slider-reference-data', 'data'),
        Output('slider-data', 'data', allow_duplicate=True),  # make sure this is reset
        Output('chromatogram-view-plot-max', 'data'),
        Output('chromatogram-view-log-y', 'checked', allow_duplicate=True),

        Input('target-preview-clicked', 'data'),
        State('chromatogram-preview-log-y', 'checked'),
        State('sample-type-tree', 'checkedKeys'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def chromatogram_view_modal(target_clicked, log_scale, checkedKeys, wdir):

        with duckdb_connection(wdir) as conn:
            d = conn.execute("SELECT rt, rt_min, rt_max FROM targets WHERE peak_label = ?", [target_clicked]).fetchall()
            rt, rt_min, rt_max = d[0] if d else (None, None, None)

            query = """
                    WITH picked_samples AS (SELECT ms_file_label, color, label, sample_type
                                            FROM samples
                                            WHERE use_for_optimization = TRUE
                        -- AND ms_file_label IN (SELECT unnest(?::VARCHAR[]))
                    ),
                         picked_target AS (SELECT peak_label,
                                                  rt,
                                                  rt_min,
                                                  rt_max,
                                                  intensity_threshold
                                           FROM targets
                                           WHERE peak_label = ?),
                         base AS (SELECT c.*,
                                         s.color,
                                         s.label,
                                         s.sample_type,
                                         t.intensity_threshold
                                  FROM chromatograms c
                                           JOIN picked_samples s USING (ms_file_label)
                                           JOIN picked_target t USING (peak_label)),
                         -- Emparejamos (scan_time[i], intensity[i]) en una lista de structs
                         zipped AS (SELECT ms_file_label,
                                           color,
                                           label,
                                           sample_type,
                                           intensity_threshold,
                                           list_transform(
                                                   range(1, len(scan_time) + 1),
                                                   i -> struct_pack(
                                                           t := list_extract(scan_time, i),
                                                           i := list_extract(intensity,  i)
                                                        )
                                           ) AS pairs
                                    FROM base),

                         sliced AS (SELECT ms_file_label,
                                           color,
                                           label,
                                           sample_type,
                                           pairs,
                                           list_filter(pairs, p -> p.i >= COALESCE(intensity_threshold, 0)) AS pairs_in
                                    FROM zipped),
                         -- Reconstruimos listas y calculamos min/max de intensidad COMPLETO
                         final AS (SELECT ms_file_label,
                                          color,
                                          label,
                                          sample_type,
                                          list_transform(pairs_in, p -> p.t)                            AS scan_time_sliced,
                                          list_transform(pairs_in, p -> p.i)                            AS intensity_sliced,
                                          CASE
                                              WHEN len(pairs) = 0 THEN NULL
                                              ELSE list_max(list_transform(pairs, p -> p.i)) * 1.10 END AS
                                                                                                           intensity_max_in_range,
                                          CASE
                                              WHEN len(pairs) = 0 THEN NULL
                                              ELSE list_min(list_transform(pairs, p -> p.i)) END        AS intensity_min_in_range,
                                          CASE
                                              WHEN len(pairs) = 0 THEN NULL
                                              ELSE list_max(list_transform(pairs, p -> p.t)) END        AS scan_time_max_in_range,
                                          CASE
                                              WHEN len(pairs) = 0 THEN NULL
                                              ELSE list_min(list_transform(pairs, p -> p.t)) END        AS scan_time_min_in_range

                                   FROM sliced)
                    SELECT *
                    FROM final
                    ORDER BY ms_file_label;
                    """

            chrom_df = conn.execute(query, [target_clicked]).pl()

        t1 = time.perf_counter()
        fig = Patch()
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

        legend_groups = set()
        traces = []
        # TODO: check if chrom_df is empty and Implement an empty widget to show when no data
        for i, row in enumerate(chrom_df.iter_rows(named=True)):
            # row: ms_file_label, color, label, scan_time_sliced, intensity_sliced, intensity_max_in_range,
            #                 intensity_min_in_range, scan_time_max_in_range, scan_time_min_in_range

            trace = {
                'type': 'scattergl',
                'mode': 'lines',
                'x': row['scan_time_sliced'],
                'y': row['intensity_sliced'],
                'line': {'color': row['color']},
                'name': row['label'] or row['ms_file_label'],
                'visible': True if row['label'] in checkedKeys else 'legendonly',  # solo en leyenda si no está
                'legendgroup': row['sample_type']
            }

            if row['sample_type'] not in legend_groups:
                trace['legendgrouptitle'] = {'text': row['sample_type']}
                legend_groups.add(row['sample_type'])

            traces.append(trace)

            x_min = min(x_min, row['scan_time_min_in_range'])
            x_max = max(x_max, row['scan_time_max_in_range'])
            y_min = min(y_min, row['intensity_min_in_range'])
            y_max = max(y_max, row['intensity_max_in_range'])

        fig['data'] = traces
        fig['layout']['legend']['groupclick'] = 'toggleitem'

        fig['layout']['annotations'][0] = {
            'bgcolor': 'white',
            'font': {'color': 'black', 'size': 14, 'weight': 'bold'},
            'showarrow': True,
            'text': 'RT',
            'x': rt,
            'xanchor': 'left',
            'xref': 'x',
            'y': 1,
            'yanchor': 'top',
            'yref': 'y domain'
        }

        fig['layout']['shapes'] = [
            {
                'line': {'color': 'black', 'width': 3},
                'type': 'line',
                'x0': rt,
                'x1': rt,
                'xref': 'x',
                'y0': 0,
                'y1': 1,
                'yref': 'y domain'
            },
            {
                'fillcolor': 'green',
                'line': {'width': 0},
                'opacity': 0.1,
                'type': 'rect',
                'x0': rt_min,
                'x1': rt_max,
                'xref': 'x',
                'y0': 0,
                'y1': 1,
                'yref': 'y domain'
            }
        ]
        fig['layout']['template'] = 'plotly_white'

        fig['layout']['xaxis']['range'] = [x_min, x_max]
        fig['layout']['xaxis']['autorange'] = False
        fig['layout']['yaxis']['autorange'] = False

        if log_scale:
            fig['layout']['yaxis']['type'] = 'log'
            log_y_min = math.log10(y_min) if y_min > 0 else y_min
            log_y_max = math.log10(y_max) if y_max > 0 else y_max
            fig['layout']['yaxis']['range'] = [log_y_min, log_y_max]
        else:
            fig['layout']['yaxis']['type'] = 'linear'
            fig['layout']['yaxis']['range'] = [y_min, y_max]

        fig['layout']['margin'] = dict(l=60, r=10, t=40, b=40)

        s_data = {
            'min': x_min,
            'max': x_max,
            'pushable': 1,
            'step': 1,
            'tooltip': None,
            'marks': None,
            'value': {'rt_min': rt_min, 'rt': rt, 'rt_max': rt_max},
            'v_comp': {'rt_min': True, 'rt': True, 'rt_max': True}
        }

        print(f"{time.perf_counter() - t1 = }")
        return fig, target_clicked, False, s_data, None, [y_min, y_max], log_scale

    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Output('slider-data', 'data', allow_duplicate=True),
        Output('action-buttons-container', 'style'),

        Input('rt-range-slider', 'value'),
        State('slider-data', 'data'),
        State('slider-reference-data', 'data'),
        prevent_initial_call=True
    )
    def slider_value_changed(slider_value, slider_data, slider_reference_data):
        fig = Patch()
        s_data = slider_value.copy()

        if not slider_data['v_comp']['rt_min']:
            s_data = [slider_data['value']['rt_min']] + s_data
        if not slider_data['v_comp']['rt']:
            s_data.insert(1, slider_data['value']['rt'])
        if not slider_data['v_comp']['rt_max']:
            s_data = s_data + [slider_data['value']['rt_max']]

        rt_min, rt, rt_max = s_data

        fig['layout']['shapes'] = [
            {
                'line': {'color': 'black', 'width': 3},
                'type': 'line',
                'x0': rt,
                'x1': rt,
                'xref': 'x',
                'y0': 0,
                'y1': 1,
                'yref': 'y domain'
            },
            {
                'fillcolor': 'green',
                'line': {'width': 0},
                'opacity': 0.1,
                'type': 'rect',
                'x0': rt_min,
                'x1': rt_max,
                'xref': 'x',
                'y0': 0,
                'y1': 1,
                'yref': 'y domain'
            }
        ]
        fig['layout']['annotations'][0] = {
            'bgcolor': 'white',
            'font': {'color': 'black', 'size': 14, 'weight': 'bold'},
            'showarrow': True,
            'text': 'RT',
            'x': rt,
            'xanchor': 'left',
            'xref': 'x',
            'y': 1,
            'yanchor': 'top',
            'yref': 'y domain'
        }

        slider_data['value'] = {'rt_min': rt_min, 'rt': rt, 'rt_max': rt_max}
        has_changes = slider_data['value'] != slider_reference_data['value']
        buttons_style = {
            'visibility': 'visible' if has_changes else 'hidden',
            'opacity': '1' if has_changes else '0',
            'transition': 'opacity 0.3s ease-in-out'
        }
        return fig, slider_data, buttons_style

    @app.callback(
        Output('slider-data', 'data'),
        Output('rt-range-slider', 'min'),
        Output('rt-range-slider', 'max'),
        Output('rt-range-slider', 'step'),
        Output('rt-range-slider', 'value'),
        Output('rt-range-slider', 'pushable'),
        Output('rt-range-slider', 'tooltip'),
        Output('rt-range-slider', 'marks'),
        Output('chromatogram-view-options-drawer', 'visible', allow_duplicate=True),

        Input("chromatogram-view-plot", "relayoutData"),
        Input('slider-reference-data', 'data'),
        State('chromatogram-preview-log-y', 'checked'),
        State('chromatogram-view-log-y', 'checked'),
        State('slider-data', 'data'),
        prevent_initial_call=True
    )
    def set_chromatogram_view_options(relayout, slider_reference_data, global_log_scale, log_scale, slider_data):

        if not relayout or not slider_reference_data:
            raise PreventUpdate

        ctx = dash.callback_context
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if prop_id == 'slider-reference-data':
            if not slider_data:
                slider_data = slider_reference_data.copy()
            else:
                slider_data['value'] = slider_reference_data['value']
            value = [vc for vc, sv in zip(slider_data['value'].values(), slider_data['v_comp'].values()) if sv]
            s_min = slider_data['min']
            s_max = slider_data['max']
            if s_max - s_min > 1:
                slider_data['step'] = 0.1
                decimals = 1
            else:
                slider_data['step'] = 0.01
                decimals = 2
            slider_data['pushable'] = slider_data['step']
            slider_data['marks'] = {
                float(i): str(round(i, decimals))
                for i in np.linspace(s_min, s_max, 6)
            }
            return (slider_data, slider_data['min'], slider_data['max'], slider_data['step'], value,
                    slider_data['pushable'], {"placement": "bottom", "always_visible": False}, slider_data['marks'],
                    dash.no_update)
        else:
            if not relayout:
                raise PreventUpdate
            if relayout.get('xaxis.range[0]', None):
                x_min_relayout = round(relayout.get('xaxis.range[0]'), 1)
            elif relayout.get('xaxis.range', None):
                x_min_relayout = round(relayout.get('xaxis.range')[0], 1)
            elif relayout.get('xaxis.autorange', False):
                x_min_relayout = slider_reference_data['min']
            elif relayout.get('autosize', False):
                x_min_relayout = slider_reference_data['min']
            else:
                x_min_relayout = None

            if relayout.get('xaxis.range[1]', None):
                x_max_relayout = round(relayout.get('xaxis.range[1]'), 1)
            elif relayout.get('xaxis.range', None):
                x_max_relayout = round(relayout.get('xaxis.range')[1], 1)
            elif relayout.get('xaxis.autorange', False):
                x_max_relayout = slider_reference_data['max']
            elif relayout.get('autosize', False):
                x_max_relayout = slider_reference_data['max']
            else:
                x_max_relayout = None

            if x_min_relayout is None or x_max_relayout is None:
                return dash.no_update

            if not slider_data:
                slider_data = slider_reference_data.copy()

            slider_repr = {'rt_min': False, 'rt': False, 'rt_max': False}
            if x_min_relayout < slider_data['value']['rt_min']:
                slider_repr['rt_min'] = True
            if x_min_relayout < slider_data['value']['rt'] < x_max_relayout:
                slider_repr['rt'] = True
            if x_max_relayout > slider_data['value']['rt_max']:
                slider_repr['rt_max'] = True

            s_min = x_min_relayout
            s_max = x_max_relayout
            if s_max - s_min > 1:
                step = 0.1
                decimals = 1
            else:
                step = 0.01
                decimals = 2

            new_slider_data = slider_data.copy()
            new_slider_data['min'] = x_min_relayout
            new_slider_data['max'] = x_max_relayout
            new_slider_data['marks'] = {
                float(i): str(round(i, decimals))
                for i in np.linspace(s_min, s_max, 6)
            },
            new_slider_data['v_comp'] = slider_repr
            new_slider_data['step'] = step
            new_slider_data['pushable'] = step

            value = [vc for vc, sv in zip(slider_data['value'].values(), slider_repr.values()) if sv]

            return (new_slider_data, x_min_relayout, x_max_relayout, step, value, step,
                    {"placement": "bottom", "always_visible": False},
                    {float(i): str(round(i, decimals)) for i in np.linspace(s_min, s_max, 6)},
                    False
                    )

    ############# VIEW END #######################################

    ############# COMPUTE CHROMATOGRAM BEGIN #####################################
    @app.callback(
        Output("compute-chromatogram-modal", "visible"),
        Output("chromatogram-warning", "style"),
        Output("chromatogram-compute-ram", "max"),
        Output("chromatogram-compute-ram-item", "help"),

        Input("compute-chromatograms-btn", "nClicks"),
        State('chromatogram-compute-ram', 'value'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def open_compute_chromatogram_modal(nClicks, ram_value, wdir):
        if not nClicks:
            raise PreventUpdate

        computed_chromatograms = 0
        # check if some chromatogram was computed
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update

            chromatograms = conn.execute("SELECT COUNT(*) FROM chromatograms").fetchone()
            if chromatograms:
                computed_chromatograms = chromatograms[0]

        style = {'display': 'block'} if computed_chromatograms else {'display': 'none'}

        ram_max = round(psutil.virtual_memory().available / (1024 ** 3), 1)
        help = f"Selected {ram_value}GB / {ram_max}GB available RAM"


        return True, style, ram_max, help

    @app.callback(
        Output('chromatograms', 'data'),
        Output('compute-chromatogram-modal', 'visible', allow_duplicate=True),

        Input('compute-chromatogram-modal', 'okCounts'),
        State("chromatograms-recompute-ms1", "checked"),
        State("chromatograms-recompute-ms2", "checked"),
        State("chromatogram-compute-cpu", "value"),
        State("chromatogram-compute-ram", "value"),
        State("wdir", "data"),
        background=True,
        running=[
            (Output('chromatogram-processing-progress-container', 'style'), {
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "flexDirection": "column",
                "minWidth": "200px",
                "maxWidth": "400px",
                "margin": "auto",
                'height': "60vh"
            }, {'display': 'none'}),
            (Output("chromatogram-compute-options-container", "style"), {'display': 'none'}, {'display': 'flex'}),

            (Output('compute-chromatogram-modal', 'confirmAutoSpin'), True, False),
            (Output('compute-chromatogram-modal', 'cancelButtonProps'), {'disabled': True},
             {'disabled': False}),
            (Output('compute-chromatogram-modal', 'confirmLoading'), True, False),
        ],
        progress=[
            Output("chromatogram-processing-progress", "percent"),
        ],
        prevent_initial_call=True
    )
    def compute_chromatograms(set_progress, okCounts, recompute_ms1, recompute_ms2, n_cpus, ram, wdir):

        if not okCounts:
            raise PreventUpdate

        with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as con:
            if con is None:
                return "Could not connect to database."
            start = time.perf_counter()
            compute_and_insert_chromatograms_from_ms_data(con, set_progress, recompute_ms1=recompute_ms1,
                                                          recompute_ms2=recompute_ms2)
            # compute_and_insert_chromatograms_iteratively(con, set_progress=set_progress)
            print(f"Chromatograms computed in {time.perf_counter() - start:.2f} seconds")
        return True, False

    ############# COMPUTE CHROMATOGRAM END #######################################

    @app.callback(
        Output('rt-values-span', 'children'),
        Input('slider-data', 'data'),
        prevent_initial_call=True
    )
    def rt_representation(slider_data):
        if not slider_data:
            raise PreventUpdate

        rt_min, rt, rt_max = slider_data['value'].values()

        return fac.AntdFlex(
            [
                fac.AntdCompact(
                    [
                        fac.AntdText('RT-min:', strong=True),
                        fac.AntdText(f"{rt_min:.1f}s", code=True),
                    ],
                    style={'visibility': slider_data['v_comp']['rt_min']}
                ),
                fac.AntdCompact(
                    [
                        fac.AntdText('RT:', strong=True),
                        fac.AntdText(f"{rt:.1f}s", code=True),
                    ],
                    style={'visibility': slider_data['v_comp']['rt']}
                ),
                fac.AntdCompact(
                    [
                        fac.AntdText('RT-max:', strong=True),
                        fac.AntdText(f"{rt_max:.1f}s", code=True),
                    ],
                    style={'visibility': slider_data['v_comp']['rt_max']}
                ),
            ],
            justify='space-between',
            align='center',
            style={'width': '100%', 'padding': '0 25px'}
        )

    @app.callback(
        # Output('chromatogram-view-modal', 'visible', allow_duplicate=True),
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('action-buttons-container', 'style', allow_duplicate=True),
        Output('slider-reference-data', 'data', allow_duplicate=True),

        Input('save-btn', 'nClicks'),
        State('target-preview-clicked', 'data'),
        State('slider-data', 'data'),
        State('slider-reference-data', 'data'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def save_changes(save_clicks, target_clicked, slider_data, slider_reference, wdir):
        if not save_clicks:
            raise PreventUpdate
        rt_min, rt_, rt_max = slider_data['value'].values()

        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update
            conn.execute("UPDATE targets SET rt_min = ?, rt = ?, rt_max = ? "
                         "WHERE peak_label = ?", (rt_min, rt_, rt_max, target_clicked))
        buttons_style = {
            'visibility': 'hidden',
            'opacity': '0',
            'transition': 'opacity 0.3s ease-in-out'
        }
        slider_reference['value'].update(slider_data['value'])
        return (
            fac.AntdNotification(message=f"RT values saved...\n",
                                 type="success",
                                 placement='bottom',
                                 showProgress=True,
                                 stack=True
                                 ),
            buttons_style,
            slider_reference
        )

    @app.callback(
        # only save the current values stored in slider-reference-data since this will shut all the actions
        Output('slider-reference-data', 'data', allow_duplicate=True),

        Input('reset-btn', 'nClicks'),
        State('slider-reference-data', 'data'),
        prevent_initial_call=True
    )
    def reset_changes(reset_clicks, slider_reference):

        if not reset_clicks:
            raise PreventUpdate
        return slider_reference

    # @app.callback(
    #     Output('delete-targets-modal', 'visible'),
    #     Output('delete-targets-modal', 'children'),
    #     Output('delete-target-clicked', 'children'),
    #     Input({'type': 'delete-target-btn', 'index': ALL}, 'nClicks'),
    #     prevent_initial_call=True
    # )
    # def show_delete_modal(delete_clicks):
    #
    #     ctx = dash.callback_context
    #     if not delete_clicks or not ctx.triggered:
    #         raise PreventUpdate
    #     ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
    #     trigger_id = ctx_trigger['index']
    #
    #     if len(dash.callback_context.triggered) > 1:
    #         raise PreventUpdate
    #
    #     return True, fac.AntdParagraph(f"Are you sure you want to delete `{trigger_id}` target?"), trigger_id
    #
    # @app.callback(
    #     Output({"index": "pko-drop-target-output", "type": "output"}, "children"),
    #     Input('delete-targets-modal', 'okCounts'),
    #     Input('delete-targets-modal', 'cancelCounts'),
    #     Input('delete-targets-modal', 'closeCounts'),
    #     Input('delete-target-clicked', 'children'),
    #     State("wdir", "data"),
    #     prevent_initial_call=True
    # )
    # def plk_delete(okCounts, cancelCounts, closeCounts, target, wdir):
    #     if not okCounts or cancelCounts or closeCounts:
    #         raise PreventUpdate
    #     targets = T.get_targets(wdir)
    #     targets = targets[targets['peak_label'] != target]
    #
    #     T.write_targets(targets, wdir)
    #     return fac.AntdNotification(message=f"{target} deleted",
    #                                 type="success",
    #                                 duration=3,
    #                                 placement='bottom',
    #                                 showProgress=True,
    #                                 stack=True
    #                                 )

    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),

        Input({'type': 'bookmark-target-card', 'index': ALL}, 'value'),
        State({'type': 'target-card-preview', 'index': ALL}, 'data-target'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def bookmark_target(bookmarks, targets, wdir):
        # TODO: change bookmark to bool since the AntdRate component returns an int and the db require a bool
        ctx = dash.callback_context
        if not ctx.triggered or len(dash.callback_context.triggered) > 1:
            raise PreventUpdate

        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_id = ctx_trigger['index']

        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update
            conn.execute("UPDATE targets SET bookmark = ? WHERE peak_label = ?", [bool(bookmarks[trigger_id]),
                                                                                  targets[trigger_id]])
        return fac.AntdNotification(message=f"Target {targets[trigger_id]} has been "
                                            f"{'' if bookmarks[trigger_id] else 'un'}bookmarked",
                                    duration=3,
                                    placement='bottom',
                                    type="success",
                                    showProgress=True,
                                    stack=True)
