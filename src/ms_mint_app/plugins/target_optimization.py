import json

import dash
import feffery_antd_components as fac
import math
import numpy as np
import plotly.graph_objects as go
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
                                            showIcon=True
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
                                                        id='chromatogram-preview-filter',
                                                        options=['All', 'Bookmarked', 'Unmarked'],
                                                        value='All',
                                                        placeholder='Select filter',
                                                        style={'width': '100%'},
                                                        allowClear=False,
                                                        locale="en-us",
                                                    ),
                                                    label='Select:',
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
                        fac.AntdForm(
                            [
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
                                fac.AntdAlert(
                                    message='There are already computed chromatograms',
                                    type='warning',
                                    showIcon=True,
                                    id='chromatogram-warning',
                                    style={'display': 'none'},
                                )
                            ]
                        ),
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
                                            label='Intensity Log scale:',
                                        )
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
            const root = document.getElementById("chromatogram-view-plot");
            if (!root) return window.dash_clientside.no_update;

            const bg = root.querySelector("div > div > div svg > g.draglayer > g.xy > rect");
            if (!bg) return window.dash_clientside.no_update;

            const rw = root.offsetWidth;
            const pl = bg.x.baseVal.value - 25;
            const w = bg.width.baseVal.value + 50;

            if (!isFinite(w) || w <= 0) return window.dash_clientside.no_update;
            return {"marginLeft": pl + "px", "width": w + "px"};
        }
        """,
        Output("rslider", "style"),
        Input("chromatogram-view-plot", "relayoutData"),
        Input("chromatogram-view-modal", 'visible'),
        prevent_initial_call=True
    )

    @app.callback(
        Output('sample-type-tree', 'treeData'),
        Output('sample-type-tree', 'checkedKeys'),
        Output('sample-type-tree', 'expandedKeys'),
        Input("chromatograms", "data"),
        State("wdir", "data"),
    )
    def update_sample_type_tree(chromatograms, wdir):
        if not chromatograms:
            raise PreventUpdate

        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update, dash.no_update, dash.no_update
            metadata = conn.execute("SELECT label, sample_type, use_for_optimization FROM samples_metadata").df()

        if metadata.empty:
            return dash.no_update, dash.no_update, dash.no_update

        sample_type_dict = metadata.groupby('sample_type')['label'].apply(list).to_dict()
        checked_keys = metadata[metadata['use_for_optimization']]['label'].tolist()

        tree_data = []
        for k, v in sample_type_dict.items():
            children = [{'title': ms, 'key': ms} for ms in v if ms in checked_keys]
            if children:
                tree_data.append({'title': k, 'key': k, 'children': children})

        expanded_keys = [
            k
            for k, v in sample_type_dict.items()
            if any(ms in checked_keys for ms in v)
        ]
        return tree_data, checked_keys, expanded_keys

    @app.callback(
        Output('pko-info-modal', 'visible'),
        Output('pko-info-modal', 'title'),
        Output('has-unsaved-changes', 'data'),

        Input("pko-image-clicked", "children"),
        Input('close-modal-btn', 'nClicks'), # nClicks (antd) != n_clicks (dash)
        Input('close-without-save-btn', 'nClicks'),
        Input('sample-type-tree', 'checkedKeys'),
        State('has-unsaved-changes', 'data'),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def handle_modal_open_close(image_clicked, close_clicks, close_without_save_clicks,
                                checkedkeys, has_changes, wdir):
        """
        Handle opening and closing of the modal dialog for display the selected target plot.
        :param image_clicked:
        :param close_clicks:
        :param close_without_save_clicks:
        :param saved_values:
        :param has_changes:
        :return:
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'pko-image-clicked':

            return (True, image_clicked,
                    # no_update,
                    # saved_values,
                    False)
        elif trigger_id == 'close-without-save-btn':
            # Close modal without saving changes
            return (False, no_update,
                    # no_update,
                    # no_update,
                    False)
        elif trigger_id == 'close-modal-btn':
            # if not has_changes, close it
            if not has_changes:
                return (False, no_update,
                        # no_update,
                        # no_update,
                        False)
            # if it has_changes, don't close it
            return (no_update, no_update,
                    # no_update,
                    # no_update,
                    no_update)
        else:
            print(f"modal {trigger_id = }")

        return (no_update, no_update,
                # no_update,
                # no_update,
                no_update)

    @app.callback(
        Output('confirm-modal', 'visible'),
        Input('close-modal-btn', 'nClicks'),
        State('has-unsaved-changes', 'data'),
        prevent_initial_call=True
    )
    def show_confirm_modal(close_clicks, has_changes):
        return bool(close_clicks and has_changes)

    @app.callback(
        Output('pko-plot', 'figure'),
        Output('action-buttons-container', 'style'),
        Output("rt-range-slider", "min"),
        Output("rt-range-slider", "max"),
        Output("rt-range-slider", "value"),
        Output("rt-range-slider", "marks"),
        Output("rt-range-slider", "tooltip"),
        Output('has-unsaved-changes', 'data', allow_duplicate=True),

        Input("pko-image-clicked", "children"),
        Input("pko-figure-options", "value"),
        Input('rt-range-slider', 'value'),
        Input('sample-type-tree', 'checkedKeys'),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def update_from_slider(image_clicked, options, slider_values, checkedkeys, wdir):

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # reset slider values when is not an update to avoid use stored values from other plots
        if trigger_id != 'rt-range-slider':
            slider_values = None
        if image_clicked is None:
            raise PreventUpdate

        # use peak_label instead of image_clicked, more intuitive
        peak_label = image_clicked

        with duckdb_connection(wdir) as conn:

            target_df = conn.execute("SELECT rt, rt_min, rt_max, peak_label FROM targets "
                                     "WHERE peak_label = ?", [peak_label]).df()

            print(f"{target_df = }")
            rt, rt_min, rt_max, label = target_df.iloc[0]

            # round values to int because the slider steps are int
            orig_values = [int(rt_min), int(rt), int(rt_max)]

            if trigger_id == 'pko-image-clicked':
                samples_for_optimization = conn.execute("SELECT ms_file_label, color, label FROM samples_metadata "
                                                        "WHERE use_for_optimization = True"
                                                        ).df()

                rt_slider_min, st_slider, rt_slider_max = slider_values or orig_values

                query = f"""SELECT 
                                sel.label,
                                sel.color,
                                chr.scan_time, 
                                chr.intensity
                            FROM chromatograms AS chr
                            JOIN samples_for_optimization AS sel ON sel.ms_file_label = chr.ms_file_label
                            WHERE chr.peak_label = ?"""

                chrom_df = conn.execute(query, [peak_label]).df()

                fig, slider_min, slider_max = create_plot(
                    chromatogram_data=chrom_df,
                    checkedkeys=checkedkeys,
                    rt=st_slider,
                    rt_min=rt_slider_min,
                    rt_max=rt_slider_max,
                    title=peak_label,
                    log='log' in options,
                )

                slider_marks = {i: str(i) for i in range(slider_min, slider_max, (int(slider_max - slider_min)//5))}
                has_changes = slider_values != orig_values if slider_values else False

                buttons_style = {
                    'visibility': 'visible' if has_changes else 'hidden',
                    'opacity': '1' if has_changes else '0',
                    'transition': 'opacity 0.3s ease-in-out'
                }
                return (fig, buttons_style, slider_min, slider_max,
                        [rt_slider_min, st_slider, rt_slider_max],
                        slider_marks,
                        {"placement": "bottom", "always_visible": False},
                        has_changes,
                        )

            elif trigger_id == 'rt-range-slider':
                patch_fig = Patch()
                if slider_values[0]:
                    patch_fig['layout']['shapes'][1]['x0'] = slider_values[0]
                if slider_values[2]:
                    patch_fig['layout']['shapes'][1]['x1'] = slider_values[2]
                if slider_values[1]:
                    patch_fig['layout']['shapes'][0]['x0'] = slider_values[1]
                    patch_fig['layout']['shapes'][0]['x1'] = slider_values[1]
                    patch_fig['layout']['annotations'][0]['x'] = slider_values[1]

                has_changes = slider_values != orig_values if slider_values else False
                buttons_style = {
                    'visibility': 'visible' if has_changes else 'hidden',
                    'opacity': '1' if has_changes else '0',
                    'transition': 'opacity 0.3s ease-in-out'
                }

                return (patch_fig, buttons_style, no_update, no_update, no_update, no_update, no_update, has_changes)
        return (no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update)

    @app.callback(
        Output('rt-values-span', 'children'),
        Input('rt-range-slider', 'value'),
        prevent_initial_call=True
    )
    def rt_representation(values):
        rt_min, rt, rt_max = values

        return [html.Span(f"RT-min: {rt_min}", className="rt-value"),
                html.Span(f"RT: {rt}", className="rt-value"),
                html.Span(f"RT-max: {rt_max}", className="rt-value")]


    @app.callback(
        Output('saved-range', 'data', allow_duplicate=True),
        Output('pko-info-modal', 'visible', allow_duplicate=True),
        Output('notifications-container', 'children'),
        Output('has-unsaved-changes', 'data', allow_duplicate=True),
        Output('action-buttons-container', 'style', allow_duplicate=True),

        Input("pko-image-clicked", "children"),
        Input('save-btn', 'nClicks'),
        Input("rt-range-slider", "value"),
        State("has-unsaved-changes", "data"),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def save_changes(image_clicked, save_clicks, current_values, unsaved_changes, wdir):

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'save-btn' and unsaved_changes:
            targets = T.get_targets(wdir)
            rt_min, rt, rt_max = current_values
            peak_label = targets.loc[targets['peak_label'] == image_clicked, "peak_label"].iloc[0]
            T.update_targets(wdir, peak_label, rt_min, rt_max, rt)

            buttons_style = {
                'visibility': 'hidden',
                'opacity': '0',
                'transition': 'opacity 0.3s ease-in-out'
            }

            return (
                current_values,  # Actualizar valores guardados
                False,  # Cerrar modal
                fac.AntdNotification(message=f"RT values saved...\n"
                                             f"RT-min: {current_values[0]:.2f}<br>"
                                             f"RT: {current_values[1]:.2f}<br> "
                                             f"RT-max: {current_values[2]:.2f}",
                                     type="success",
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True
                                     ),
                False,
                buttons_style
            )
        raise PreventUpdate

    @app.callback(
        Output('rt-range-slider', 'value', allow_duplicate=True),
        Output('has-unsaved-changes', 'data', allow_duplicate=True),
        Output('action-buttons-container', 'style', allow_duplicate=True),

        Input("pko-image-clicked", "children"),
        Input('reset-btn', 'nClicks'),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def reset_changes(image_clicked, reset_clicks, wdir):
        """
        Reset the slider values to the saved RT values
        :param image_clicked:
        :param reset_clicks:
        :param wdir:
        :return:
        """
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'reset-btn' and reset_clicks:
            targets = T.get_targets(wdir)
            rt_min, rt, rt_max = targets.loc[targets['peak_label'] == image_clicked, ["rt_min", "rt", "rt_max"]].iloc[0]

            buttons_style = {
                'visibility': 'hidden',
                'opacity': '0',
                'transition': 'opacity 0.3s ease-in-out'
            }
            return [int(rt_min), int(rt), int(rt_max)], False, buttons_style
        return no_update, no_update, no_update

    @app.callback(
        Output('confirm-modal', 'visible', allow_duplicate=True),
        Input('stay-btn', 'nClicks'),
        Input('close-without-save-btn', 'nClicks'),
        prevent_initial_call=True
    )
    def handle_confirm_modal(stay_clicks, close_clicks):
        return False

    # callback to compute or read the chromatograms
    @app.callback(
        Output('chromatograms', 'data'),
        # Output('chromatograms-progress-container', 'style'),
        Input('tab', 'value'),
        Input('chromatograms', 'data'),
        State("wdir", "data"),
        background=True,
        running=[
            (
                    Output("chromatograms-progress-container", "style"),
                    {"display": "flex",
                     "justifyContent": "center",
                     "alignItems": "center",
                     "flexDirection": "column",
                     "minWidth": "200px",
                     "maxWidth": "500px",
                     "margin": "auto",},
                    {"display": "none"},
            ),
        ],
        progress=[Output("chromatograms-progress", "percent"),
                  ],
        prevent_initial_call=True
    )
    def compute_chromatograms(set_progress, tab, chromatograms, wdir):

        ctx = dash.callback_context
        prop_id = ctx.triggered[0]['prop_id']
        print(f"{prop_id = }")

        if tab != 'Optimization':
            raise PreventUpdate

        with DDB.duckdb_connection(wdir) as con:
            if con is None:
                return "Could not connect to database."
            start = time.perf_counter()
            # DDB.compute_and_insert_chromatograms_from_ms_data(con)
            DDB.compute_and_insert_chromatograms_iteratively(con, set_progress=set_progress)
            print(f"Chromatograms computed in {time.perf_counter() - start:.2f} seconds")
            df = con.execute("SELECT * FROM chromatograms WHERE peak_label = 'Acetoacetic acid'").df()
            import pandas as pd
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(f"{df.head() = }")
        return True

    @app.callback(
        Output("pko-peak-preview-images", "children"),
        Output('plot-preview-pagination', 'total'),

        Input('chromatograms', 'data'),

        Input('plot-preview-pagination', 'current'),
        Input('plot-preview-pagination', 'pageSize'),

        Input('sample-type-tree', 'checkedKeys'),
        Input("pko-figure-options", "value"),
        Input('card-plot-selection', 'value'),
        Input({"index": "pko-drop-target-output", "type": "output"}, "children"),
        State("wdir", "data"),
    )
    def peak_preview(chromatograms, current_page, page_size, checkedkeys, options, selection, dropped_target, wdir):
        logging.info(f'Create peak previews {wdir}')

        if not chromatograms:
            raise PreventUpdate

        with duckdb_connection(wdir) as conn:
            ms_files_selection = conn.execute(f"SELECT ms_file_label, color, label FROM samples_metadata WHERE label IN"
                                              f" {checkedkeys}").df()

            if ms_files_selection.empty:
                return (fac.AntdNotification(
                    message="No chromatogram to preview",
                    description='No files selected for peak optimization in MS-Files tab. Please, mark some files in the '
                            'Sample Type tree. "use_for_optimization" is use as the initial selection only',
                    type="error", duration=None,
                    placement='bottom',
                    showProgress=True,
                    stack=True
                ), no_update)
            logging.info(f"Using {len(ms_files_selection)} files for peak preview. ({ms_files_selection = })")

            if selection == 'all':
                total_targets_query = "SELECT COUNT(*) FROM targets WHERE preselected_processing = TRUE"
                query = "SELECT * FROM targets WHERE preselected_processing = TRUE LIMIT ? OFFSET ?"
            elif selection == 'bookmarked':
                total_targets_query = "SELECT COUNT(*) FROM targets WHERE bookmark = TRUE"
                query = "SELECT * FROM targets WHERE bookmark = TRUE LIMIT ? OFFSET ?"
            else:
                total_targets_query = "SELECT COUNT(*) FROM targets WHERE bookmark = FALSE"
                query = "SELECT * FROM targets WHERE bookmark = FALSE LIMIT ? OFFSET ?"

            total_targets = conn.execute(total_targets_query).fetchone()[0]
            target_selection = conn.execute(query, [page_size, (current_page - 1) * page_size]).df()

            plots = []
            for _, row in target_selection.iterrows():
                peak_label, mz_mean, mz_width, rt, rt_min, rt_max, bookmark, score = row[
                    ["peak_label", "mz_mean", "mz_width", "rt", "rt_min", "rt_max", 'bookmark', 'score']
                ]

                query = f"""SELECT 
                                sel.label,
                                sel.color,
                                chr.scan_time, 
                                chr.intensity
                            FROM chromatograms AS chr
                            JOIN ms_files_selection AS sel ON sel.ms_file_label = chr.ms_file_label
                            WHERE chr.peak_label = ?"""

                all_chrom_df = conn.execute(query, [peak_label]).df()
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                print(f"{all_chrom_df.head(5) = }")

                fig = create_preview_peakshape_plotly(
                    all_chrom_df,
                    rt,
                    rt_min,
                    rt_max,
                    peak_label=peak_label,
                    log='log' in options
                )

                plots.append(
                    fac.AntdCard([
                        dcc.Graph(
                            id={'type': 'graph-card-preview', 'index': f"{peak_label}-{uuid.uuid4().hex[:6]}"},
                            figure=fig,
                            style={'height': '150px', 'width': '200px', 'margin': '0px'},
                            config={
                                'displayModeBar': False,
                                'staticPlot': True,  # Totalmente estático para máxima performance
                                'doubleClick': False,
                                'showTips': False,
                                'responsive': False  # Tamaño fijo
                            },
                        ),
                        fac.AntdTooltip(
                            fac.AntdButton(
                                icon=fac.AntdIcon(icon='antd-delete'),
                                type='text',
                                size='small',
                                id={'type': 'delete-target-btn', 'index': peak_label},
                                style={
                                    'padding': '4px',
                                    'minWidth': '24px',
                                    'height': '24px',
                                    'borderRadius': '50%',
                                    'background': 'rgba(0, 0, 0, 0.5)',
                                    'color': 'white',
                                    'position': 'absolute',
                                    'bottom': '8px',
                                    'right': '8px',
                                    'zIndex': 10,
                                    'opacity': '0',
                                    'transition': 'opacity 0.3s ease'
                                },
                                className='peak-action-button',
                            ),
                            title='Delete target',
                            color='red',
                            placement='bottom',
                        ),
                        fac.AntdRate(
                            id={'type': 'rate-target-card', 'index': peak_label},
                            count=1,
                            defaultValue=0,
                            value=int(bookmark),
                            allowHalf=False,
                            tooltips=['Bookmark this target'],
                            style={'position': 'absolute', 'top': '8px', 'right': '8px', 'zIndex': 10},
                        )],
                        id={'type': 'plot-card-preview', 'index': peak_label},
                        style={'cursor': 'pointer'},
                        styles={'header': {'display': 'none'}, 'body': {'padding': '5px'}},
                        hoverable=True,
                        className='peak-card-container',
                    )
                )
        return plots, total_targets

    @app.callback(
        Output("pko-image-clicked", "children"),
        [Input({"type": "plot-card-preview", "index": ALL}, "nClicks")],
        [Input({"type": "delete-target-btn", "index": ALL}, "nClicks")],
        [Input({"type": "rate-target-card", "index": ALL}, "value")],
        prevent_initial_call=True,
    )
    def pko_image_clicked(ndx, dt_ndx, b_ndx):

        if ndx is None or len(ndx) == 0:
            raise PreventUpdate

        ctx = dash.callback_context

        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_id = ctx_trigger['index']
        trigger_type = ctx_trigger['type']

        if len(ctx.triggered) > 1 or trigger_type != "plot-card-preview":
            raise PreventUpdate
        return trigger_id

    @app.callback(
        Output('delete-modal', 'visible'),
        Output('delete-modal', 'children'),
        Output('delete-target-clicked', 'children'),
        Input({"type": "delete-target-btn", "index": ALL}, 'nClicks'),
        prevent_initial_call=True
    )
    def show_delete_modal(delete_clicks):

        ctx = dash.callback_context
        if not delete_clicks or not ctx.triggered:
            raise PreventUpdate
        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_id = ctx_trigger['index']

        if len(dash.callback_context.triggered) > 1:
            raise PreventUpdate

        return True, fac.AntdParagraph(f"Are you sure you want to delete `{trigger_id}` target?"), trigger_id

    @app.callback(
        Output({"index": "pko-drop-target-output", "type": "output"}, "children"),
        Input('delete-modal', 'okCounts'),
        Input('delete-modal', 'cancelCounts'),
        Input('delete-modal', 'closeCounts'),
        Input('delete-target-clicked', 'children'),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def plk_delete(okCounts, cancelCounts, closeCounts, target, wdir):
        if not okCounts or cancelCounts or closeCounts:
            raise PreventUpdate
        targets = T.get_targets(wdir)
        targets = targets[targets['peak_label'] != target]

        T.write_targets(targets, wdir)
        return fac.AntdNotification(message=f"{target} deleted",
                                    type="success",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    )

    @app.callback(
        Output('notifications-container', "children", allow_duplicate=True),
        Input({"type": "rate-target-card", "index": ALL}, "value"),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def bookmark_target(value, wdir):
        # TODO: change bookmark to bool since the AntdRate component returns an int and the db require a bool
        ctx = dash.callback_context
        if not ctx.triggered or len(dash.callback_context.triggered) > 1:
            raise PreventUpdate

        targets = T.get_targets(wdir)

        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_id = ctx_trigger['index']

        if targets['bookmark'].values.tolist() == value:
            raise PreventUpdate

        targets['bookmark'] = value
        T.write_targets(targets, wdir)
        new_value = targets.loc[targets['peak_label'] == trigger_id, 'bookmark'].values[0]

        return fac.AntdNotification(message=f"Target {trigger_id} {'' if new_value else 'un'}bookmarked", duration=3,
                                    placement='bottom', type="success", showProgress=True, stack=True)




