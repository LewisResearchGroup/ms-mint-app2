import json
from os import cpu_count

import dash
import feffery_antd_components as fac
import math
import numpy as np
import plotly.graph_objects as go
import psutil
import time
from pathlib import Path
from dash import html, dcc, Patch
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from ..duckdb_manager import duckdb_connection, compute_chromatograms_in_batches
from ..plugin_interface import PluginInterface
from ..tools import sparsify_chrom, proportional_min1_selection

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


MAX_NUM_CARDS = 50

_layout = fac.AntdLayout(
    [
        fac.AntdHeader(
            [
                fac.AntdFlex(
                    [
                        fac.AntdTitle(
                            'Optimization', level=4, style={'margin': '0', 'whiteSpace': 'nowrap'}
                        ),
                        fac.AntdIcon(
                            id='optimization-tour-icon',
                            icon='pi-info',
                            style={"cursor": "pointer", 'paddingLeft': '10px'},
                        ),
                        fac.AntdFlex(
                            [
                                fac.AntdButton(
                                    'Compute Chromatograms',
                                    id='compute-chromatograms-btn',
                                    style={'textTransform': 'uppercase'},
                                ),
                                fac.AntdSelect(
                                    id='targets-select',
                                    options=[],
                                    mode="multiple",
                                    autoSpin=True,
                                    maxTagCount="responsive",
                                    style={"width": "450px"},
                                    locale="en-us",
                                )
                            ],
                            justify='space-between',
                            style={"margin": "0 40px 0 50px", 'width': '100%'},
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
                                    },
                                    id='sample-selection'
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
                                                        value='mz_mean',
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
                                    style={'overflow': 'hidden'},
                                    id='sidebar-options'
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
                                                                        'margin': '0 0 14px 0',
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
                                                                ),
                                                                fac.AntdTooltip(
                                                                    fac.AntdButton(
                                                                        icon=fac.AntdIcon(icon='antd-delete'),
                                                                        type='text',
                                                                        size='small',
                                                                        id={'type': 'delete-target-card', 'index': i},
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
                                                                            'zIndex': 20,
                                                                            'opacity': '0.1',
                                                                            'transition': 'opacity 0.3s ease'
                                                                        },
                                                                        className='peak-action-button',
                                                                    ),
                                                                    title='Delete target',
                                                                    color='red',
                                                                    placement='bottom',
                                                                ),
                                                            ],
                                                            **{'data-target': None},
                                                            className='is-hidden'
                                                        ) for i in range(MAX_NUM_CARDS)
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
                                                    pageSizeOptions=[4, 10, 20, 50],
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
                            ],
                            layout='inline'
                        ),
                        fac.AntdDivider('Configuration'),
                        fac.AntdForm(
                            [
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='chromatogram-compute-cpu',
                                        defaultValue=cpu_count() // 2,
                                        min=1,
                                        max=cpu_count() - 2,
                                    ),
                                    label='CPU:',
                                    hasFeedback=True,
                                    help=f"Selected {cpu_count() // 2} / {cpu_count()} cpus"
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='chromatogram-compute-ram',
                                        value=round(psutil.virtual_memory().available * 0.5 / (1024 ** 3), 1),
                                        min=1,
                                        precision=1,
                                        step=0.1,
                                        suffix='GB'
                                    ),
                                    label='RAM:',
                                    hasFeedback=True,
                                    id='chromatogram-compute-ram-item',
                                    help=f"Selected "
                                         f"{round(psutil.virtual_memory().available * 0.5 / (1024 ** 3), 1)}GB / "
                                         f"{round(psutil.virtual_memory().available / (1024 ** 3), 1)}GB available RAM"
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='chromatogram-compute-batch-size',
                                        defaultValue=1000,
                                        min=50,
                                        step=50,
                                    ),
                                    label='Batch Size:',
                                    tooltip='Number of pairs to process in each batch. This will affect the memory '
                                            'usage, progress and processing time.',
                                ),
                            ],
                            layout='inline'
                        ),

                        fac.AntdDivider(),
                        fac.AntdAlert(
                            message='There are no targets selected. The chromatograms will be computed for all targets.',
                            type='info',
                            showIcon=True,
                            id='chromatogram-targets-info',
                        ),
                        fac.AntdAlert(
                            message='There are already computed chromatograms',
                            type='warning',
                            showIcon=True,
                            id='chromatogram-warning',
                            style={'display': 'none'},
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
            width=900,
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
                fac.AntdLayout(
                    [
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
                            ],
                            id='chromatogram-view-container',
                            className='ant-layout-content css-1v28nim',
                            style={
                                # 'position': 'relative',
                                'overflowX': 'hidden',
                                'background': 'white',
                                'alignContent': 'center'
                            },
                        ),
                        fac.AntdSider(
                            [
                                fac.AntdTitle(
                                    'Options',
                                    level=4,
                                    style={'margin': '0px', 'marginBottom': '8px'}
                                ),
                                fac.AntdSpace(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    'Intensity Scale:',
                                                    style={
                                                        'display': 'inline-block',
                                                        'width': '170px',
                                                        'textAlign': 'left',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                html.Div(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-view-log-y',
                                                        checked=False,
                                                        checkedChildren='Log',
                                                        unCheckedChildren='Lin'
                                                    ),
                                                    style={
                                                        'width': '110px',
                                                        'display': 'flex',
                                                        'justifyContent': 'flex-start'
                                                    }
                                                ),
                                            ],
                                            style={'display': 'flex', 'alignItems': 'center'}
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    'Legend Behavior:',
                                                    style={
                                                        'display': 'inline-block',
                                                        'width': '170px',
                                                        'textAlign': 'left',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                html.Div(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-view-groupclick',
                                                        checked=False,
                                                        checkedChildren='Grp',
                                                        unCheckedChildren='Sng'
                                                    ),
                                                    style={
                                                        'width': '110px',
                                                        'display': 'flex',
                                                        'justifyContent': 'flex-start'
                                                    }
                                                ),
                                            ],
                                            style={'display': 'flex', 'alignItems': 'center'}
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    'Notes:',
                                                    style={
                                                        'display': 'inline-block',
                                                        'width': '170px',
                                                        'textAlign': 'left',
                                                        'paddingRight': '8px',
                                                        'marginBottom': '6px'
                                                    }
                                                ),
                                                fac.AntdInput(
                                                    id='target-note',
                                                    allowClear=True,
                                                    mode='text-area',
                                                    autoSize={'minRows': 2, 'maxRows': 4},
                                                    style={'width': '225px'},
                                                    placeholder='Add notes for this target'
                                                ),
                                            ],
                                            style={
                                                'display': 'flex',
                                                'flexDirection': 'column',
                                                'alignItems': 'flex-start',
                                                'width': '100%'
                                            }
                                        ),
                                    ],
                                    direction='vertical',
                                    size='small',
                                    style={'alignItems': 'flex-start'}
                                )
                            ],
                            collapsible=False,
                            theme='light',
                            width=250,
                            style={'marginLeft': 20,
                                   'background': 'white'}
                        )
                    ],
                    style={'background': 'white'}
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

        dcc.Store(id='chromatograms', data=True),
        dcc.Store(id='drop-chromatogram'),
        dcc.Store(id="delete-target-clicked"),
        dcc.Store(id='chromatogram-view-plot-max'),
        dcc.Store(id='update-chromatograms', data=False),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Optimization Tour',
                    'description': 'Quick guide to compute and preview chromatograms.',
                },
                {
                    'title': 'Compute chromatograms',
                    'description': 'Runs chromatogram extraction for selected targets and files.',
                    'targetSelector': '#compute-chromatograms-btn'
                },
                {
                    'title': 'Target filter',
                    'description': 'Narrow targets shown below using this selector.',
                    'targetSelector': '#targets-select'
                },
                {
                    'title': 'Sample selection',
                    'description': 'Selected samples to show in the cards.',
                    'targetSelector': '#sample-selection'
                },
                {
                    'title': 'Options',
                    'description': 'Configure MS type, selection, sorting, log scale, and plot dimensions.',
                    'targetSelector': '#sidebar-options'
                },
            ],
            id='optimization-tour',
            open=False,
            current=0,
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Need help?',
                    'description': 'Click the info icon to open a quick tour of Optimization.',
                    'targetSelector': '#optimization-tour-icon',
                },
            ],
            mask=False,
            placement='rightTop',
            open=False,
            current=0,
            id='optimization-tour-hint',
            className='targets-tour-hint',
        ),
        dcc.Store(id='optimization-tour-hint-store', data={'open': True}, storage_type='session'),
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
                }, 250);  // 150ms suele ser suficiente
            });
        }
        """,
        Output("rslider", "style"),
        Input("chromatogram-view-plot", "relayoutData"),
        Input("chromatogram-view-modal", 'visible'),
        prevent_initial_call=True
    )

    @app.callback(
        Output('optimization-tour', 'current'),
        Output('optimization-tour', 'open'),
        Input('optimization-tour-icon', 'nClicks'),
        prevent_initial_call=True,
    )
    def optimization_tour_open(n_clicks):
        return 0, True

    @app.callback(
        Output('optimization-tour-hint', 'open'),
        Output('optimization-tour-hint', 'current'),
        Input('optimization-tour-hint-store', 'data'),
    )
    def optimization_hint_sync(store_data):
        if not store_data:
            raise PreventUpdate
        return store_data.get('open', True), 0

    @app.callback(
        Output('optimization-tour-hint-store', 'data'),
        Input('optimization-tour-hint', 'closeCounts'),
        Input('optimization-tour-icon', 'nClicks'),
        State('optimization-tour-hint-store', 'data'),
        prevent_initial_call=True,
    )
    def optimization_hide_hint(close_counts, n_clicks, store_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'optimization-tour-icon':
            return {'open': False}

        if close_counts:
            return {'open': False}

        return store_data or {'open': True}

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
                                     list({'title': label, 'key': label}) as children,
                                     list(label)                          as checked_keys
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
                return [], [], [], {'display': 'none'}, {'display': 'block'}

            if prop_id in ['mark-tree-action', 'section-context']:
                quotas, checked_keys = proportional_min1_selection(df, 'sample_type', 'checked_keys', 50, 12345)
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
        } for _ in range(MAX_NUM_CARDS)]

    ############# GRAPH OPTIONS END #######################################

    ############# PREVIEW BEGIN #####################################
    @app.callback(
        Output({'type': 'target-card-preview', 'index': ALL}, 'data-target'),
        Output({'type': 'graph', 'index': ALL}, 'figure'),
        Output({'type': 'bookmark-target-card', 'index': ALL}, 'value'),
        Output('chromatogram-preview-pagination', 'total'),
        Output('chromatograms-dummy-output', 'children'),
        Output('targets-select', 'options'),

        Input('chromatograms', 'data'),
        Input('chromatogram-preview-pagination', 'current'),
        Input('chromatogram-preview-pagination', 'pageSize'),
        Input('sample-type-tree', 'checkedKeys'),
        Input('chromatogram-preview-log-y', 'checked'),
        Input('chromatogram-preview-filter-bookmark', 'value'),
        Input('chromatogram-preview-filter-ms-type', 'value'),
        Input('chromatogram-preview-order', 'value'),
        Input('drop-chromatogram', 'data'),
        Input('targets-select', 'value'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def chromatograms_preview(chromatograms, current_page, page_size, checkedkeys, log_scale, selection_bookmark,
                              selection_ms_type, targets_order, dropped_target, selected_targets, wdir):

        ctx = dash.callback_context
        if 'targets-select' in ctx.triggered[0]['prop_id'] and selected_targets:
            current_page = 1

        start_idx = (current_page - 1) * page_size
        t1 = time.perf_counter()

        with duckdb_connection(wdir) as conn:
            all_targets = conn.execute("""
                                       SELECT peak_label
                                       from targets t
                                       WHERE (
                                           CASE
                                               WHEN ? = 'ms1' THEN t.ms_type = 'ms1'
                                               WHEN ? = 'ms2' THEN t.ms_type = 'ms2'
                                               ELSE TRUE
                                               END
                                           )
                                         AND (
                                           CASE
                                               WHEN ? = 'Bookmarked' THEN t.bookmark = TRUE
                                               WHEN ? = 'Unmarked' THEN t.bookmark = FALSE
                                               ELSE TRUE -- 'all' case 
                                               END
                                           )
                                         AND (
                                           t.peak_selection IS TRUE
                                               OR NOT EXISTS (SELECT 1
                                                              FROM targets t1
                                                              WHERE t1.peak_selection IS TRUE)
                                           )
                                         AND (
                                           (SELECT COUNT(*) FROM unnest(?::VARCHAR[])) = 0
                                               OR peak_label IN (SELECT unnest(?::VARCHAR[]))
                                           )
                                       """, [selection_ms_type, selection_ms_type,
                                             selection_bookmark, selection_bookmark,
                                            selected_targets, selected_targets]).fetchall()

            all_targets = [row[0] for row in all_targets]

            # Autosave the current targets table to the workspace data folder so UI edits are persisted.
            try:
                data_dir = Path(wdir) / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                targets_df = conn.execute("SELECT * FROM targets").df()
                targets_df.to_csv(data_dir / "targets_backup.csv", index=False)
            except Exception:
                pass

            query = f"""
                                WITH picked_samples AS (
                                    SELECT ms_file_label, color, label
                                    FROM samples
                                    WHERE use_for_optimization = TRUE
                                      AND ms_file_label IN (SELECT unnest(?::VARCHAR[]))
                                ),
                                picked_targets AS (
                                    SELECT 
                                           t.peak_label,
                                           t.ms_type,
                                           t.bookmark,
                                           t.rt_min,
                                           t.rt_max,
                                           t.rt,
                                           t.intensity_threshold,
                                           t.mz_mean,
                                           t.filterLine
                                    FROM targets t
                                    WHERE (
                                        CASE
                                            WHEN ? = 'ms1' THEN t.ms_type = 'ms1'
                                            WHEN ? = 'ms2' THEN t.ms_type = 'ms2'
                                            ELSE TRUE
                                        END
                                    )
                                    AND (
                                        CASE
                                            WHEN ? = 'Bookmarked' THEN t.bookmark = TRUE
                                            WHEN ? = 'Unmarked' THEN t.bookmark = FALSE
                                            ELSE TRUE -- 'all' case 
                                        END
                                    )
                                    AND (
                                        t.peak_selection IS TRUE
                                        OR NOT EXISTS (
                                                SELECT 1 
                                                FROM targets t1
                                                WHERE t1.peak_selection IS TRUE
                                            )
                                    )
                                    AND (
                                        (SELECT COUNT(*) FROM unnest(?::VARCHAR[])) = 0
                                        OR peak_label IN (SELECT unnest(?::VARCHAR[]))
                                    )
                                    ORDER BY 
                                        CASE WHEN ? = 'mz_mean' THEN mz_mean END,
                                        peak_label
                                    -- 3) order by
                                    LIMIT ? -- 1) limit
                                        OFFSET ? -- 2) offset
                                ),
                                base AS (
                                    SELECT 
                                       c.*,
                                       s.color,
                                       s.label,
                                       t.rt_min,
                                       t.rt_max,
                                       t.rt,
                                       t.intensity_threshold,
                                       t.mz_mean,
                                       t.bookmark,  -- Add additional fields as needed
                                       t.ms_type,
                                       t.filterLine
                                    FROM chromatograms c
                                          JOIN picked_samples s USING (ms_file_label)
                                          JOIN picked_targets t USING (peak_label)
                                ),
                                     -- Pair up (scan_time[i], intensity[i]) into a list of structs
                                filtered AS (
                                    SELECT peak_label,
                                           ms_file_label,
                                           color,
                                           label,
                                           rt_min,
                                           rt_max,
                                           rt,
                                           mz_mean,
                                           bookmark,
                                           ms_type,
                                           filterLine,
                                           list_filter(
                                               list_transform(
                                                   range(1, len(scan_time) + 1),
                                                   i -> struct_pack(
                                                       t := list_extract(scan_time, i),
                                                       i := list_extract(intensity, i)
                                                   )
                                               ),
                                               p -> p.t >= rt_min AND p.t <= rt_max
                                                     AND p.i >= COALESCE(intensity_threshold, 0)
                                           ) AS pairs_in
                                    FROM base
                                ),
                                final AS (
                                    SELECT peak_label,
                                           ms_file_label,
                                           color,
                                           label,
                                           mz_mean,
                                           rt_min,
                                           rt_max,
                                           rt,
                                           bookmark,
                                           ms_type,
                                           filterLine,
                                           list_transform(pairs_in, p -> p.t) AS scan_time_sliced,
                                           list_transform(pairs_in, p -> p.i) AS intensity_sliced
                                    FROM filtered
                                )
                                SELECT *
                                FROM final
                                ORDER BY CASE WHEN ? = 'mz_mean' THEN mz_mean END,
                                        peak_label;
                                """
            df = conn.execute(query, [checkedkeys, selection_ms_type, selection_ms_type,
                                      selection_bookmark, selection_bookmark,
                                      selected_targets, selected_targets,
                                      targets_order, page_size, start_idx, targets_order]
                              ).pl()

        titles = []
        figures = []
        bookmarks = []

        for peak_label_data, peak_data in df.group_by(
                ['peak_label', 'ms_type', 'bookmark', 'rt_min', 'rt_max', 'rt', 'mz_mean', 'filterLine'],
                maintain_order=True):
            peak_label, ms_type, bookmark, rt_min, rt_max, rt, mz_mean, filterLine = peak_label_data

            titles.append(peak_label)
            bookmarks.append(int(bookmark))  # convert bool to int

            fig = Patch()
            traces = []
            for i, row in enumerate(peak_data.iter_rows(named=True)):
                scan_time_sparse, intensity_sparse = sparsify_chrom(
                    row['scan_time_sliced'], row['intensity_sliced']
                )
                traces.append({
                    'type': 'scatter',
                    'mode': 'lines',
                    'x': scan_time_sparse,
                    'y': intensity_sparse,
                    'name': row['label'] or row['ms_file_label'],
                    'line': {'color': row['color'], 'width': 1.5},
                })

            fig['data'] = traces

            fig['layout']['shapes'] = [
                {
                    'line': {'color': 'black', 'width': 1.5, 'dash': 'dashdot'},
                    'type': 'line',
                    'x0': rt,
                    'x1': rt,
                    'xref': 'x',
                    'y0': 0,
                    'y1': 1,
                    'yref': 'y domain'
                }
            ]
            fig['layout']['template'] = 'plotly_white'

            filter_type = (f"mz_mean = {mz_mean}"
                           if ms_type == 'ms1'
                           else f"{filterLine}")
            fig['layout']['title'] = dict(
                text=f"{peak_label}<br><sup>{filter_type}</sup>",
                font={'size': 14},
                y=0.90,
                yanchor='top'
            )

            fig['layout']['xaxis']['title'] = dict(text="Retention Time [s]", font={'size': 10})
            fig['layout']['xaxis']['autorange'] = False
            fig['layout']['xaxis']['fixedrange'] = True
            fig['layout']['xaxis']['range'] = [rt_min, rt_max]

            fig['layout']['yaxis']['title'] = dict(text="Intensity", font={'size': 10})
            fig['layout']['yaxis']['autorange'] = True
            fig['layout']['yaxis']['automargin'] = True
            fig['layout']['yaxis']['tickformat'] = "~s"

            if log_scale:
                fig['layout']['yaxis']['type'] = 'log'
                fig['layout']['yaxis']['dtick'] = 1  # Only show powers of 10 to avoid cramped labels
                fig['layout']['yaxis']['tickfont'] = {'size': 9}
            else:
                fig['layout']['yaxis']['type'] = 'linear'
                fig['layout']['yaxis']['tickfont'] = {'size': 9}

            fig["layout"]["showlegend"] = False
            fig['layout']['margin'] = dict(l=45, r=5, t=55, b=30)
            # fig['layout']['uirevision'] = f"xr_{peak_label}"
            figures.append(fig)

        titles.extend([None for _ in range(MAX_NUM_CARDS - len(figures))])
        figures.extend([{} for _ in range(MAX_NUM_CARDS - len(figures))])
        bookmarks.extend([0 for _ in range(MAX_NUM_CARDS - len(bookmarks))])

        if 'targets-select' in ctx.triggered[0]['prop_id']:
            targets_select_options = dash.no_update
        else:
            targets_select_options = all_targets

        print(f"{time.perf_counter() - t1 = }")
        return titles, figures, bookmarks, len(all_targets), [], targets_select_options

    @app.callback(
        Output({'type': 'target-card-preview', 'index': ALL}, 'className'),
        Output('chromatogram-preview-container', 'style'),
        Output('chromatogram-preview-empty', 'style'),

        Input({'type': 'graph', 'index': ALL}, 'figure'),
        State({'type': 'target-card-preview', 'index': ALL}, 'className'),
        prevent_initial_call=True
    )
    def toggle_card_visibility(figures, current_class):
        visible_fig = 0
        cards_classes = []

        for i, figure in enumerate(figures):
            if figure:
                cc = current_class[i].split() if current_class[i] else []
                cc.remove('is-hidden') if 'is-hidden' in cc else None
                cards_classes.append(' '.join(cc))
                visible_fig += 1
            else:
                cc = current_class[i].split() if current_class[i] else []
                cc.append('is-hidden') if 'is-hidden' not in cc else None
                cards_classes.append(' '.join(cc))

        show_empty = {'display': 'block'} if visible_fig == 0 else {'display': 'none'}
        show_space = {'display': 'none'} if visible_fig == 0 else {'display': 'block'}
        return cards_classes, show_space, show_empty

    ############# PREVIEW END #######################################

    ############# VIEW MODAL BEGIN #####################################
    @app.callback(
        Output('target-preview-clicked', 'data'),

        Input({'type': 'target-card-preview', 'index': ALL}, 'nClicks'),
        Input({'type': 'bookmark-target-card', 'index': ALL}, 'value'),
        Input({'type': 'delete-target-card', 'index': ALL}, 'nClicks'),
        State({'type': 'target-card-preview', 'index': ALL}, 'data-target'),
        prevent_initial_call=True
    )
    def open_chromatogram_view_modal(card_preview_clicks, bookmark_target_clicks, delete_target_clicks, data_target):
        if not card_preview_clicks:
            raise PreventUpdate

        ctx = dash.callback_context
        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_type = ctx_trigger['type']

        if len(ctx.triggered) > 1 or trigger_type != 'target-card-preview':
            raise PreventUpdate

        prop_id = ctx_trigger['index']
        return data_target[prop_id]

    @app.callback(
        Output('chromatogram-view-modal', 'visible'),
        Output('slider-reference-data', 'data', allow_duplicate=True),
        Output('chromatograms', 'data', allow_duplicate=True),

        Input('target-preview-clicked', 'data'),
        Input('chromatogram-view-close', 'nClicks'),
        Input('confirm-unsave-modal', 'okCounts'),
        State('update-chromatograms', 'data'),
        State('target-note', 'value'),

        State('slider-reference-data', 'data'),
        State('slider-data', 'data'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def handle_modal_open_close(target_clicked, close_clicks, close_without_save_clicks, update_chromatograms,
                                target_note, slider_ref, slider_data, wdir):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'target-preview-clicked':
            return True, dash.no_update, dash.no_update
            # if not has_changes, close it
        elif trigger_id == 'chromatogram-view-close':
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    return dash.no_update, dash.no_update, dash.no_update
                conn.execute("UPDATE targets SET notes = ? WHERE peak_label = ?",
                             (target_note, target_clicked))

            if slider_ref and slider_data and slider_ref['value'] == slider_data['value']:
                return False, None, update_chromatograms or dash.no_update
            # if it has_changes, don't close it
            return dash.no_update, dash.no_update, dash.no_update
        elif trigger_id == 'confirm-unsave-modal':
            # Close modal without saving changes
            if close_without_save_clicks:
                return False, None, update_chromatograms or dash.no_update

        return dash.no_update, dash.no_update, dash.no_update

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
        Output('chromatogram-view-groupclick', 'checked', allow_duplicate=True),
        Output('target-note', 'value', allow_duplicate=True),

        Input('target-preview-clicked', 'data'),
        State('chromatogram-preview-log-y', 'checked'),
        State('sample-type-tree', 'checkedKeys'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def chromatogram_view_modal(target_clicked, log_scale, checkedKeys, wdir):

        with duckdb_connection(wdir) as conn:
            d = conn.execute("SELECT rt, rt_min, rt_max, COALESCE(notes, '') FROM targets WHERE peak_label = ?",
                             [target_clicked]).fetchall()
            rt, rt_min, rt_max, note = d[0] if d else (None, None, None, '')

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

        MAX_TRACES = 200

        if len(chrom_df) <= MAX_TRACES:
            # ------------------------------------
            # MODO NORMAL: una trace por cromatograma
            # ------------------------------------
            for i, row in enumerate(chrom_df.iter_rows(named=True)):

                scan_time_sparse, intensity_sparse = sparsify_chrom(
                    row['scan_time_sliced'], row['intensity_sliced'], w=1, baseline=1.0, eps=0.0
                )

                trace = {
                    'type': 'scattergl',
                    'mode': 'lines',
                    'x': scan_time_sparse,
                    'y': intensity_sparse,
                    'line': {'color': row['color']},
                    'name': row['label'] or row['ms_file_label'],
                    'legendgroup': row['sample_type'],
                    'hoverlabel': dict(namelength=-1)
                }

                if row['sample_type'] not in legend_groups:
                    trace['legendgrouptitle'] = {'text': row['sample_type']}
                    legend_groups.add(row['sample_type'])

                traces.append(trace)

                x_min = min(x_min, row['scan_time_min_in_range'])
                x_max = max(x_max, row['scan_time_max_in_range'])
                y_min = min(y_min, row['intensity_min_in_range'])
                y_max = max(y_max, row['intensity_max_in_range'])

        else:
            # ------------------------------------
            # MODO REDUCIDO: una trace por sample_type
            # ------------------------------------
            grouped = chrom_df.group_by('sample_type')

            for g, group_df in grouped:
                xs = []
                ys = []
                sample_color = None
                group_name = str(g[0])

                for row in group_df.iter_rows(named=True):

                    scan_time_sparse, intensity_sparse = sparsify_chrom(
                        row['scan_time_sliced'], row['intensity_sliced'], w=1, baseline=1.0, eps=0.0
                    )

                    # Concat y dejar None para separar cromatogramas
                    xs.extend(scan_time_sparse)
                    ys.extend(intensity_sparse)
                    xs.append(None)
                    ys.append(None)

                    if sample_color is None:
                        sample_color = row['color']

                    # expandir rangos globales
                    x_min = min(x_min, row['scan_time_min_in_range'])
                    x_max = max(x_max, row['scan_time_max_in_range'])
                    y_min = min(y_min, row['intensity_min_in_range'])
                    y_max = max(y_max, row['intensity_max_in_range'])

                # Crear una sola trace por sample_type
                trace = {
                    'type': 'scattergl',
                    'mode': 'lines',
                    'x': xs,
                    'y': ys,
                    'line': {'color': sample_color}, 'name': f"{group_name} (merged)",
                    'legendgroup': group_name,
                    'hoverlabel': dict(namelength=-1),
                    'legendgrouptitle': {'text': group_name}
                }

                traces.append(trace)

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

        t_xmin = (rt_min - (rt_max - rt_min)) if rt_min else 0
        nx_min = max(t_xmin, x_min)

        t_xmax = (rt_max + (rt_max - rt_min)) if rt_max else 0
        nx_max = min(t_xmax, x_max)

        fig['layout']['xaxis']['range'] = [nx_min, nx_max]
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
            'min': nx_min,
            'max': nx_max,
            'pushable': 1,
            'step': 1,
            'tooltip': None,
            'marks': None,
            'value': {'rt_min': rt_min, 'rt': rt, 'rt_max': rt_max},
            'v_comp': {'rt_min': True, 'rt': True, 'rt_max': True}
        }

        print(f"{time.perf_counter() - t1 = }")
        return fig, target_clicked, False, s_data, None, [y_min, y_max], log_scale, False, note

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
        has_changes = False
        if slider_data and slider_reference_data:
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
                    slider_data['pushable'], {"placement": "bottom", "always_visible": False}, slider_data['marks'])
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
                    {float(i): str(round(i, decimals)) for i in np.linspace(s_min, s_max, 6)}
                    )

    ############# VIEW END #######################################

    ############# COMPUTE CHROMATOGRAM BEGIN #####################################
    @app.callback(
        Output("compute-chromatogram-modal", "visible"),
        Output("chromatogram-warning", "style"),
        Output("chromatogram-warning", "message"),
        Output("chromatogram-targets-info", "message"),
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
        selected_targets = 0
        # check if some chromatogram was computed
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update

            chromatograms = conn.execute("SELECT COUNT(*) FROM chromatograms").fetchone()
            if chromatograms:
                computed_chromatograms = chromatograms[0]

            targets = conn.execute("SELECT COUNT(*) FROM targets WHERE peak_selection = TRUE").fetchone()
            if targets:
                selected_targets = targets[0]

        warning_style = {'display': 'flex'} if computed_chromatograms else {'display': 'none'}
        warning_message = f"There are already computed {computed_chromatograms} chromatograms" if computed_chromatograms else ""

        ram_max = round(psutil.virtual_memory().available / (1024 ** 3), 1)
        help = f"Selected {ram_value}GB / {ram_max}GB available RAM"
        target_message = (f'Selected {selected_targets} targets'
                          if selected_targets
                          else 'There are no targets selected. The chromatograms will be computed for all targets.')
        return True, warning_style, warning_message, target_message, ram_max, help

    @app.callback(
        Output('chromatograms', 'data'),
        Output('compute-chromatogram-modal', 'visible', allow_duplicate=True),

        Input('compute-chromatogram-modal', 'okCounts'),
        State("chromatograms-recompute-ms1", "checked"),
        State("chromatograms-recompute-ms2", "checked"),
        State("chromatogram-compute-cpu", "value"),
        State("chromatogram-compute-ram", "value"),
        State('chromatogram-compute-batch-size', "value"),
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
    def compute_chromatograms(set_progress, okCounts, recompute_ms1, recompute_ms2, n_cpus, ram, batch_size, wdir):

        if not okCounts:
            raise PreventUpdate

        with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as con:
            if con is None:
                return "Could not connect to database."
            start = time.perf_counter()
            compute_chromatograms_in_batches(wdir, use_for_optimization=True, batch_size=batch_size,
                                             set_progress=set_progress, recompute_ms1=recompute_ms1,
                                             recompute_ms2=recompute_ms2, n_cpus=n_cpus, ram=ram)
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
        Output('update-chromatograms', 'data'),

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
            slider_reference,
            True
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

    @app.callback(
        Output('delete-targets-modal', 'visible'),
        Output('delete-targets-modal', 'children'),
        Output('delete-target-clicked', 'children'),

        Input({'type': 'delete-target-card', 'index': ALL}, 'nClicks'),
        State({'type': 'target-card-preview', 'index': ALL}, 'data-target'),
        prevent_initial_call=True
    )
    def show_delete_modal(delete_clicks, data_target):

        ctx = dash.callback_context
        if not delete_clicks or not ctx.triggered:
            raise PreventUpdate
        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])

        if len(dash.callback_context.triggered) > 1:
            raise PreventUpdate

        prop_id = ctx_trigger['index']
        return True, fac.AntdParagraph(f"Are you sure you want to delete `{data_target[prop_id]}` target?"), \
        data_target[prop_id]

    #
    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('drop-chromatogram', 'data'),

        Input('delete-targets-modal', 'okCounts'),
        State('delete-target-clicked', 'children'),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def delete_targets_chromatograms(okCounts, target, wdir):
        if not okCounts:
            raise PreventUpdate
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update
            conn.execute("DELETE FROM chromatograms WHERE peak_label = ?", [target])
            conn.execute("DELETE FROM targets WHERE peak_label = ?", [target])
            conn.execute("DELETE FROM results WHERE peak_label = ?", [target])

        return (fac.AntdNotification(message=f"{target} chromatograms deleted",
                                     type="success",
                                     duration=3,
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True
                                     ),
                True)

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
