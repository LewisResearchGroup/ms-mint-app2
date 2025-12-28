import json
import logging
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
from ..plugins.analysis_tools.trace_helper import generate_chromatogram_traces, calculate_rt_alignment, calculate_shifts_per_sample_type
from .workspaces import activate_workspace_logging

_label = "Optimization"

logger = logging.getLogger(__name__)


class TargetOptimizationPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 6
        logger.info(f'Initiated {_label} plugin')

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
DEFAULT_GRAPH_WIDTH = 250
DEFAULT_GRAPH_HEIGHT = 180

# Use a valid empty Plotly figure (not `{}`) so container resizes/redraws don't
# trigger Plotly's "_doPlot" warning for graphs that haven't been plotted yet.
EMPTY_PLOTLY_FIGURE = {"data": [], "layout": {"template": "plotly_white"}}

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
                                fac.AntdTooltip(
                                    fac.AntdButton(
                                        'Compute Chromatograms',
                                        id='compute-chromatograms-btn',
                                        style={'textTransform': 'uppercase', "margin": "0 10px"},
                                    ),
                                    title="Calculate chromatograms from the MS files and Targets.",
                                    placement="bottom"
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
                            style={"margin": "0 40px 0 10px", 'width': '100%'},
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
                                    fac.AntdSpin(
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
                                        style={'height': '100%'}
                                    ),
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
                                                                value=DEFAULT_GRAPH_WIDTH,
                                                                defaultValue=DEFAULT_GRAPH_WIDTH,
                                                                min=180,
                                                                max=1400
                                                            ),
                                                            fac.AntdInputNumber(
                                                                id='chromatogram-graph-height',
                                                                value=DEFAULT_GRAPH_HEIGHT,
                                                                defaultValue=DEFAULT_GRAPH_HEIGHT,
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
                                                fac.AntdTooltip(
                                                    fac.AntdButton(
                                                        # 'Apply',
                                                        id='chromatogram-graph-button',
                                                        icon=fac.AntdIcon(icon='pi-broom', style={'fontSize': 20}),
                                                        # type='primary'
                                                    ),
                                                    title='Update graph size and clean plots',
                                                    placement="bottom"
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
                                                    pageSizeOptions=[4, 9, 20, 50],
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
        html.Div(id="optimization-notifications-container"),
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
                        fac.AntdText(
                            id='chromatogram-processing-stage',
                            style={'marginBottom': '0.5rem'},
                        ),
                        fac.AntdProgress(
                            id='chromatogram-processing-progress',
                            percent=0,
                        ),
                        fac.AntdText(
                            id='chromatogram-processing-detail',
                            type='secondary',
                            style={
                                'marginTop': '0.5rem',
                                'marginBottom': '0.75rem',
                            },
                        ),
                        fac.AntdButton(
                            'Cancel',
                            id='cancel-chromatogram-processing',
                            style={
                                'alignText': 'center',
                                'marginTop': '0.25rem',
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
                                    config={'displayModeBar': True, 'edits': {'shapePosition': True}},
                                    style={'width': '100%', 'height': '600px'}
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
                                                html.Div(
                                                    [
                                                        html.Span('Megatrace:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='Merge traces to improve performance'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                html.Div(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-view-megatrace',
                                                        checked=True,
                                                        checkedChildren='On',
                                                        unCheckedChildren='Off'
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
                                                html.Div(
                                                    [
                                                        html.Span('Intensity Scale:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='Linear vs Logarithmic scale'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
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
                                                html.Div(
                                                    [
                                                        html.Span('Legend Behavior:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='Single vs Group toggle'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
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
                                                html.Div(
                                                    [
                                                        html.Span('Edit RT-span:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='Unlock to edit RT range'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                html.Div(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-view-lock-range',
                                                        checked=False,
                                                        checkedChildren='Lock',
                                                        unCheckedChildren='Edit'
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
                                                html.Div(
                                                    [
                                                        html.Span('RT Alignment:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='Align chromatograms by peak apex'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                html.Div(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-view-rt-align',
                                                        checked=False,
                                                        checkedChildren='On',
                                                        unCheckedChildren='Off'
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
                                                html.Div(
                                                    [
                                                        html.Span('Notes:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='User notes for target'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
                                                        'paddingRight': '8px'
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
                                                'width': '100%',
                                                'marginTop': '6px'
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
                                    "Delete target",
                                    id="delete-target-from-modal",
                                    danger=True,
                                    type="dashed",
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
        dcc.Store(id='rt-alignment-data'),  # Stores RT alignment info for saving to notes
        dcc.Store(id='target-preview-clicked'),

        dcc.Store(id='chromatograms', data=True),
        dcc.Store(id='drop-chromatogram'),
        dcc.Store(id="delete-target-clicked"),
        dcc.Store(id='chromatogram-view-plot-max'),
        dcc.Store(id='chromatogram-view-plot-points'),
        dcc.Store(id='update-chromatograms', data=False),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'Follow this tutorial to compute chromatograms and tune targets.',
                },
                {
                    'title': 'Compute chromatograms',
                    'description': 'Extract chromatograms for the selected targets/files.',
                    'targetSelector': '#compute-chromatograms-btn'
                },
                {
                    'title': 'Review cards',
                    'description': 'Scroll the cards below to inspect chromatograms and edit target bounds directly.',
                    'targetSelector': '#chromatogram-preview-container'
                },
                {
                    'title': 'Select samples',
                    'description': 'Choose which samples are plotted in the cards for quick comparison.',
                    'targetSelector': '#sample-selection'
                },
                {
                    'title': 'Select targets',
                    'description': 'Filter which targets appear in the optimization cards.',
                    'targetSelector': '#targets-select'
                },
                {
                    'title': 'Tune options',
                    'description': 'Adjust MS type, target ordering, log scale, and plot sizing to aid review.',
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
            style={
                'background': '#ffffff',
                'border': '0.5px solid #1677ff',
                'boxShadow': '0 6px 16px rgba(0,0,0,0.15), 0 0 0 1px rgba(22,119,255,0.2)',
                'opacity': 1,
            },
        ),
        dcc.Store(id='optimization-tour-hint-store', data={'open': False}, storage_type='local'),
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

    @app.callback(
        Output("optimization-notifications-container", "children"),
        Input('section-context', 'data'),
        Input("wdir", "data"),
    )
    def warn_missing_workspace(section_context, wdir):
        if not section_context or section_context.get('page') != 'Optimization':
            return dash.no_update
        if wdir:
            return []
        return fac.AntdNotification(
            message="Activate a workspace",
            description="Select or create a workspace before using Optimization.",
            type="warning",
            duration=4,
            placement='bottom',
            showProgress=True,
            stack=True,
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

        if not section_context or section_context.get('page') != 'Optimization':
            raise PreventUpdate
        if not wdir:
            return [], [], [], {'display': 'none'}, {'display': 'block'}

        with duckdb_connection(wdir) as conn:
            if conn is None:
                return [], [], [], {'display': 'none'}, {'display': 'block'}
            
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

            if prop_id == 'mark-tree-action':
                logger.debug(f"{df['checked_keys'].values = }")
                checked_keys = [v for value in df['checked_keys'].values for v in value]
            elif prop_id == 'section-context':
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
                    for row in df.to_dict('records')
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
        Output('chromatogram-graph-width', 'value'),
        Output('chromatogram-graph-height', 'value'),
        Input('chromatogram-graph-button', 'nClicks'),
        Input('chromatogram-preview-pagination', 'pageSize'),
        State('chromatogram-graph-width', 'value'),
        State('chromatogram-graph-height', 'value'),
        prevent_initial_call=True
    )
    def set_chromatogram_graph_size(nClicks, page_size, width, height):
        """
        Auto-tune preview plot size based on cards per page, while still allowing manual overrides.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        def autosize(ps, w, h):
            if ps == 4:
                return 500, 350
            if ps == 9:
                return 350, 220
            if ps in (20, 50):
                return DEFAULT_GRAPH_WIDTH, DEFAULT_GRAPH_HEIGHT
            return w, h

        width = width or DEFAULT_GRAPH_WIDTH
        height = height or DEFAULT_GRAPH_HEIGHT

        if trigger == 'chromatogram-preview-pagination':
            width, height = autosize(page_size, width, height)
        elif trigger != 'chromatogram-graph-button':
            raise PreventUpdate

        return ([{
            'width': width,
            'height': height,
            'margin': '0px',
        } for _ in range(MAX_NUM_CARDS)],
                width,
                height)

    ############# GRAPH OPTIONS END #######################################

    ############# PREVIEW BEGIN #####################################
    @app.callback(
        Output({'type': 'target-card-preview', 'index': ALL}, 'data-target'),
        Output({'type': 'graph', 'index': ALL}, 'figure'),
        Output({'type': 'bookmark-target-card', 'index': ALL}, 'value'),
        Output('chromatogram-preview-pagination', 'total'),
        Output('chromatogram-preview-pagination', 'current', allow_duplicate=True),
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
        State('chromatograms-dummy-output', 'children'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def chromatograms_preview(chromatograms, current_page, page_size, checkedkeys, log_scale, selection_bookmark,
                              selection_ms_type, targets_order, dropped_target, selected_targets,
                              preview_y_range, wdir):

        ctx = dash.callback_context
        if 'targets-select' in ctx.triggered[0]['prop_id'] and selected_targets:
            current_page = 1
        if not wdir:
            raise PreventUpdate

        page_size = page_size or 1
        start_idx = (current_page - 1) * page_size
        t1 = time.perf_counter()

        with duckdb_connection(wdir) as conn:
            if conn is None:
                # If the DB is locked/unavailable, keep current preview as-is
                raise PreventUpdate
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
            
            # Adjust current_page if it's beyond the available pages (e.g., after deleting all targets on current page)
            total_targets = len(all_targets)
            max_page = max(1, math.ceil(total_targets / page_size)) if total_targets else 1
            if current_page > max_page:
                current_page = max_page
                start_idx = (current_page - 1) * page_size

            # Autosave the current targets table to the workspace data folder, but throttle I/O.
            try:
                data_dir = Path(wdir) / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                backup_path = data_dir / "targets_backup.csv"
                should_write = True
                if backup_path.exists():
                    last_write = backup_path.stat().st_mtime
                    # Avoid hammering disk on every preview refresh.
                    should_write = (time.time() - last_write) > 30
                if should_write:
                    targets_df = conn.execute("SELECT * FROM targets").df()
                    targets_df.to_csv(backup_path, index=False)
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
        if not isinstance(preview_y_range, dict):
            preview_y_range = {}
        updated_preview_y_range = dict(preview_y_range)

        for peak_label_data, peak_data in df.group_by(
                ['peak_label', 'ms_type', 'bookmark', 'rt_min', 'rt_max', 'rt', 'mz_mean', 'filterLine'],
                maintain_order=True):
            peak_label, ms_type, bookmark, rt_min, rt_max, rt, mz_mean, filterLine = peak_label_data

            titles.append(peak_label)
            bookmarks.append(int(bookmark))  # convert bool to int

            fig = Patch()
            traces = []
            y_max = 0.0
            y_min_pos = None
            for i, row in enumerate(peak_data.iter_rows(named=True)):
                scan_time_sparse, intensity_sparse = sparsify_chrom(
                    row['scan_time_sliced'], row['intensity_sliced']
                )
                if len(intensity_sparse):
                    local_max = max(intensity_sparse)
                    if local_max > y_max:
                        y_max = local_max
                    local_min_pos = min((v for v in intensity_sparse if v > 0), default=None)
                    if local_min_pos is not None:
                        if y_min_pos is None or local_min_pos < y_min_pos:
                            y_min_pos = local_min_pos
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

            y_key = f"{peak_label}|{ms_type}"
            prev_range = preview_y_range.get(y_key, {})
            prev_y_max = prev_range.get("y_max")
            prev_y_min = prev_range.get("y_min_pos")
            use_prev = False
            if prev_y_max and y_max:
                diff_ratio = abs(y_max - prev_y_max) / max(prev_y_max, 1.0)
                use_prev = diff_ratio < 0.05
            if use_prev:
                y_max_use = prev_y_max
                y_min_use = prev_y_min if prev_y_min else y_min_pos
            else:
                y_max_use = y_max
                y_min_use = y_min_pos

            if log_scale:
                fig['layout']['yaxis']['type'] = 'log'
                fig['layout']['yaxis']['dtick'] = 1  # Only show powers of 10 to avoid cramped labels
                fig['layout']['yaxis']['tickfont'] = {'size': 9}
                if y_max_use and y_max_use > 0:
                    y_min_use = y_min_use if y_min_use and y_min_use > 0 else max(y_max_use * 1e-6, 1e-6)
                    fig['layout']['yaxis']['range'] = [math.log10(y_min_use), math.log10(y_max_use)]
                    fig['layout']['yaxis']['autorange'] = False
            else:
                fig['layout']['yaxis']['type'] = 'linear'
                fig['layout']['yaxis']['tickfont'] = {'size': 9}
                if y_max_use and y_max_use > 0:
                    fig['layout']['yaxis']['range'] = [0, y_max_use * 1.05]
                    fig['layout']['yaxis']['autorange'] = False

            if y_max_use:
                updated_preview_y_range[y_key] = {
                    "y_max": y_max_use,
                    "y_min_pos": y_min_use,
                }

            fig["layout"]["showlegend"] = False
            fig['layout']['margin'] = dict(l=45, r=5, t=55, b=30)
            # fig['layout']['uirevision'] = f"xr_{peak_label}"
            figures.append(fig)

        titles.extend([None for _ in range(MAX_NUM_CARDS - len(figures))])
        figures.extend([EMPTY_PLOTLY_FIGURE for _ in range(MAX_NUM_CARDS - len(figures))])
        bookmarks.extend([0 for _ in range(MAX_NUM_CARDS - len(bookmarks))])

        if 'targets-select' in ctx.triggered[0]['prop_id']:
            targets_select_options = dash.no_update
        else:
            targets_select_options = [
                {"label": target, "value": target} for target in all_targets
            ]
        
        logger.debug(f"Preview refreshed in {time.perf_counter() - t1:.4f}s")
        return titles, figures, bookmarks, len(all_targets), current_page, "", targets_select_options

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
            has_traces = bool(figure) and bool(figure.get('data'))
            if has_traces:
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
        State('rt-alignment-data', 'data'),  # Get RT alignment calculation data
        State('chromatogram-view-rt-align', 'checked'),  # Get current toggle state

        State('slider-reference-data', 'data'),
        State('slider-data', 'data'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def handle_modal_open_close(target_clicked, close_clicks, close_without_save_clicks, update_chromatograms,
                                target_note, rt_alignment_data, rt_align_toggle, slider_ref, slider_data, wdir):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'target-preview-clicked':
            return True, dash.no_update, dash.no_update
            # if not has_changes, close it
        elif trigger_id == 'chromatogram-view-close':
            logger.info(f"Closing modal for {target_clicked}, note value received: '{target_note}'")
            logger.info(f"RT alignment toggle state: {rt_align_toggle}")
            
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    return dash.no_update, dash.no_update, dash.no_update
                
                # Always save the current RT alignment toggle state
                if rt_align_toggle:
                    # Toggle is ON - save alignment data
                    if rt_alignment_data and rt_alignment_data.get('enabled'):
                        # We have valid alignment data to save
                        import json
                        shifts_json = json.dumps(rt_alignment_data.get('shifts_by_sample_type', {}))
                       
                        conn.execute("""
                            UPDATE targets 
                            SET rt_align_enabled = TRUE,
                                rt_align_reference_rt = ?,
                                rt_align_shifts = ?,
                                rt_align_rt_min = ?,
                                rt_align_rt_max = ?
                            WHERE peak_label = ?
                        """, [
                            rt_alignment_data['reference_rt'],
                            shifts_json,
                            rt_alignment_data['rt_min'],
                            rt_alignment_data['rt_max'],
                            target_clicked
                        ])
                        logger.info(f"Saved RT alignment: enabled=TRUE, ref={rt_alignment_data['reference_rt']:.2f}s")
                    else:
                        logger.warning("RT align toggle is ON but no alignment data available - not saving")
                else:
                    # Toggle is OFF - clear alignment data
                    conn.execute("""
                        UPDATE targets 
                        SET rt_align_enabled = FALSE,
                            rt_align_reference_rt = NULL,
                            rt_align_shifts = NULL,
                            rt_align_rt_min = NULL,
                            rt_align_rt_max = NULL
                        WHERE peak_label = ?
                    """, [target_clicked])
                    logger.info("Saved RT alignment: enabled=FALSE (cleared all data)")
                
                # Save notes (this was accidentally removed)
                conn.execute("UPDATE targets SET notes = ? WHERE peak_label = ?",
                             (target_note, target_clicked))
                logger.info(f"Saved notes for {target_clicked}: '{target_note}'")

            # allow close if no slider data or no changes
            if (not slider_ref or not slider_data) or slider_ref.get('value') == slider_data.get('value'):
                return False, None, update_chromatograms or dash.no_update

            # if it has changes, don't close it (handled via confirm modal)
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
        if not close_clicks:
            raise PreventUpdate

        if not reference_data or not slider_data:
            return False

        reference_value = reference_data.get('value') if isinstance(reference_data, dict) else None
        slider_value = slider_data.get('value') if isinstance(slider_data, dict) else None
        if reference_value is None or slider_value is None:
            return False

        has_changes = slider_value != reference_value
        return bool(has_changes)

    ############# VIEW MODAL END #######################################

    ############# VIEW BEGIN #######################################
    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),

        Input('chromatogram-view-log-y', 'checked'),
        State('chromatogram-view-plot', 'figure'),
        State('chromatogram-view-plot-max', 'data'),
        State('chromatogram-view-plot-points', 'data'),
        prevent_initial_call=True
    )
    def chromatogram_view_y_scale(log_scale, figure, max_y, total_points):

        y_min, y_max = max_y
        fig = Patch()

        def _calc_y_range(data, x_left, x_right, is_log=False):
            ys = []
            for trace in data or []:
                xs = trace.get('x', [])
                ys_trace = trace.get('y', [])
                for xv, yv in zip(xs, ys_trace):
                    if xv is None or yv is None:
                        continue
                    if x_left <= xv <= x_right:
                        ys.append(yv)

            if not ys:
                return None

            if is_log:
                ys = [y for y in ys if y is not None and y > 1]
                if not ys:
                    return None
                y_min_local, y_max_local = min(ys), max(ys)
                return [math.log10(y_min_local), math.log10(y_max_local * 1.05)]

            y_min_local, y_max_local = min(ys), max(ys)
            y_min_local = 0 if y_min_local > 0 else y_min_local
            return [y_min_local, y_max_local * 1.05]

        # Try to use the current x-range (zoom or RT span) to compute an informed y-range.
        x_range = figure.get('layout', {}).get('xaxis', {}).get('range')
        if x_range and len(x_range) == 2:
            x_left, x_right = x_range
        else:
            shape = (figure.get('layout', {}).get('shapes') or [{}])[0]
            x_left, x_right = shape.get('x0'), shape.get('x1')
            if x_left is None or x_right is None:
                x_left = x_right = None

        if log_scale:
            fig['layout']['yaxis']['type'] = 'log'
            y_range_calc = None
            if x_left is not None and x_right is not None:
                y_range_calc = _calc_y_range(figure.get('data', []), min(x_left, x_right), max(x_left, x_right), True)
            if y_range_calc:
                fig['layout']['yaxis']['range'] = y_range_calc
            else:
                log_y_min = math.log10(y_min) if y_min > 0 else y_min
                log_y_max = math.log10(y_max) if y_max > 0 else y_max
                fig['layout']['yaxis']['range'] = [log_y_min, log_y_max]
        else:
            fig['layout']['yaxis']['type'] = 'linear'
            y_range_calc = None
            if x_left is not None and x_right is not None:
                y_range_calc = _calc_y_range(figure.get('data', []), min(x_left, x_right), max(x_left, x_right), False)
            if y_range_calc:
                fig['layout']['yaxis']['range'] = y_range_calc
            else:
                fig['layout']['yaxis']['range'] = [y_min, y_max]
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
        Input('chromatogram-view-megatrace', 'checked'),
        State('chromatogram-view-plot', 'figure'),
        State('target-preview-clicked', 'data'),
        State('wdir', 'data'),
        State('rt-alignment-data', 'data'),  # Check if RT alignment is active
        prevent_initial_call=True
    )
    def update_megatrace_mode(use_megatrace, figure, target_clicked, wdir, rt_alignment_data):
        if not wdir or not target_clicked:
            raise PreventUpdate
        
        with duckdb_connection(wdir) as conn:
            query = """
                    WITH picked_samples AS (SELECT ms_file_label, color, label, sample_type
                                            FROM samples
                                            WHERE use_for_optimization = TRUE
                    ),
                         picked_target AS (SELECT peak_label,
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
        
        # Apply RT alignment if active
        rt_alignment_shifts = None
        if rt_alignment_data and rt_alignment_data.get('enabled'):
            rt_alignment_shifts = calculate_rt_alignment(
                chrom_df, 
                rt_alignment_data['rt_min'], 
                rt_alignment_data['rt_max']
            )
            logger.info(f"Megatrace callback: Applying RT alignment with {len(rt_alignment_shifts)} shifts")
        
        traces, x_min, x_max, y_min, y_max = generate_chromatogram_traces(
            chrom_df, 
            use_megatrace=use_megatrace,
            rt_alignment_shifts=rt_alignment_shifts
        )
        
        fig = Patch()
        fig['data'] = traces
        # We don't necessarily update ranges here to preserve user zoom/pan if desired, 
        # but to be consistent with main load, we might want to. 
        # For now let's update data only, or update everything if user expects a "reset" view.
        # Given this is a toggle, replacing data is key.
        if use_megatrace:
             fig['layout']['hovermode'] = False
        else:
             fig['layout']['hovermode'] = 'closest'
        return fig

    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Output('rt-alignment-data', 'data'),  # Store alignment info for saving to notes
        Input('chromatogram-view-rt-align', 'checked'),
        State('chromatogram-view-plot', 'figure'),
        State('chromatogram-view-megatrace', 'checked'),
        State('slider-data', 'data'),  # Use current slider values, not reference
        State('target-preview-clicked', 'data'),
        State('wdir', 'data'),
        State('rt-alignment-data', 'data'),  # Check if this is a restoration
        prevent_initial_call=True
    )
    def apply_rt_alignment(use_alignment, figure, use_megatrace, slider_current, target_clicked, wdir, existing_rt_data):
        """Apply or remove RT alignment when toggle changes"""
        logger.info(f"RT Alignment callback triggered: use_alignment={use_alignment}")
        
        # If turning ON and we already have matching alignment data in the store,
        # this is likely a state restoration - skip to avoid overwriting pre-aligned figure
        if use_alignment and existing_rt_data and existing_rt_data.get('enabled'):
            logger.info("Skipping RT alignment callback - state restoration detected (data already in store)")
            raise PreventUpdate
        
        if not wdir or not target_clicked or not slider_current:
            logger.warning("RT Alignment: Missing required data, raising PreventUpdate")
            raise PreventUpdate
        
        # rt_min and rt_max are nested inside the 'value' key
        value_dict = slider_current.get('value', {})
        rt_min = value_dict.get('rt_min')
        rt_max = value_dict.get('rt_max')
        
        logger.info(f"RT range: rt_min={rt_min}, rt_max={rt_max}")
        
        if rt_min is None or rt_max is None:
            logger.warning("RT Alignment: rt_min or rt_max is None, raising PreventUpdate")
            raise PreventUpdate
        
        with duckdb_connection(wdir) as conn:
            query = """
                    WITH picked_samples AS (SELECT ms_file_label, color, label, sample_type
                                            FROM samples
                                            WHERE use_for_optimization = TRUE
                    ),
                         picked_target AS (SELECT peak_label,
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
        
        # Calculate RT alignment shifts if alignment is enabled
        rt_alignment_shifts = None
        alignment_data = None
        
        if use_alignment:
            rt_alignment_shifts = calculate_rt_alignment(chrom_df, rt_min, rt_max)
            
            # Calculate shifts per sample type for notes
            shifts_per_sample_type = calculate_shifts_per_sample_type(chrom_df, rt_alignment_shifts)
            
            # Find reference RT (median of apex RTs)
            apex_rts = []
            for row in chrom_df.iter_rows(named=True):
                scan_time = np.array(row['scan_time_sliced'])
                intensity = np.array(row['intensity_sliced'])
                mask = (scan_time >= rt_min) & (scan_time <= rt_max)
                if mask.any():
                    rt_in_range = scan_time[mask]
                    int_in_range = intensity[mask]
                    apex_idx = int_in_range.argmax()
                    apex_rts.append(rt_in_range[apex_idx])
            
            reference_rt = float(np.median(apex_rts)) if apex_rts else None
            
            # Store alignment data for saving to notes
            alignment_data = {
                'enabled': True,
                'reference_rt': reference_rt,
                'shifts_by_sample_type': shifts_per_sample_type,
                'rt_min': rt_min,
                'rt_max': rt_max
            }
            logger.info(f"RT Alignment data prepared: {alignment_data}")
        
        # Regenerate traces with or without alignment
        traces, x_min, x_max, y_min, y_max = generate_chromatogram_traces(
            chrom_df, 
            use_megatrace=use_megatrace,
            rt_alignment_shifts=rt_alignment_shifts
        )
        
        fig = Patch()
        fig['data'] = traces
        
        # Update x-axis range if alignment is applied
        if use_alignment and rt_alignment_shifts:
            # Recalculate x_min and x_max based on aligned data
            all_x_values = []
            for trace in traces:
                if trace.get('x'):
                    all_x_values.extend([x for x in trace['x'] if x is not None])
            if all_x_values:
                fig['layout']['xaxis']['range'] = [min(all_x_values), max(all_x_values)]
        
        return fig, alignment_data


    @app.callback(
        Output('chromatogram-view-lock-range', 'checked', allow_duplicate=True),
        Input('chromatogram-view-rt-align', 'checked'),
        prevent_initial_call=True
    )
    def lock_rt_span_when_aligning(rt_align_on):
        """Force RT span to Lock mode when RT alignment is ON"""
        logger.info(f"Lock RT span callback: rt_align_on={rt_align_on}, setting Lock mode (checked={rt_align_on})")
        return rt_align_on  # True = Lock mode, False = Edit mode


    @app.callback(
        Output('chromatogram-view-rt-align', 'checked', allow_duplicate=True),
        Input('chromatogram-view-lock-range', 'checked'),
        State('chromatogram-view-rt-align', 'checked'),
        prevent_initial_call=True
    )
    def turn_off_alignment_when_editing(is_locked, rt_align_on):
        """Turn OFF RT alignment when user switches from Lock to Edit mode"""
        # When switching to Edit mode (is_locked=False), turn off alignment
        if not is_locked and rt_align_on:
            logger.info("RT span switched to Edit mode - turning OFF RT alignment")
            return False
        raise PreventUpdate


    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Output('chromatogram-view-modal', 'title'),
        Output('chromatogram-view-modal', 'loading'),
        Output('slider-reference-data', 'data'),
        Output('slider-data', 'data', allow_duplicate=True),  # make sure this is reset
        Output('chromatogram-view-plot-max', 'data'),
        Output('chromatogram-view-plot-points', 'data'),
        Output('chromatogram-view-megatrace', 'checked', allow_duplicate=True),
        Output('chromatogram-view-log-y', 'checked', allow_duplicate=True),
        Output('chromatogram-view-groupclick', 'checked', allow_duplicate=True),
        Output('chromatogram-view-rt-align', 'checked', allow_duplicate=True),  # Set RT alignment state
        Output('rt-alignment-data', 'data', allow_duplicate=True),  # Load alignment data
        Output('target-note', 'value', allow_duplicate=True),

        Input('target-preview-clicked', 'data'),
        State('chromatogram-preview-log-y', 'checked'),
        State('sample-type-tree', 'checkedKeys'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def chromatogram_view_modal(target_clicked, log_scale, checkedKeys, wdir):

        if not wdir:
            raise PreventUpdate
        with duckdb_connection(wdir) as conn:
            # Load target data including RT alignment columns
            d = conn.execute("""
                SELECT rt, rt_min, rt_max, COALESCE(notes, ''),
                       rt_align_enabled, rt_align_reference_rt, rt_align_shifts,
                       rt_align_rt_min, rt_align_rt_max
                FROM targets 
                WHERE peak_label = ?
            """, [target_clicked]).fetchall()
            
            if d:
                rt, rt_min, rt_max, note, align_enabled, align_ref_rt, align_shifts_json, align_rt_min, align_rt_max = d[0]
            else:
                rt, rt_min, rt_max, note = None, None, None, ''
                align_enabled = False
                align_ref_rt = None
                align_shifts_json = None
                align_rt_min = None
                align_rt_max = None

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
            
            try:
                n_sample_types = chrom_df['sample_type'].n_unique()
                group_legend = True if n_sample_types > 1 else False
            except Exception as e:
                logger.warning(f"Error determining sample types: {e}")
                group_legend = False

        t1 = time.perf_counter()
        fig = Patch()
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

        legend_groups = set()
        traces = []
        total_points = 0
        # TODO: check if chrom_df is empty and Implement an empty widget to show when no data

        MAX_TRACES = 200

        if len(chrom_df) <= MAX_TRACES:
            use_megatrace = False
        else:
            use_megatrace = True

        # Calculate RT alignment shifts if enabled in database
        rt_alignment_shifts_to_apply = None
        if align_enabled and align_ref_rt is not None:
            # Calculate alignment shifts from stored data
            rt_alignment_shifts_to_apply = calculate_rt_alignment(chrom_df, align_rt_min, align_rt_max)
            logger.info(f"Applying saved RT alignment on modal open: ref={align_ref_rt:.2f}s")
            logger.info(f"Calculated shifts for initial figure: {rt_alignment_shifts_to_apply}")
        
        traces, x_min, x_max, y_min, y_max = generate_chromatogram_traces(
            chrom_df, 
            use_megatrace=use_megatrace,
            rt_alignment_shifts=rt_alignment_shifts_to_apply
        )
        logger.info(f"Initial figure generated with {len(traces) if traces else 0} traces, alignment_applied={rt_alignment_shifts_to_apply is not None}")

        if traces:
            total_points = sum(len(t['x']) for t in traces)

        # ------------------------------------
        # Assemble Figure
        # ------------------------------------
        fig['layout']['xaxis']['range'] = [x_min, x_max]
        fig['layout']['yaxis']['range'] = [y_min, y_max * 1.05]
        fig['layout']['xaxis']['autorange'] = False
        fig['layout']['yaxis']['autorange'] = False
        fig['layout']['_initial_alignment_applied'] = (rt_alignment_shifts_to_apply is not None)  # Marker for debugging
        fig['data'] = traces
        # fig['layout']['title'] = {'text': f"{target_clicked} (rt={rt})"}
        fig['layout']['shapes'] = []
        if use_megatrace:
            fig['layout']['hovermode'] = False
        else:
            fig['layout']['hovermode'] = 'closest'

        fig['layout']['annotations'] = [
            {
                'bgcolor': 'white',
                'font': {'color': 'black', 'size': 12, 'weight': 'bold'},
                'showarrow': False,
                'ax': -20,
                'ay': -15,
                'axref': 'pixel',
                'ayref': 'pixel',
                'text': f"RT-min: {rt_min:.2f}s" if rt_min is not None else 'RT-min',
                'x': rt_min,
                'xanchor': 'right',
                'xref': 'x',
                'y': 1,
                'yanchor': 'top',
                # 'yref': 'y domain',
                'yref': 'paper',
                'yshift': 15

            },
            {
                'bgcolor': 'white',
                'font': {'color': 'black', 'size': 12, 'weight': 'bold'},
                'showarrow': False,
                'ax': 20,
                'ay': -15,
                'axref': 'pixel',
                'ayref': 'pixel',
                'text': f"RT-max: {rt_max:.2f}s" if rt_max is not None else 'RT-max',
                'x': rt_max,
                'xanchor': 'left',
                'xref': 'x',
                'y': 1,
                'yanchor': 'top',
                # 'yref': 'y domain',
                'yref': 'paper',
                'yshift': 15
            },
        ]

        fig['layout']['shapes'] = [
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
            },
            # RT vertical line (dashdot) - same style as cards, not editable
            {
                'line': {'color': 'black', 'width': 1.5, 'dash': 'dashdot'},
                'type': 'line',
                'x0': rt,
                'x1': rt,
                'xref': 'x',
                'y0': 0,
                'y1': 1,
                'yref': 'y domain',
                'editable': False  # Prevent dragging - use click to set position
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

        def _calc_y_range_from_traces(data, x_left, x_right, is_log=False):
            ys = []
            for trace in data or []:
                xs = trace.get('x', [])
                ys_trace = trace.get('y', [])
                for xv, yv in zip(xs, ys_trace):
                    if xv is None or yv is None:
                        continue
                    if x_left <= xv <= x_right:
                        ys.append(yv)

            if not ys:
                return None

            if is_log:
                # Drop non-positive values; use the smallest intensity strictly greater than 1 as the floor.
                ys = [y for y in ys if y is not None and y > 1]
                if not ys:
                    return None
                y_min_local, y_max_local = min(ys), max(ys)
                return [math.log10(y_min_local), math.log10(y_max_local * 1.05)]

            y_min_local, y_max_local = min(ys), max(ys)
            y_min_local = 0 if y_min_local > 0 else y_min_local
            return [y_min_local, y_max_local * 1.05]

        fig['layout']['yaxis']['type'] = 'log' if log_scale else 'linear'
        y_range_zoom = _calc_y_range_from_traces(traces, nx_min, nx_max, is_log=log_scale)
        if y_range_zoom:
            fig['layout']['yaxis']['range'] = y_range_zoom
            fig['layout']['yaxis']['autorange'] = False
        else:
            if log_scale:
                log_y_min = math.log10(y_min) if y_min > 0 else y_min
                log_y_max = math.log10(y_max) if y_max > 0 else y_max
                fig['layout']['yaxis']['range'] = [log_y_min, log_y_max]
            else:
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
        slider_reference = s_data
        slider_dict = slider_reference.copy()
        
        # Parse RT alignment data from database
        # Simple approach: restore the exact state that was saved
        rt_align_toggle_state = False  # Default if no alignment saved
        rt_alignment_data_to_load = None
        
        if align_enabled and align_ref_rt is not None:
            try:
                import json
                shifts_dict = json.loads(align_shifts_json) if align_shifts_json else {}
                rt_alignment_data_to_load = {
                    'enabled': True,
                    'reference_rt': align_ref_rt,
                    'shifts_by_sample_type': shifts_dict,
                    'rt_min': align_rt_min,
                    'rt_max': align_rt_max
                }
                rt_align_toggle_state = True  # Set toggle to ON to match saved state
                logger.info(f"Restoring RT alignment state: toggle=ON, ref={align_ref_rt:.2f}s")
            except Exception as e:
                logger.error(f"Error parsing RT alignment data: {e}")

        logger.debug(f"Modal view prepared in {time.perf_counter() - t1:.4f}s")
        return (fig, f"{target_clicked} (rt={rt})", False, slider_reference,
                slider_dict, {"min_y": y_min, "max_y": y_max}, total_points, use_megatrace, log_scale, group_legend, 
                rt_align_toggle_state, rt_alignment_data_to_load, note)

    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Output('slider-data', 'data', allow_duplicate=True),
        Output('action-buttons-container', 'style', allow_duplicate=True),

        Input('chromatogram-view-plot', 'relayoutData'),
        Input('slider-reference-data', 'data'),
        State('slider-data', 'data'),
        State('chromatogram-view-plot', 'figure'),
        State('chromatogram-view-plot-points', 'data'),
        State('chromatogram-view-lock-range', 'checked'),
        prevent_initial_call=True
    )
    def update_rt_range_from_shape(relayout, slider_reference_data, slider_data, figure_state, total_points, lock_range):
        if not slider_reference_data:
            raise PreventUpdate

        def _calc_y_range(data, x_left, x_right, is_log=False):
            ys = []
            for trace in data or []:
                xs = trace.get('x', [])
                ys_trace = trace.get('y', [])
                for xv, yv in zip(xs, ys_trace):
                    if xv is None or yv is None:
                        continue
                    if x_left <= xv <= x_right:
                        ys.append(yv)

            if not ys:
                return None

            if is_log:
                # Drop non-positive values; use the smallest intensity strictly greater than 1 as the floor.
                ys = [y for y in ys if y is not None and y > 1]
                if not ys:
                    return None
                y_min_local, y_max_local = min(ys), max(ys)
                return [math.log10(y_min_local), math.log10(y_max_local * 1.05)]

            y_min_local, y_max_local = min(ys), max(ys)
            y_min_local = 0 if y_min_local > 0 else y_min_local
            return [y_min_local, y_max_local * 1.05]

        def _maybe_pad_x_range(current_range, rt_min_val, rt_max_val, pad_seconds):
            if rt_min_val is None or rt_max_val is None:
                return None

            desired_min = rt_min_val - pad_seconds
            desired_max = rt_max_val + pad_seconds

            if not current_range or len(current_range) != 2:
                return [desired_min, desired_max]

            cur_min, cur_max = current_range
            if cur_min is None or cur_max is None:
                return [desired_min, desired_max]
            if cur_min > cur_max:
                cur_min, cur_max = cur_max, cur_min

            new_min = cur_min
            new_max = cur_max

            # Only expand if the current padding is smaller than the minimum requested.
            if rt_min_val < cur_min or (rt_min_val - cur_min) < pad_seconds:
                new_min = desired_min
            if rt_max_val > cur_max or (cur_max - rt_max_val) < pad_seconds:
                new_max = desired_max

            return [new_min, new_max]

        ctx = dash.callback_context
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if prop_id == 'slider-reference-data':
            if not slider_data:
                slider_data = slider_reference_data.copy()
            else:
                slider_data['value'] = slider_reference_data['value']
            rt_min_ref = slider_data['value']['rt_min']
            rt_max_ref = slider_data['value']['rt_max']
            is_log = figure_state and figure_state.get('layout', {}).get('yaxis', {}).get('type') == 'log'
            pad = 2.5  # seconds of padding on each side of the RT span
            current_x_range = figure_state.get('layout', {}).get('xaxis', {}).get('range') if figure_state else None

            fig = Patch()
            fig['layout']['shapes'][0]['x0'] = rt_min_ref
            fig['layout']['shapes'][0]['x1'] = rt_max_ref
            fig['layout']['shapes'][0]['y0'] = 0
            fig['layout']['shapes'][0]['y1'] = 1
            fig['layout']['shapes'][0]['yref'] = 'y domain'
            fig['layout']['shapes'][0]['fillcolor'] = 'red' if lock_range else 'green'
            fig['layout']['shapes'][0]['opacity'] = 0.1
            x_range = _maybe_pad_x_range(current_x_range, rt_min_ref, rt_max_ref, pad)
            if x_range:
                fig['layout']['xaxis']['range'] = x_range
                fig['layout']['xaxis']['autorange'] = False

            if figure_state:
                y_calc = _calc_y_range(figure_state.get('data', []), rt_min_ref, rt_max_ref, is_log)
                if y_calc:
                    fig['layout']['yaxis']['range'] = y_calc
                    fig['layout']['yaxis']['autorange'] = False

            fig['layout']['annotations'][0]['x'] = rt_min_ref
            fig['layout']['annotations'][0]['text'] = f"RT-min: {rt_min_ref:.2f}s"
            fig['layout']['annotations'][1]['x'] = rt_max_ref
            fig['layout']['annotations'][1]['text'] = f"RT-max: {rt_max_ref:.2f}s"

            buttons_style = {
                'visibility': 'hidden',
                'opacity': '0',
                'transition': 'opacity 0.3s ease-in-out'
            }
            return fig, slider_data, buttons_style

        if not relayout:
            raise PreventUpdate

        x_range = (relayout.get('xaxis.range[0]'), relayout.get('xaxis.range[1]'))
        y_range = (relayout.get('yaxis.range[0]'), relayout.get('yaxis.range[1]'))
        has_shape_update = relayout.get('shapes[0].x0') is not None or relayout.get('shapes[0].x1') is not None

        # Allow plotly zooming (x and y) to drive axis ranges even when the RT span is locked.
        if (x_range[0] is not None and x_range[1] is not None) or (y_range[0] is not None and y_range[1] is not None):
            fig_zoom = Patch()
            is_log = figure_state and figure_state.get('layout', {}).get('yaxis', {}).get('type') == 'log'

            if x_range[0] is not None and x_range[1] is not None:
                fig_zoom['layout']['xaxis']['range'] = [x_range[0], x_range[1]]
                fig_zoom['layout']['xaxis']['autorange'] = False

            if y_range[0] is not None and y_range[1] is not None:
                fig_zoom['layout']['yaxis']['range'] = [y_range[0], y_range[1]]
                fig_zoom['layout']['yaxis']['autorange'] = False
            elif x_range[0] is not None and x_range[1] is not None and figure_state:
                y_calc = _calc_y_range(figure_state.get('data', []), x_range[0], x_range[1], is_log)
                if y_calc:
                    fig_zoom['layout']['yaxis']['range'] = y_calc
                    fig_zoom['layout']['yaxis']['autorange'] = False

            return fig_zoom, dash.no_update, dash.no_update

        if lock_range and has_shape_update:
            raise PreventUpdate

        # Handle RT line drag (shapes[1]) - constrain to within span
        rt_line_x0 = relayout.get('shapes[1].x0')
        if rt_line_x0 is not None:
            rt_min_current = slider_data['value'].get('rt_min') if slider_data else None
            rt_max_current = slider_data['value'].get('rt_max') if slider_data else None
            
            if rt_min_current is not None and rt_max_current is not None:
                # Check if dragged RT is within span
                if rt_min_current <= rt_line_x0 <= rt_max_current:
                    # Valid position - update RT
                    slider_data['value']['rt'] = rt_line_x0
                    has_changes = slider_data['value'] != slider_reference_data['value']
                    buttons_style = {
                        'visibility': 'visible' if has_changes else 'hidden',
                        'opacity': '1' if has_changes else '0',
                        'transition': 'opacity 0.3s ease-in-out'
                    }
                    fig = Patch()
                    fig['layout']['shapes'][1]['x0'] = rt_line_x0
                    fig['layout']['shapes'][1]['x1'] = rt_line_x0
                    # Reset y coordinates to ensure full height
                    fig['layout']['shapes'][1]['y0'] = 0
                    fig['layout']['shapes'][1]['y1'] = 1
                    fig['layout']['shapes'][1]['yref'] = 'y domain'
                    return fig, slider_data, buttons_style
                else:
                    # Outside span - snap back to max intensity
                    rt_at_max = (rt_min_current + rt_max_current) / 2  # fallback to midpoint
                    max_intensity = -1
                    for trace in (figure_state.get('data', []) if figure_state else []):
                        xs = trace.get('x', [])
                        ys = trace.get('y', [])
                        for xv, yv in zip(xs, ys):
                            if xv is None or yv is None:
                                continue
                            if rt_min_current <= xv <= rt_max_current and yv > max_intensity:
                                max_intensity = yv
                                rt_at_max = xv
                    
                    slider_data['value']['rt'] = rt_at_max
                    has_changes = slider_data['value'] != slider_reference_data['value']
                    buttons_style = {
                        'visibility': 'visible' if has_changes else 'hidden',
                        'opacity': '1' if has_changes else '0',
                        'transition': 'opacity 0.3s ease-in-out'
                    }
                    fig = Patch()
                    fig['layout']['shapes'][1]['x0'] = rt_at_max
                    fig['layout']['shapes'][1]['x1'] = rt_at_max
                    # Reset y coordinates to ensure full height
                    fig['layout']['shapes'][1]['y0'] = 0
                    fig['layout']['shapes'][1]['y1'] = 1
                    fig['layout']['shapes'][1]['yref'] = 'y domain'
                    return fig, slider_data, buttons_style

        x0 = relayout.get('shapes[0].x0')
        x1 = relayout.get('shapes[0].x1')
        if x0 is None or x1 is None:
            raise PreventUpdate

        if not slider_data:
            slider_data = slider_reference_data.copy()

        rt_min_new = min(x0, x1)
        rt_max_new = max(x0, x1)

        # Get current RT value
        current_rt = slider_data['value'].get('rt')
        
        # Only recalculate RT to max intensity if current RT is outside the new span
        if current_rt is not None and rt_min_new <= current_rt <= rt_max_new:
            # Current RT is still within span, keep it
            rt_new = current_rt
        else:
            # Current RT is outside span, find max intensity position
            rt_new = (rt_min_new + rt_max_new) / 2  # fallback to midpoint
            max_intensity = -1
            for trace in (figure_state.get('data', []) if figure_state else []):
                xs = trace.get('x', [])
                ys = trace.get('y', [])
                for xv, yv in zip(xs, ys):
                    if xv is None or yv is None:
                        continue
                    if rt_min_new <= xv <= rt_max_new and yv > max_intensity:
                        max_intensity = yv
                        rt_new = xv

        slider_data['value'] = {
            'rt_min': rt_min_new,
            'rt': rt_new,
            'rt_max': rt_max_new,
        }

        has_changes = slider_data['value'] != slider_reference_data['value']
        buttons_style = {
            'visibility': 'visible' if has_changes else 'hidden',
            'opacity': '1' if has_changes else '0',
            'transition': 'opacity 0.3s ease-in-out'
        }

        fig = Patch()
        fig['layout']['shapes'][0]['x0'] = rt_min_new
        fig['layout']['shapes'][0]['x1'] = rt_max_new
        fig['layout']['shapes'][0]['y0'] = 0
        fig['layout']['shapes'][0]['y1'] = 1
        fig['layout']['shapes'][0]['yref'] = 'y domain'
        fig['layout']['shapes'][0]['fillcolor'] = 'green'
        fig['layout']['shapes'][0]['opacity'] = 0.1
        
        # Update RT line position (shapes[1]) to new RT value
        fig['layout']['shapes'][1]['x0'] = rt_new
        fig['layout']['shapes'][1]['x1'] = rt_new

        # adjust axes to the current RT span box for better scaling
        is_log = figure_state and figure_state.get('layout', {}).get('yaxis', {}).get('type') == 'log'
        if figure_state:
            y_range_zoom = _calc_y_range(figure_state.get('data', []), rt_min_new, rt_max_new, is_log)
            if y_range_zoom:
                fig['layout']['yaxis']['range'] = y_range_zoom
                fig['layout']['yaxis']['autorange'] = False
            pad = 2.5  # seconds of padding on each side
            current_x_range = figure_state.get('layout', {}).get('xaxis', {}).get('range')
            x_range = _maybe_pad_x_range(current_x_range, rt_min_new, rt_max_new, pad)
            if x_range:
                fig['layout']['xaxis']['range'] = x_range
                fig['layout']['xaxis']['autorange'] = False

        fig['layout']['annotations'][0]['x'] = rt_min_new
        fig['layout']['annotations'][0]['text'] = f"RT-min: {rt_min_new:.2f}s"
        fig['layout']['annotations'][1]['x'] = rt_max_new
        fig['layout']['annotations'][1]['text'] = f"RT-max: {rt_max_new:.2f}s"

        return fig, slider_data, buttons_style

    @app.callback(
        Output('chromatogram-view-plot', 'config', allow_duplicate=True),
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Input('chromatogram-view-lock-range', 'checked'),
        prevent_initial_call=True
    )
    def chromatogram_view_lock_range(lock_range):
        config_patch = Patch()
        config_patch['edits']['shapePosition'] = not lock_range

        fig = Patch()
        fig['layout']['shapes'][0]['fillcolor'] = 'red' if lock_range else 'green'
        fig['layout']['shapes'][0]['opacity'] = 0.1
        fig['layout']['shapes'][0]['y0'] = 0
        fig['layout']['shapes'][0]['y1'] = 1
        fig['layout']['shapes'][0]['yref'] = 'y domain'

        return config_patch, fig

    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Output('slider-data', 'data', allow_duplicate=True),
        Output('action-buttons-container', 'style', allow_duplicate=True),
        Input('chromatogram-view-plot', 'clickData'),
        State('slider-data', 'data'),
        State('slider-reference-data', 'data'),
        prevent_initial_call=True
    )
    def set_rt_on_click(click_data, slider_data, slider_reference):
        """Set RT position when user clicks on the chromatogram."""
        if not click_data or not slider_data or not slider_reference:
            raise PreventUpdate
        
        # Get clicked x position (retention time)
        point = click_data.get('points', [{}])[0]
        clicked_rt = point.get('x')
        if clicked_rt is None:
            raise PreventUpdate
        
        # Ensure clicked RT is within the RT span
        rt_min = slider_data['value'].get('rt_min')
        rt_max = slider_data['value'].get('rt_max')
        if rt_min is not None and rt_max is not None:
            if clicked_rt < rt_min or clicked_rt > rt_max:
                raise PreventUpdate  # Don't allow setting RT outside the span
        
        # Update slider_data with new RT
        slider_data['value']['rt'] = clicked_rt
        
        # Check if there are changes compared to reference
        has_changes = slider_data['value'] != slider_reference['value']
        buttons_style = {
            'visibility': 'visible' if has_changes else 'hidden',
            'opacity': '1' if has_changes else '0',
            'transition': 'opacity 0.3s ease-in-out'
        }
        
        # Update RT line position in the figure
        fig = Patch()
        fig['layout']['shapes'][1]['x0'] = clicked_rt
        fig['layout']['shapes'][1]['x1'] = clicked_rt
        
        return fig, slider_data, buttons_style

    ############# VIEW END #######################################

    ############# COMPUTE CHROMATOGRAM BEGIN #####################################
    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("compute-chromatogram-modal", "visible"),
        Output("chromatogram-warning", "style"),
        Output("chromatogram-warning", "message"),
        Output("chromatogram-targets-info", "message"),
        Output("chromatogram-compute-ram", "max"),
        Output("chromatogram-compute-ram-item", "help"),
        Output("chromatogram-processing-progress", "percent", allow_duplicate=True),
        Output("chromatogram-processing-stage", "children", allow_duplicate=True),
        Output("chromatogram-processing-detail", "children", allow_duplicate=True),
        Output("chromatograms-recompute-ms1", "checked"),
        Output("chromatograms-recompute-ms2", "checked"),

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
                return (
                    fac.AntdNotification(
                        message="Compute Chromatograms",
                        description="Workspace is not available. Please select or create a workspace.",
                        type="error",
                        duration=4,
                        placement="bottom",
                        showProgress=True,
                    ),
                    False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 0
                )

            ms_files = conn.execute("SELECT COUNT(*) FROM samples WHERE use_for_optimization = TRUE").fetchone()
            targets = conn.execute("SELECT COUNT(*) FROM targets").fetchone()

            if not ms_files or ms_files[0] == 0 or not targets or targets[0] == 0:
                return (
                    fac.AntdNotification(
                        message="Compute Chromatograms",
                        description="Need at least one MS-file (marked for optimization) and one target before computing chromatograms.",
                        type="warning",
                        duration=4,
                        placement="bottom",
                        showProgress=True,
                    ),
                    False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 0
                )

            chromatograms = conn.execute("SELECT COUNT(*) FROM chromatograms").fetchone()
            if chromatograms:
                computed_chromatograms = chromatograms[0]

            targets = conn.execute("SELECT COUNT(*) FROM targets WHERE peak_selection = TRUE").fetchone()
            if targets:
                selected_targets = targets[0]

            chromatograms_ms1 = conn.execute("SELECT COUNT(*) FROM chromatograms WHERE ms_type='ms1'").fetchone()
            if chromatograms_ms1:
                computed_chromatograms_ms1 = chromatograms_ms1[0]
            else:
                computed_chromatograms_ms1 = 0

            chromatograms_ms2 = conn.execute("SELECT COUNT(*) FROM chromatograms WHERE ms_type='ms2'").fetchone()
            if chromatograms_ms2:
                computed_chromatograms_ms2 = chromatograms_ms2[0]
            else:
                computed_chromatograms_ms2 = 0

        warning_style = {'display': 'flex'} if computed_chromatograms else {'display': 'none'}
        warning_message = f"There are already computed {computed_chromatograms} chromatograms" if computed_chromatograms else ""

        ram_max = round(psutil.virtual_memory().available / (1024 ** 3), 1)
        help = f"Selected {ram_value}GB / {ram_max}GB available RAM"
        target_message = (f'Selected {selected_targets} targets'
                          if selected_targets
                          else 'There are no targets selected. The chromatograms will be computed for all targets.')
        return (
            dash.no_update,
            True,
            warning_style,
            warning_message,
            target_message,
            ram_max,
            help,
            0,
            "",
            "",
            bool(computed_chromatograms_ms1),
            bool(computed_chromatograms_ms2),
        )

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
            Output("chromatogram-processing-progress", "percent", allow_duplicate=True),
            Output("chromatogram-processing-stage", "children", allow_duplicate=True),
            Output("chromatogram-processing-detail", "children", allow_duplicate=True),
        ],
        cancel=[
            Input('cancel-chromatogram-processing', 'nClicks')
        ],
        prevent_initial_call=True
    )
    def compute_chromatograms(set_progress, okCounts, recompute_ms1, recompute_ms2, n_cpus, ram, batch_size, wdir):

        if not okCounts:
            raise PreventUpdate

        def progress_adapter(percent, stage="", detail=""):
            if set_progress:
                set_progress((percent, stage or "", detail or ""))

        activate_workspace_logging(wdir)

        with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as con:
            if con is None:
                logger.error("Could not connect to database for chromatogram computation.")
                return "Could not connect to database."
            start = time.perf_counter()
            logger.info("Starting chromatogram computation.")
            progress_adapter(0, "Chromatograms", "Preparing batches...")
            compute_chromatograms_in_batches(wdir, use_for_optimization=True, batch_size=batch_size,
                                             set_progress=progress_adapter, recompute_ms1=recompute_ms1,
                                             recompute_ms2=recompute_ms2, n_cpus=n_cpus, ram=ram)
            logger.info(f"Chromatograms computed in {time.perf_counter() - start:.2f} seconds")
            
            # Update RT values to max intensity time only for targets that had RT auto-adjusted
            progress_adapter(95, "Chromatograms", "Updating RT to peak apex...")
            try:
                # Chromatograms stores scan_time and intensity as arrays, so we need to unnest them
                # Only update targets where rt_auto_adjusted = TRUE
                update_sql = """
                    UPDATE targets
                    SET rt = subq.rt_at_max,
                        rt_auto_adjusted = FALSE
                    FROM (
                        WITH adjusted_targets AS (
                            SELECT peak_label, rt_min, rt_max
                            FROM targets
                            WHERE rt_auto_adjusted = TRUE
                        ),
                        unnested AS (
                            SELECT c.peak_label, 
                                   UNNEST(c.scan_time) AS scan_time,
                                   UNNEST(c.intensity) AS intensity
                            FROM chromatograms c
                            WHERE c.peak_label IN (SELECT peak_label FROM adjusted_targets)
                        ),
                        filtered AS (
                            SELECT u.peak_label, u.scan_time, u.intensity
                            FROM unnested u
                            JOIN adjusted_targets t ON u.peak_label = t.peak_label
                            WHERE u.scan_time BETWEEN t.rt_min AND t.rt_max
                        ),
                        max_per_target AS (
                            SELECT peak_label, MAX(intensity) AS max_intensity
                            FROM filtered
                            GROUP BY peak_label
                        )
                        SELECT f.peak_label, f.scan_time AS rt_at_max
                        FROM filtered f
                        JOIN max_per_target m ON f.peak_label = m.peak_label AND f.intensity = m.max_intensity
                    ) AS subq
                    WHERE targets.peak_label = subq.peak_label
                """
                con.execute(update_sql)
                logger.info("Updated RT to max intensity for auto-adjusted targets")
            except Exception as e:
                logger.warning(f"Could not update RT to max intensity: {e}")
            
        return True, False

    ############# COMPUTE CHROMATOGRAM END #######################################

    @app.callback(
        Output('chromatograms', 'data', allow_duplicate=True),
        Output('chromatogram-processing-progress-container', 'style', allow_duplicate=True),
        Output('chromatogram-compute-options-container', 'style', allow_duplicate=True),
        Output('compute-chromatogram-modal', 'visible', allow_duplicate=True),
        Output('chromatogram-processing-progress', 'percent', allow_duplicate=True),
        Output('chromatogram-processing-stage', 'children', allow_duplicate=True),
        Output('chromatogram-processing-detail', 'children', allow_duplicate=True),
        Input('cancel-chromatogram-processing', 'nClicks'),
        prevent_initial_call=True
    )
    def cancel_compute_chromatograms(cancel_clicks):
        if not cancel_clicks:
            raise PreventUpdate
        return (
            {'action': 'processing', 'status': 'cancelled', 'timestamp': time.time()},
            {'display': 'none'},
            {'display': 'flex'},
            False,
            0,
            "",
            "",
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
        
        logger.info(f"Updated RT for target '{target_clicked}': rt_min={rt_min}, rt={rt_}, rt_max={rt_max}")

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
        Input('delete-target-from-modal', 'nClicks'),
        State({'type': 'target-card-preview', 'index': ALL}, 'data-target'),
        State('target-preview-clicked', 'data'),
        prevent_initial_call=True
    )
    def show_delete_modal(delete_clicks, delete_modal_click, data_target, target_clicked):

        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        # Delete from card icon
        if trigger.startswith('{'):
            if not delete_clicks:
                raise PreventUpdate
            ctx_trigger = json.loads(trigger)
            if len(dash.callback_context.triggered) > 1:
                raise PreventUpdate
            prop_id = ctx_trigger['index']
            target = data_target[prop_id]
        # Delete from modal button
        elif trigger == 'delete-target-from-modal':
            if not delete_modal_click or not target_clicked:
                raise PreventUpdate
            target = target_clicked
        else:
            raise PreventUpdate

        return True, fac.AntdParagraph(f"Are you sure you want to delete `{target}` target?"), target

    #
    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('drop-chromatogram', 'data'),
        Output('delete-targets-modal', 'visible', allow_duplicate=True),
        Output('chromatogram-view-modal', 'visible', allow_duplicate=True),

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
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            try:
                conn.execute("BEGIN")
                conn.execute("DELETE FROM chromatograms WHERE peak_label = ?", [target])
                conn.execute("DELETE FROM targets WHERE peak_label = ?", [target])
                conn.execute("DELETE FROM results WHERE peak_label = ?", [target])
                conn.execute("COMMIT")
                logger.info(f"Deleted target '{target}' and associated chromatograms/results.")
            except Exception:
                conn.execute("ROLLBACK")
                logger.error(f"Failed to delete target '{target}'", exc_info=True)
                return (fac.AntdNotification(
                            message="Delete target failed",
                            description="Could not delete target data; no changes were applied.",
                            type="error",
                            duration=4,
                            placement='bottom',
                            showProgress=True,
                            stack=True
                        ),
                        dash.no_update,
                        False,
                        False)

        return (fac.AntdNotification(message=f"{target} chromatograms deleted",
                                     type="success",
                                     duration=3,
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True
                                     ),
                True,
                False,
                False)

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
        if not wdir:
            raise PreventUpdate

        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_id = ctx_trigger['index']

        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update
            conn.execute("UPDATE targets SET bookmark = ? WHERE peak_label = ?", [bool(bookmarks[trigger_id]),
                                                                                  targets[trigger_id]])
        
        status = "bookmarked" if bookmarks[trigger_id] else "unbookmarked"
        logger.info(f"Target '{targets[trigger_id]}' was {status}.")

        return fac.AntdNotification(message=f"Target {targets[trigger_id]} has been "
                                            f"{'' if bookmarks[trigger_id] else 'un'}bookmarked",
                                    duration=3,
                                    placement='bottom',
                                    type="success",
                                    showProgress=True,
                                    stack=True)
