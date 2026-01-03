import math
from os import cpu_count
from pathlib import Path

import dash
import feffery_antd_components as fac
import psutil
import time
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .. import tools as T
from ..logging_setup import activate_workspace_logging
import logging
from ..duckdb_manager import (
    build_order_by,
    build_where_and_params,
    calculate_optimal_batch_size,
    compute_chromatograms_in_batches,
    compute_results_in_batches,
    create_pivot,
    duckdb_connection,
    duckdb_connection_mint,
)
from ..plugin_interface import PluginInterface
from .target_optimization import (
    _get_cpu_help_text,
    _get_ram_help_text
)

_label = "Processing"

logger = logging.getLogger(__name__)


class ProcessingPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 7
        logger.info(f'Initiated {_label} plugin')

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)

    def outputs(self):
        return None


_layout = html.Div(
    [
        fac.AntdFlex(
            [
                fac.AntdFlex(
                    [
                        fac.AntdTitle(
                            'Processing', level=4, style={'margin': '0'}
                        ),
                        fac.AntdIcon(
                            id='processing-tour-icon',
                            icon='pi-info',
                            style={"cursor": "pointer", 'paddingLeft': '10px'},
                        ),
                        fac.AntdSpace(
                            [
                                fac.AntdTooltip(
                                    fac.AntdButton(
                                        'Run MINT',
                                        id='processing-btn',
                                        style={'textTransform': 'uppercase', "margin": "0 10px"},
                                    ),
                                    title="Calculate peak areas (integration) for all targets and MS-files.",
                                    placement="bottom"
                                ),
                            ],
                            addSplitLine=False,
                            size="small",
                            style={"margin": "0 10px"},
                        ),
                    ],
                    align='center',
                ),
                fac.AntdFlex(
                    [
                        fac.AntdTooltip(
                            fac.AntdButton(
                                'Download Results',
                                id='processing-download-btn',
                                icon=fac.AntdIcon(icon='antd-download'),
                                iconPosition='end',
                                style={'textTransform': 'uppercase'},
                            ),
                            title="Download the complete results table as a CSV file.",
                            placement="bottom"
                        ),
                        fac.AntdDropdown(
                            id='processing-options',
                            title='Options',
                            buttonMode=True,
                            arrow=True,
                            menuItems=[
                                {'title': fac.AntdText('Delete selected', strong=True, type='warning'),
                                 'key': 'processing-delete-selected'},
                                {'title': fac.AntdText('Clear table', strong=True, type='danger'),
                                 'key': 'processing-delete-all'},
                            ],
                            buttonProps={'style': {'textTransform': 'uppercase'}},
                        ),
                    ],
                    id='processing-options-wrapper',
                    align='center',
                    gap='small',
                ),
            ],
            justify="space-between",
            align="center",
            gap="middle",
        ),
        html.Div(
            [
                fac.AntdFlex(
                    [
                        fac.AntdText('Targets to display', strong=True),
                        fac.AntdSelect(
                            id='processing-peak-select',
                            mode='multiple',
                            allowClear=True,
                            placeholder='Select one or more targets',
                            maxTagCount=4,
                            style={'minWidth': '320px', 'maxWidth': '540px'},
                            optionFilterProp='label',
                        ),
                        fac.AntdText(
                            'Use the dropdown to limit the table to specific compounds.',
                            type='secondary',
                        ),
                    ],
                    align='center',
                    gap='small',
                    wrap=True,
                    style={'paddingBottom': '0.75rem'},
                ),
                fac.AntdSpin(
                    fac.AntdTable(
                        id='results-table',
                        containerId='results-table-container',
                        columns=[
                            {
                                'title': 'Target',
                                'dataIndex': 'peak_label',
                                'width': '260px',
                                'fixed': 'left'
                            },
                            {
                                'title': 'MS-File Label',
                                'dataIndex': 'ms_file_label',
                                'width': '260px',
                                'fixed': 'left'
                            },
                            {
                                'title': 'MS-Type',
                                'dataIndex': 'ms_type',
                                'width': '120px',
                            },
                            {
                                'title': 'peak_area',
                                'dataIndex': 'peak_area',
                                'width': '120px',
                            },
                            {
                                'title': 'peak_area_top3',
                                'dataIndex': 'peak_area_top3',
                                'width': '150px',
                            },
                            {
                                'title': 'peak_mean',
                                'dataIndex': 'peak_mean',
                                'width': '120px',
                            },
                            {
                                'title': 'peak_median',
                                'dataIndex': 'peak_median',
                                'width': '130px',
                            },
                            {
                                'title': 'peak_n_datapoints',
                                'dataIndex': 'peak_n_datapoints',
                                'width': '170px',
                            },
                            {
                                'title': 'peak_min',
                                'dataIndex': 'peak_min',
                                'width': '120px',
                            },
                            {
                                'title': 'peak_max',
                                'dataIndex': 'peak_max',
                                'width': '120px',
                            },
                            {
                                'title': 'peak_rt_of_max',
                                'dataIndex': 'peak_rt_of_max',
                                'width': '150px',
                            },
                            {
                                'title': 'total_intensity',
                                'dataIndex': 'total_intensity',
                                'width': '150px',
                            },
                            {
                                'title': 'Intensity',
                                'dataIndex': 'intensity',
                                'width': '260px',
                                'renderOptions': {'renderType': 'mini-area'},
                            },

                        ],
                        titlePopoverInfo={
                            'ms_file_label': {
                                'title': 'MS-File Label',
                                'content': 'MS file identifier; matches the MS-Files table.',
                            },
                            'label': {
                                'title': 'Label',
                                'content': 'Friendly label from MS-Files.',
                            },
                            'dash_component': {
                                'title': 'Component',
                                'content': 'Internal dash component name.',
                            },
                            'use_for_optimization': {
                                'title': 'For Optimization',
                                'content': 'If true, file was included in optimization steps.',
                            },
                            'use_for_analysis': {
                                'title': 'For Analysis',
                                'content': 'If true, file was included in analysis outputs.',
                            },
                            'sample_type': {
                                'title': 'Sample Type',
                                'content': 'Sample category (Sample, QC, Blank, Standard).',
                            },
                            'polarity': {
                                'title': 'Polarity',
                                'content': 'Instrument polarity (Positive or Negative).',
                            },
                            'ms_type': {
                                'title': 'MS Type',
                                'content': 'Acquisition type (ms1 or ms2).',
                            },
                            'file_type': {
                                'title': 'File Type',
                                'content': 'File format (e.g., mzML, mzXML).',
                            },
                            'peak_label': {
                                'title': 'Target',
                                'content': 'Target name from the Targets table.',
                            },
                            'peak_area': {
                                'title': 'peak_area',
                                'content': 'Integrated area under the peak.',
                            },
                            'peak_area_top3': {
                                'title': 'peak_area_top3',
                                'content': 'Sum of the three highest intensity points.',
                            },
                            'peak_mean': {
                                'title': 'peak_mean',
                                'content': 'Mean intensity across the peak.',
                            },
                            'peak_median': {
                                'title': 'peak_median',
                                'content': 'Median intensity across the peak.',
                            },
                            'peak_n_datapoints': {
                                'title': 'peak_n_datapoints',
                                'content': 'Number of datapoints spanning the peak.',
                            },
                            'peak_min': {
                                'title': 'peak_min',
                                'content': 'Minimum intensity within the peak.',
                            },
                            'peak_max': {
                                'title': 'peak_max',
                                'content': 'Maximum intensity within the peak.',
                            },
                            'peak_rt_of_max': {
                                'title': 'peak_rt_of_max',
                                'content': 'Retention time at maximum intensity.',
                            },
                            'total_intensity': {
                                'title': 'total_intensity',
                                'content': 'Sum of all intensities in the peak window.',
                            },
                            'intensity': {
                                'title': 'Intensity',
                                'content': 'Mini-plot showing the chromatogram shape.',
                            },
                        },
                        filterOptions={
                            'peak_label': {'filterMode': 'keyword'},
                            'ms_file_label': {'filterMode': 'keyword'},
                            'sample_type': {'filterMode': 'checkbox'},
                            'ms_type': {'filterMode': 'checkbox',
                                        'filterCustomItems': ['MS1', 'MS2']},
                        },
                        sortOptions={'sortDataIndexes': ['peak_label', 'peak_area', 'peak_area_top3', 'peak_mean',
                                                         'peak_median', 'peak_n_datapoints', 'peak_min', 'peak_max',
                                                         'peak_rt_of_max', 'total_intensity']},
                        pagination={
                            'position': 'bottomCenter',
                            'pageSize': 15,
                            'showQuickJumper': True,
                            'showSizeChanger': True,
                            'pageSizeOptions': [5, 10, 15, 25, 50, 100],
                        },
                        tableLayout='fixed',
                        maxWidth="calc(100vw - 250px - 4rem)",
                        maxHeight="75vh",
                        locale='en-us',
                        showSorterTooltip=False,
                        rowSelectionType='checkbox',
                        size='small',
                        mode='server-side',
                    ),
                    id='results-table-spin',
                    text='Updating table...',
                    size='small',
                    spinning=False,
                )
            ],
            id='results-table-container',
            style={'paddingTop': '1rem'},
        ),
        html.Div(id="processing-notifications-container"),
        fac.AntdModal(
            [
                fac.AntdFlex(
                    [
                        fac.AntdDivider('Options'),
                        fac.AntdForm(
                            [
                                fac.AntdFormItem(
                                    fac.AntdCheckbox(
                                        id='processing-targets-selection',
                                        label='Compute Bookmarked Targets only',
                                    ),
                                    tooltip='Check to compute only the targets that are bookmarked.'
                                ),
                            ]
                        ),
                        fac.AntdForm(
                            [
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='processing-chromatogram-compute-cpu',
                                        defaultValue=cpu_count() // 2,
                                        min=1,
                                        max=cpu_count() - 2,
                                    ),
                                    label='CPU:',
                                    hasFeedback=True,
                                    help=f"Selected {cpu_count() // 2} / {cpu_count()} cpus",
                                    id='processing-chromatogram-compute-cpu-item'
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='processing-chromatogram-compute-ram',
                                        value=round(psutil.virtual_memory().available * 0.5 / (1024 ** 3), 1),
                                        min=1,
                                        precision=1,
                                        step=0.1,
                                        suffix='GB'
                                    ),
                                    label='RAM:',
                                    hasFeedback=True,
                                    id='processing-chromatogram-compute-ram-item',
                                    help=f"Selected "
                                         f"{round(psutil.virtual_memory().available * 0.5 / (1024 ** 3), 1)}GB / "
                                         f"{round(psutil.virtual_memory().available / (1024 ** 3), 1)}GB available RAM"
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='processing-chromatogram-compute-batch-size',
                                        defaultValue=calculate_optimal_batch_size(
                                            int(psutil.virtual_memory().available * 0.5 / (1024 ** 3)),
                                            100000,  # Assume 100k pairs as default estimate
                                            cpu_count() // 2
                                        ),
                                        min=50,
                                        step=50,
                                    ),
                                    label='Batch Size:',
                                    tooltip='Optimal pairs per batch based on RAM/CPU. '
                                            'Higher values = faster but more memory.',
                                ),
                            ],
                            layout='inline'
                        ),
                        fac.AntdDivider('Recompute'),
                        fac.AntdForm(
                            [
                                fac.AntdFormItem(
                                    fac.AntdCheckbox(
                                        id='processing-recompute',
                                        label='Recompute results'
                                    ),

                                ),
                            ]
                        ),
                        fac.AntdAlert(
                            message='There are already computed results',
                            type='warning',
                            showIcon=True,
                            id='processing-warning',
                            style={'display': 'none'},
                        )
                    ],
                    id='processing-options-container',
                    vertical=True
                ),

                html.Div(
                    [
                        html.H4("Running MINT..."),
                        fac.AntdText(
                            id='processing-progress-stage',
                            style={'marginBottom': '0.5rem'},
                        ),
                        fac.AntdProgress(
                            id='processing-progress',
                            percent=0,
                        ),
                        fac.AntdText(
                            id='processing-progress-detail',
                            type='secondary',
                            style={
                                'marginTop': '0.5rem',
                                'marginBottom': '0.75rem',
                            },
                        ),
                        fac.AntdButton(
                            'Cancel',
                            id='cancel-processing',
                            style={
                                'alignText': 'center',
                                'marginTop': '0.25rem',
                            },
                        ),
                    ],
                    id='processing-progress-container',
                    style={'display': 'none'},
                ),
            ],
            id='processing-modal',
            title='Run MINT',
            width=900,
            renderFooter=True,
            locale='en-us',
            confirmAutoSpin=True,
            loadingOkText='Running...',
            okClickClose=False,
            closable=False,
            maskClosable=False,
            destroyOnClose=True,
            okText="Run",
            centered=True,
            styles={'body': {'height': "50vh"}},
        ),
        fac.AntdModal(
            "Are you sure you want to delete the selected results?",
            title="Delete confirmation",
            id="processing-delete-confirmation-modal",
            okButtonProps={"danger": True},
            renderFooter=True,
            locale='en-us',
        ),
        fac.AntdModal(
            [
                fac.AntdFlex(
                    [
                        fac.AntdDivider(
                            'All results',
                            size='small'
                        ),

                        fac.AntdFlex(
                            [
                                fac.AntdForm(
                                    [
                                        fac.AntdFormItem(
                                            [
                                                fac.AntdSelect(
                                                    options=['peak_area', 'peak_area_top3', 'peak_mean',
                                                             'peak_median', 'peak_n_datapoints', 'peak_min', 'peak_max',
                                                             'peak_rt_of_max', 'total_intensity'],
                                                    mode="multiple",
                                                    value=['peak_area', 'peak_area_top3', 'peak_mean',
                                                           'peak_median', 'peak_n_datapoints', 'peak_min', 'peak_max',
                                                           'peak_rt_of_max', 'total_intensity'],
                                                    style={"width": "100%"},
                                                    locale="en-us",
                                                    id='download-options-all-results'
                                                ),
                                            ],
                                            label='Download all results',
                                            tooltip='Download all results in tabular format',
                                            hasFeedback=True,
                                            id='download-options-all-results-item'
                                        ),
                                    ],
                                    layout='vertical',
                                    style={'flex': 1}
                                ),
                                fac.AntdButton(
                                    'Download',
                                    id='download-all-results-btn',
                                    type='primary',
                                )
                            ],
                            justify='center',
                            align='center',
                            gap=10
                        ),
                        fac.AntdDivider(
                            'Dense Matrix',
                            size='small'
                        ),
                        fac.AntdFlex(
                            [
                                fac.AntdForm(
                                    [
                                        fac.AntdFormItem(
                                            [
                                                fac.AntdSelect(
                                                    options=['ms_file_label', 'peak_label', 'ms_type'],
                                                    value=['ms_file_label'],
                                                    style={"width": "100%"},
                                                    locale="en-us",
                                                    id='download-densematrix-rows',
                                                ),
                                            ],
                                            label='Rows:',
                                            tooltip='Rows information',
                                            hasFeedback=True,
                                            id='download-densematrix-rows-item',
                                        ),
                                        fac.AntdFormItem(
                                            [
                                                fac.AntdSelect(
                                                    options=['peak_label', 'ms_file_label', 'ms_type'],
                                                    value=['peak_label'],
                                                    style={"width": "100%"},
                                                    locale="en-us",
                                                    id='download-densematrix-cols'
                                                ),
                                            ],
                                            label='Columns:',
                                            tooltip='Columns information',
                                            hasFeedback=True,
                                            id='download-densematrix-cols-item',
                                        ),
                                        fac.AntdFormItem(
                                            [
                                                fac.AntdSelect(
                                                    options=['peak_area', 'peak_area_top3', 'peak_mean', 'peak_median',
                                                             'peak_n_datapoints', 'peak_min', 'peak_max',
                                                             'peak_rt_of_max', 'total_intensity'],
                                                    value=['peak_area'],
                                                    style={"width": "100%"},
                                                    locale="en-us",
                                                    id='download-densematrix-value'
                                                ),
                                            ],
                                            label='Value:',
                                            tooltip='Value information',
                                            hasFeedback=True,
                                            id='download-densematrix-value-item'
                                        ),
                                    ],
                                    layout='vertical',
                                    style={'flexGrow': 1}
                                ),
                                fac.AntdButton(
                                    'Download',
                                    type='primary',
                                    id='download-densematrix-results-btn',
                                )
                            ],
                            justify='center',
                            align='center',
                            gap=10
                        ),

                    ],
                    vertical=True
                )
            ],
            title="Download results",
            id="download-results-modal",
            width=700,
            locale='en-us',
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'This tutorial walks through running MINT, selecting targets, and reviewing results.',
                },
                {
                    'title': 'Run processing',
                    'description': 'Click “Run MINT” to compute chromatograms and results for the selected workspace.',
                    'targetSelector': '#processing-btn'
                },
                {
                    'title': 'Pick targets',
                    'description': 'Choose one or more targets to show in the results table after running MINT.',
                    'targetSelector': '#processing-peak-select'
                },
                {
                    'title': 'Review results',
                    'description': 'Filter/sort results.',
                    'targetSelector': '#results-table-container'
                },
                {
                    'title': 'Export or clean up',
                    'description': 'Download all results/dense matrices or delete selected/all rows from the options.',
                    'targetSelector': '#processing-options-wrapper'
                },
            ],
            id='processing-tour',
            open=False,
            current=0,
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Need help?',
                    'description': 'Click the info icon to open a quick tour of Processing.',
                    'targetSelector': '#processing-tour-icon',
                },
            ],
            mask=False,
            placement='rightTop',
            open=False,
            current=0,
            id='processing-tour-hint',
            className='targets-tour-hint',
            style={
                'background': '#ffffff',
                'border': '0.5px solid #1677ff',
                'boxShadow': '0 6px 16px rgba(0,0,0,0.15), 0 0 0 1px rgba(22,119,255,0.2)',
                'opacity': 1,
            },
        ),
        dcc.Store('results-action-store'),
        dcc.Store(id='processing-tour-hint-store', data={'open': False}, storage_type='local'),
        dcc.Download('download-csv'),
    ]
)


def layout():
    return _layout


def _build_delete_modal_content(clickedKey: str, selectedRows: list) -> tuple[bool, fac.AntdFlex]:
    """
    Build the delete confirmation modal content based on the clicked action.
    
    Args:
        clickedKey: The key of the clicked option ('processing-delete-selected' or 'processing-delete-all')
        selectedRows: List of selected rows from the table
        
    Returns:
        Tuple of (modal_visible, modal_children)
        
    Raises:
        PreventUpdate: If validation fails (no rows selected for delete-selected)
    """
    if clickedKey == "processing-delete-selected":
        if not selectedRows:
            logger.debug("_build_delete_modal_content: PreventUpdate because no rows selected for delete-selected")
            raise PreventUpdate
        
        children = fac.AntdFlex(
            [
                fac.AntdText("This action will delete selected results and cannot be undone?",
                             strong=True),
                fac.AntdText("Are you sure you want to delete the selected results?")
            ],
            vertical=True,
        )
    else:  # processing-delete-all
        children = fac.AntdFlex(
            [
                fac.AntdText("This action will delete ALL results and cannot be undone?",
                             strong=True, type="danger"),
                fac.AntdText("Are you sure you want to delete the ALL results?")
            ],
            vertical=True,
        )
    
    return True, children


def _load_peaks_from_results(wdir: str, current_value: list) -> tuple[list, list]:
    """
    Load available peak labels from results table and manage selection.
    
    Args:
        wdir: Workspace directory path
        current_value: Currently selected peak labels
        
    Returns:
        Tuple of (options, selected_values) for the peak selector dropdown
        
    Raises:
        PreventUpdate: If workspace is invalid or database unavailable
    """
    if not wdir:
        logger.debug("_load_peaks_from_results: PreventUpdate because wdir is not set")
        raise PreventUpdate
        
    with duckdb_connection(wdir) as conn:
        if conn is None:
            logger.debug("_load_peaks_from_results: PreventUpdate because database connection is None")
            raise PreventUpdate
        peaks = conn.execute(
            "SELECT DISTINCT peak_label FROM results ORDER BY peak_label"
        ).fetchall()

    options = [
        {'label': peak[0], 'value': peak[0]}
        for peak in peaks
        if peak and peak[0] is not None
    ]

    if not options:
        return [], []
    
    if not current_value:
        return options, [options[0]['value']]
    
    valid_values = {opt['value'] for opt in options}
    filtered_value = [v for v in current_value if v in valid_values]
    
    # If current selection is no longer valid (deleted), auto-select first available
    if not filtered_value and options:
        return options, [options[0]['value']]
    
    return options, filtered_value


def _download_all_results(wdir: str, ws_name: str, selected_columns: list) -> tuple:
    """
    Download all results with selected columns as CSV.
    
    Args:
        wdir: Workspace directory path
        ws_name: Workspace name for filename
        selected_columns: List of column names to include in download
        
    Returns:
        Tuple of (download_data, notification) where download_data is for dcc.Download
        and notification is AntdNotification or None
    """
    allowed_cols = {
        'peak_area', 'peak_area_top3', 'peak_mean', 'peak_median',
        'peak_n_datapoints', 'peak_min', 'peak_max', 'peak_rt_of_max', 'total_intensity',
    }
    
    if not selected_columns or not isinstance(selected_columns, list):
        return dash.no_update, fac.AntdNotification(
            message="Download Results",
            description="Select at least one column.",
            type="warning",
            duration=4,
            placement="bottom",
            showProgress=True,
        )
    
    safe_cols = [c for c in selected_columns if c in allowed_cols]
    if not safe_cols:
        return dash.no_update, fac.AntdNotification(
            message="Download Results",
            description="No valid columns selected.",
            type="warning",
            duration=4,
            placement="bottom",
            showProgress=True,
        )
    
    with duckdb_connection(wdir) as conn:
        if conn is None:
            return dash.no_update, fac.AntdNotification(
                message="Download Results",
                description="Database unavailable.",
                type="error",
                duration=4,
                placement="bottom",
                showProgress=True,
            )
        
        cols = ', '.join(safe_cols)
        filename = f"{T.today()}-MINT__{ws_name}-all_results.csv"
        logger.info(f"Download request: {filename}")
        
        # Use DuckDB COPY for faster export (2.87x speedup vs pandas)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            tmp_path = tmp.name
        
        conn.execute(f"""
            COPY (
                SELECT 
                    r.peak_label, 
                    r.ms_file_label, 
                    s.ms_type,
                    {cols} 
                FROM results r 
                JOIN samples s ON s.ms_file_label = r.ms_file_label 
                ORDER BY s.ms_type, r.peak_label, r.ms_file_label
            ) TO ? (HEADER, DELIMITER ',')
        """, (tmp_path,))
        
    return dcc.send_file(tmp_path, filename=filename), dash.no_update


def _download_dense_matrix(wdir: str, ws_name: str, rows: list, cols: list, value: list) -> tuple:
    """
    Download dense matrix (pivot table) as CSV.
    
    Args:
        wdir: Workspace directory path
        ws_name: Workspace name for filename
        rows: List with row column name
        cols: List with column column name
        value: List with value column name
        
    Returns:
        Tuple of (download_data, notification) where download_data is for dcc.Download
        and notification is AntdNotification or None
    """
    allowed_rows_cols = {'ms_file_label', 'peak_label', 'ms_type'}
    allowed_values = {
        'peak_area', 'peak_area_top3', 'peak_mean', 'peak_median',
        'peak_n_datapoints', 'peak_min', 'peak_max', 'peak_rt_of_max', 'total_intensity',
    }
    
    if not rows or not cols or not value:
        return dash.no_update, fac.AntdNotification(
            message="Download Results",
            description="Missing row, column, or value selection.",
            type="warning",
            duration=4,
            placement="bottom",
            showProgress=True,
        )
    
    row_col = rows[0]
    col_col = cols[0]
    val_col = value[0]
    
    if row_col not in allowed_rows_cols or col_col not in allowed_rows_cols:
        return dash.no_update, fac.AntdNotification(
            message="Download Results",
            description="Invalid row/column selection.",
            type="warning",
            duration=4,
            placement="bottom",
            showProgress=True,
        )
    
    if val_col not in allowed_values:
        return dash.no_update, fac.AntdNotification(
            message="Download Results",
            description="Invalid value selection.",
            type="warning",
            duration=4,
            placement="bottom",
            showProgress=True,
        )
    
    with duckdb_connection(wdir) as conn:
        if conn is None:
            return dash.no_update, fac.AntdNotification(
                message="Download Results",
                description="Database unavailable.",
                type="error",
                duration=4,
                placement="bottom",
                showProgress=True,
            )
        
        df = create_pivot(conn, rows[0], cols[0], value[0], table='results')
        filename = f"{T.today()}-MINT__{ws_name}-{value[0]}_results.csv"
        logger.info(f"Download request (dense matrix): {filename}")
    
    return dcc.send_data_frame(df.to_csv, filename, index=False), dash.no_update


def _delete_selected_results(wdir: str, selected_rows: list) -> tuple:
    """
    Delete selected results from the database.
    
    Args:
        wdir: Workspace directory path
        selected_rows: List of selected row dictionaries with peak_label and ms_file_label
        
    Returns:
        Tuple of (notification, results_action_store, total_removed)
        where total_removed is [unique_peaks, unique_files]
    """
    if not selected_rows:
        return (
            None,  # No notification on validation failure
            {'action': 'delete', 'status': 'failed'},
            [0, 0]
        )
    
    # Extract unique pairs
    remove_pairs = list({
        (row["peak_label"], row["ms_file_label"]) for row in selected_rows
        if row.get("peak_label") and row.get("ms_file_label")
    })
    unique_peaks = len({p for p, _ in remove_pairs})
    unique_files = len({m for _, m in remove_pairs})
    
    with duckdb_connection(wdir) as conn:
        if conn is None:
            logger.debug("_delete_selected_results: PreventUpdate because database connection is None")
            raise PreventUpdate
        
        try:
            conn.execute("BEGIN")
            total_removed = [unique_peaks, unique_files]
            if remove_pairs:
                placeholders = ", ".join(["(?, ?)"] * len(remove_pairs))
                params = [v for pair in remove_pairs for v in pair]
                conn.execute(
                    f"DELETE FROM results WHERE (peak_label, ms_file_label) IN ({placeholders})",
                    params
                )
                results_action_store = {'action': 'delete', 'status': 'success'}
            else:
                results_action_store = {'action': 'delete', 'status': 'failed'}
                total_removed = [0, 0]
            conn.execute("COMMIT")
            return (None, results_action_store, total_removed)
        except Exception as e:
            conn.execute("ROLLBACK")
            return (
                fac.AntdNotification(
                    message="Failed to delete results",
                    description=f"Error: {e}",
                    type="error",
                    duration=4,
                    placement='bottom',
                    showProgress=True,
                    stack=True
                ),
                {'action': 'delete', 'status': 'failed'},
                [0, 0]
            )


def _delete_all_results(wdir: str) -> tuple:
    """
    Delete all results from the database.
    
    Args:
        wdir: Workspace directory path
        
    Returns:
        Tuple of (notification, results_action_store, total_removed)
        where total_removed is [unique_peaks, unique_files]
    """
    with duckdb_connection(wdir) as conn:
        if conn is None:
            logger.debug("_delete_all_results: PreventUpdate because database connection is None")
            raise PreventUpdate
        
        try:
            conn.execute("BEGIN")
            total_removed_q = conn.execute("""
                SELECT COUNT(DISTINCT peak_label)    as unique_peaks,
                       COUNT(DISTINCT ms_file_label) as unique_files
                FROM results
            """).fetchone()
            
            results_action_store = {'action': 'delete', 'status': 'failed'}
            total_removed = [0, 0]
            
            if total_removed_q:
                total_removed = list(total_removed_q)
                conn.execute("DELETE FROM results")
                results_action_store = {'action': 'delete', 'status': 'success'}
                logger.info(f"Deleted {total_removed[0]} targets and {total_removed[1]} samples from results.")
            
            conn.execute("COMMIT")
            return (None, results_action_store, total_removed)
        except Exception as e:
            logger.error("Failed to delete results.", exc_info=True)
            conn.execute("ROLLBACK")
            return (
                fac.AntdNotification(
                    message="Failed to delete results",
                    description=f"Error: {e}",
                    type="error",
                    duration=4,
                    placement='bottom',
                    showProgress=True,
                    stack=True
                ),
                {'action': 'delete', 'status': 'failed'},
                [0, 0]
            )


def callbacks(app, fsc, cache):
    @app.callback(
        Output("processing-notifications-container", "children"),
        Input('section-context', 'data'),
        Input("wdir", "data"),
        prevent_initial_call=True,
    )
    def warn_missing_workspace(section_context, wdir):
        if not section_context or section_context.get('page') != 'Processing':
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
        Output("results-table", "data"),
        Output("results-table", "selectedRowKeys"),
        Output("results-table", "pagination"),

        Input('section-context', 'data'),
        Input("results-action-store", "data"),
        Input('processing-peak-select', 'value'),
        Input('results-table', 'pagination'),
        Input('results-table', 'filter'),
        Input('results-table', 'sorter'),
        State('results-table', 'filterOptions'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def results_table(section_context, results_actions, selected_peaks,
                      pagination, filter_, sorter, filterOptions, wdir):
        if section_context and section_context['page'] != 'Processing':
            logger.debug(f"results_table: PreventUpdate because current page is {section_context.get('page')}")
            raise PreventUpdate

        if not wdir:
            logger.debug("results_table: PreventUpdate because wdir is not set")
            raise PreventUpdate

        # Autosave results table on tab load/refresh for durability (throttled to limit I/O).
        # Skip if processing just completed (already backed up there)
        try:
            # Check if processing just completed
            skip_backup = False
            if isinstance(results_actions, dict):
                if results_actions.get('action') == 'processing' and results_actions.get('status') == 'completed':
                    timestamp = results_actions.get('timestamp', 0)
                    if time.time() - timestamp < 10:  # Within last 10 seconds
                        skip_backup = True
                        logger.debug("Skipping backup - processing just completed")
            
            if not skip_backup:
                results_dir = Path(wdir) / "results"
                results_dir.mkdir(parents=True, exist_ok=True)
                backup_path = results_dir / "results_backup.csv"
                should_write = True
                if backup_path.exists():
                    last_write = backup_path.stat().st_mtime
                    should_write = (time.time() - last_write) > 30
                if should_write:
                    with duckdb_connection(wdir) as conn:
                        if conn is None:
                            logger.debug("results_table: PreventUpdate because database connection is None (backup)")
                            raise PreventUpdate
                        conn.execute(
                            "COPY (SELECT * FROM results) TO ? (HEADER, DELIMITER ',')",
                            (str(backup_path),),
                        )
                    logger.debug(f"Auto-backed up results to {backup_path}")
        except PreventUpdate:
            raise
        except Exception:
            pass

        pagination = pagination or {
            'position': 'bottomCenter',
            'pageSize': 15,
            'showQuickJumper': True,
            'showSizeChanger': True,
            'pageSizeOptions': [5, 10, 15, 25, 50, 100],
        }
        base_page_size_options = [5, 10, 15, 25, 50, 100]
        page_size_options = base_page_size_options.copy()
        try:
            page_size = int(pagination.get('pageSize') or 15)
        except (TypeError, ValueError):
            page_size = 25
        if page_size <= 0:
            page_size = 25
        current = pagination.get('current') or 1

        filterOptions = filterOptions or {}
        selected_peaks = selected_peaks or []
        if not isinstance(selected_peaks, list):
            selected_peaks = [selected_peaks]

        if not selected_peaks:
            return (
                [],
                [],
                {
                    **pagination,
                    'total': 0,
                    'current': 1,
                    'pageSize': page_size,
                    'pageSizeOptions': page_size_options,
                },
            )

        with duckdb_connection(wdir) as conn:
            if conn is None:
                logger.debug("results_table: PreventUpdate because database connection is None")
                raise PreventUpdate
            results_schema = conn.execute("DESCRIBE results").fetchall()
            samples_schema = conn.execute("DESCRIBE samples").fetchall()
            column_types = {row[0]: row[1] for row in results_schema}
            column_types.update({row[0]: row[1] for row in samples_schema})

            where_sql, params = build_where_and_params(filter_, filterOptions)
            where_sql = f"{where_sql} {'AND' if where_sql else 'WHERE'} r.peak_label IN ?"
            params.append(selected_peaks)

            order_params: list = []
            order_values = ""
            if selected_peaks:
                order_values = ", ".join(["(?, ?)"] * len(selected_peaks))
                for idx, peak in enumerate(selected_peaks):
                    order_params.extend([idx, peak])

            order_by_sql = (
                "ORDER BY __peak_order__, ms_file_label"
                if selected_peaks
                else build_order_by(
                    sorter,
                    column_types,
                    tie=('peak_label', 'ASC'),
                    nocase_text=True
                )
            )

            filtered_sql = f"""
            WITH
            {f"target_order AS (SELECT * FROM (VALUES {order_values}) AS t(ord, target_peak_label))," if order_values else ""}
            filtered AS (
              SELECT r.*, UPPER(s.ms_type) AS ms_type, s.sample_type
              {", COALESCE(tord.ord, 1e9) AS __peak_order__" if order_values else ""}
              FROM results r
              LEFT JOIN samples s USING (ms_file_label)
              {f"LEFT JOIN target_order tord ON tord.target_peak_label = r.peak_label" if order_values else ""}
              {where_sql}
            )
            """

            count_sql = filtered_sql + "SELECT COUNT(*) AS __total__ FROM filtered;"
            number_records = conn.execute(count_sql, order_params + params).fetchone()[0] or 0

            page_size_options = sorted(
                set(base_page_size_options + ([number_records] if number_records else []))
            )
            if page_size not in page_size_options:
                page_size = min(25, number_records) if number_records else 25
            if number_records and page_size > number_records:
                page_size = number_records
            effective_page_size = page_size if page_size > 0 else (number_records or 1)
            max_page = max(math.ceil(number_records / effective_page_size), 1) if number_records else 1
            current = min(max(current, 1), max_page)
            offset = (current - 1) * effective_page_size if number_records else 0

            sql = filtered_sql + f"""
            , paged AS (
              SELECT *
              FROM filtered
              {(' ' + order_by_sql) if order_by_sql else ''}
              LIMIT ? OFFSET ?
            )
            SELECT * FROM paged;
            """

            params_paged = order_params + params + [effective_page_size, offset]
            df = conn.execute(sql, params_paged).df()

        if number_records and params_paged[-1] != (current - 1) * effective_page_size:
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    logger.debug("results_table: PreventUpdate because database connection is None (repaginate)")
                    raise PreventUpdate
                params_paged = order_params + params + [effective_page_size, (current - 1) * effective_page_size]
                df = conn.execute(sql, params_paged).df()

        data = [
            {
                'key': f'{row.peak_label}-{row.ms_file_label}',
                'peak_label': row.peak_label,
                'ms_file_label': row.ms_file_label,
                'ms_type': getattr(row, 'ms_type', None),
                'peak_area': row.peak_area,
                'peak_area_top3': row.peak_area_top3,
                'peak_mean': row.peak_mean,
                'peak_median': row.peak_median,
                'peak_n_datapoints': row.peak_n_datapoints,
                'peak_min': row.peak_min,
                'peak_max': row.peak_max,
                'peak_rt_of_max': row.peak_rt_of_max,
                'total_intensity': row.total_intensity,
                'intensity': row.intensity,
                'sample_type': getattr(row, 'sample_type', None),
            }
            for row in df.itertuples(index=False)
        ]

        # If current page is empty but there are records, navigate to the last valid page
        if len(data) == 0 and number_records > 0:
            max_page = max(math.ceil(number_records / effective_page_size), 1)
            current = max_page
            # Re-query with the adjusted page
            offset = (current - 1) * effective_page_size
            params_paged = order_params + params + [effective_page_size, offset]
            df = conn.execute(sql, params_paged).df()
            data = [
                {
                    'key': f'{row.peak_label}-{row.ms_file_label}',
                    'peak_label': row.peak_label,
                    'ms_file_label': row.ms_file_label,
                    'ms_type': getattr(row, 'ms_type', None),
                    'peak_area': row.peak_area,
                    'peak_area_top3': row.peak_area_top3,
                    'peak_mean': row.peak_mean,
                    'peak_median': row.peak_median,
                    'peak_n_datapoints': row.peak_n_datapoints,
                    'peak_min': row.peak_min,
                    'peak_max': row.peak_max,
                    'peak_rt_of_max': row.peak_rt_of_max,
                    'total_intensity': row.total_intensity,
                    'intensity': row.intensity,
                    'sample_type': getattr(row, 'sample_type', None),
                }
                for row in df.itertuples(index=False)
            ]

        return [
            data,
            [],
            {
                **pagination,
                'total': number_records,
                'current': current,
                'pageSize': page_size,
                'pageSizeOptions': page_size_options,
            },
        ]

    @app.callback(
        Output('processing-peak-select', 'options'),
        Output('processing-peak-select', 'value', allow_duplicate=True),
        Input('section-context', 'data'),
        Input('wdir', 'data'),
        Input('results-action-store', 'data'),
        State('processing-peak-select', 'value'),
        prevent_initial_call=True,
    )
    def load_available_peaks(section_context, wdir, results_actions, current_value):
        if section_context and section_context['page'] != 'Processing':
            logger.debug(f"load_available_peaks: PreventUpdate because current page is {section_context.get('page')}")
            raise PreventUpdate
        
        return _load_peaks_from_results(wdir, current_value)

    @app.callback(
        Output("processing-delete-confirmation-modal", "visible"),
        Output("processing-delete-confirmation-modal", "children"),

        Input("processing-options", "nClicks"),
        State("processing-options", "clickedKey"),
        State('results-table', 'selectedRows'),
        prevent_initial_call=True
    )
    def toggle_modal(nClicks, clickedKey, selectedRows):
        ctx = dash.callback_context
        if (
                not ctx.triggered or
                not nClicks or
                not clickedKey or
                clickedKey not in ['processing-delete-selected', 'processing-delete-all']
        ):
            logger.debug(f"toggle_modal: PreventUpdate because triggered={ctx.triggered}, nClicks={nClicks}, clickedKey={clickedKey}")
            raise PreventUpdate

        return _build_delete_modal_content(clickedKey, selectedRows)

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("results-action-store", "data", allow_duplicate=True),

        Input("processing-delete-confirmation-modal", "okCounts"),
        State('results-table', 'selectedRows'),
        State("processing-options", "clickedKey"),
        State("wdir", "data"),
        background=True,
        running=[
            (Output("results-table-spin", "spinning"), True, False),
            (Output("processing-delete-confirmation-modal", "confirmLoading"), True, False),
        ],
        prevent_initial_call=True,
    )
    def confirm_and_delete(okCounts, selectedRows, clickedKey, wdir):
        if okCounts is None:
            logger.debug("confirm_and_delete: PreventUpdate because okCounts is None")
            raise PreventUpdate
        if not wdir:
            logger.debug("confirm_and_delete: PreventUpdate because wdir is not set")
            raise PreventUpdate
        
        # Delegate to appropriate standalone function
        if clickedKey == "processing-delete-selected":
            error_notif, results_action_store, total_removed = _delete_selected_results(wdir, selectedRows)
        else:  # processing-delete-all
            error_notif, results_action_store, total_removed = _delete_all_results(wdir)
        
        # If there was an error, return it immediately
        if error_notif is not None:
            return (error_notif, results_action_store)
        
        # Return success notification
        return (
            fac.AntdNotification(
                message="Results deleted",
                description=f"Deleted {total_removed[0]} targets and {total_removed[1]} samples.",
                type="success" if total_removed != [0, 0] else "error",
                duration=3,
                placement='bottom',
                showProgress=True,
                stack=True
            ),
            results_action_store
        )

    @app.callback(
        Output("download-results-modal", "visible"),

        Input("processing-download-btn", "nClicks"),
        prevent_initial_call=True,
    )
    def open_download_results(n_clicks):
        if not n_clicks:
            logger.debug("open_download_results: PreventUpdate because n_clicks is None")
            raise PreventUpdate
        return True

    @app.callback(
        Output('download-all-results-btn', 'disabled'),
        Output('download-densematrix-results-btn', 'disabled'),
        Output('download-options-all-results-item', 'validateStatus'),
        Output('download-options-all-results-item', 'help'),
        Output('download-densematrix-rows-item', 'validateStatus'),
        Output('download-densematrix-rows-item', 'help'),
        Output('download-densematrix-cols-item', 'validateStatus'),
        Output('download-densematrix-cols-item', 'help'),
        Output('download-densematrix-value-item', 'validateStatus'),
        Output('download-densematrix-value-item', 'help'),

        Input('download-options-all-results', 'value'),
        Input('download-densematrix-rows', 'value'),
        Input('download-densematrix-cols', 'value'),
        Input('download-densematrix-value', 'value'),
        prevent_initial_call=True
    )
    def validate_download_input(all_results, rows, cols, value):

        if not all_results:
            all_results_status = 'error'
            all_results_help = 'Select at least one column'
            all_results_download = True
        else:
            all_results_status = 'success'
            all_results_help = None
            all_results_download = False

        if not rows:
            rows_status = 'error'
            rows_help = 'Select a valid column'
        else:
            rows_status = 'success'
            rows_help = None

        if not cols:
            cols_status = 'error'
            cols_help = 'Select a valid column'
        else:
            cols_status = 'success'
            cols_help = None

        if not value:
            value_status = 'error'
            value_help = 'Select a valid column'
        else:
            value_status = 'success'
            value_help = None

        densematrix_download = False
        if not rows or not cols or not value:
            densematrix_download = True

        if rows == cols:
            densematrix_download = True
            cols_status = 'error'
            cols_help = 'Select a different column'

        return (
            all_results_download,
            densematrix_download,
            all_results_status,
            all_results_help,
            rows_status,
            rows_help,
            cols_status,
            cols_help,
            value_status,
            value_help
        )

    @app.callback(
        Output("download-csv", "data"),
        Output("notifications-container", "children", allow_duplicate=True),

        Input("download-all-results-btn", "nClicks"),
        Input("download-densematrix-results-btn", "nClicks"),
        State("download-options-all-results", "value"),
        State('download-densematrix-rows', 'value'),
        State('download-densematrix-cols', 'value'),
        State('download-densematrix-value', 'value'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def download_results(d_all_clicks, d_dm_clicks, d_options_value, d_dm_rows, d_dm_cols, d_dm_value, wdir):

        ctx = dash.callback_context
        if not ctx.triggered:
            logger.debug("download_results: PreventUpdate because not triggered")
            raise PreventUpdate
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if prop_id == 'download-all-results-btn' and not d_all_clicks:
            logger.debug("download_results: PreventUpdate because download-all-results-btn clicked but no count")
            raise PreventUpdate
        if prop_id == 'download-densematrix-results-btn' and not d_dm_clicks:
            logger.debug("download_results: PreventUpdate because download-densematrix-results-btn clicked but no count")
            raise PreventUpdate

        if not wdir:
            return dash.no_update, fac.AntdNotification(
                message="Download Results",
                description="Workspace not available. Please select or create a workspace.",
                type="error",
                duration=4,
                placement="bottom",
                showProgress=True,
            )

        ws_key = Path(wdir).stem
        with duckdb_connection_mint(Path(wdir).parent.parent) as mint_conn:
            if mint_conn is None:
                logger.debug("download_results: PreventUpdate because mint database connection is None")
                raise PreventUpdate
            ws_row = mint_conn.execute("SELECT name FROM workspaces WHERE key = ?", [ws_key]).fetchone()
            if ws_row is None:
                logger.debug("download_results: PreventUpdate because workspace row not found")
                raise PreventUpdate
            ws_name = ws_row[0]

        # Delegate to appropriate standalone function  
        if prop_id == 'download-all-results-btn':
            return _download_all_results(wdir, ws_name, d_options_value)
        else:
            return _download_dense_matrix(wdir, ws_name, d_dm_rows, d_dm_cols, d_dm_value)

    @app.callback(
        Output('processing-tour', 'current'),
        Output('processing-tour', 'open'),
        Input('processing-tour-icon', 'nClicks'),
        prevent_initial_call=True,
    )
    def processing_tour_open(n_clicks):
        return 0, True

    @app.callback(
        Output('processing-tour-hint', 'open'),
        Output('processing-tour-hint', 'current'),
        Input('processing-tour-hint-store', 'data'),
    )
    def processing_hint_sync(store_data):
        if not store_data:
            logger.debug("processing_hint_sync: PreventUpdate because store_data is empty")
            raise PreventUpdate
        return store_data.get('open', True), 0

    @app.callback(
        Output('processing-tour-hint-store', 'data'),
        Input('processing-tour-hint', 'closeCounts'),
        Input('processing-tour-icon', 'nClicks'),
        State('processing-tour-hint-store', 'data'),
        prevent_initial_call=True,
    )
    def processing_hide_hint(close_counts, n_clicks, store_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            logger.debug("processing_hide_hint: PreventUpdate because not triggered")
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'processing-tour-icon':
            return {'open': False}

        if close_counts:
            return {'open': False}

        return store_data or {'open': True}

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("processing-modal", "visible"),
        Output("processing-warning", "style"),
        Output("processing-progress", "percent", allow_duplicate=True),
        Output("processing-progress-stage", "children", allow_duplicate=True),
        Output("processing-progress-detail", "children", allow_duplicate=True),
        Output("processing-recompute", "checked"),
        Output("processing-chromatogram-compute-cpu", "value"),
        Output("processing-chromatogram-compute-ram", "value"),
        Output("processing-chromatogram-compute-cpu-item", "help", allow_duplicate=True),
        Output("processing-chromatogram-compute-ram-item", "help", allow_duplicate=True),

        Input("processing-btn", "nClicks"),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def open_run_mint_modal(nClicks, wdir):
        if not nClicks:
            logger.debug("open_run_mint_modal: PreventUpdate because nClicks is None")
            raise PreventUpdate

        computed_results = 0
        # check if some results was computed
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return (
                    fac.AntdNotification(
                        message="Run MINT",
                        description="Workspace not available. Please select or create a workspace.",
                        type="error",
                        duration=4,
                        placement="bottom",
                        showProgress=True,
                    ),
                    False,
                    dash.no_update,
                    0,
                    "",
                    "",
                    False,
                    dash.no_update,
                    dash.no_update,
                )

            ms_files = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
            targets = conn.execute("SELECT COUNT(*) FROM targets").fetchone()

            if not ms_files or ms_files[0] == 0 or not targets or targets[0] == 0:
                return (
                    fac.AntdNotification(
                        message="Requirements not met",
                        description="At least one MS-file and one target are required.",
                        type="warning",
                        duration=4,
                        placement="bottom",
                        showProgress=True,
                    ),
                    False,
                    dash.no_update,
                    0,
                    "",
                    "",
                    False,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )

            results = conn.execute("SELECT COUNT(*) FROM results").fetchone()
            if results:
                computed_results = results[0]

        style = {'display': 'block'} if computed_results else {'display': 'none'}

        recompute = bool(computed_results)

        # Smart Default CPU/RAM
        from os import cpu_count
        n_cpus_total = cpu_count()
        default_cpus = max(1, n_cpus_total // 2)
        
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        default_ram = min(float(default_cpus), available_ram_gb)
        available_ram_gb_rounded = round(available_ram_gb, 1)

        help_cpu = f"Selected {default_cpus} / {n_cpus_total} cpus"
        help_ram = f"Selected {default_ram}GB / {available_ram_gb_rounded}GB available RAM"

        return (
            dash.no_update, 
            True, 
            style, 
            0, 
            "", 
            "", 
            recompute,
            default_cpus,
            default_ram,
            help_cpu,
            help_ram
        )

    @app.callback(
        Output('results-action-store', 'data'),
        Output('processing-modal', 'visible', allow_duplicate=True),

        Input('processing-modal', 'okCounts'),
        State('processing-recompute', 'checked'),
        State("processing-chromatogram-compute-cpu", "value"),
        State("processing-chromatogram-compute-ram", "value"),
        State('processing-chromatogram-compute-batch-size', "value"),
        State('processing-targets-selection', 'checked'),
        State('wdir', 'data'),
        background=True,
        running=[
            (Output('processing-progress-container', 'style'), {
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "flexDirection": "column",
                "minWidth": "200px",
                "maxWidth": "400px",
                "margin": "auto",
                'height': "60vh"
            }, {'display': 'none'}),
            (Output("processing-options-container", "style"), {'display': 'none'}, {'display': 'flex'}),

            (Output('processing-modal', 'confirmAutoSpin'), True, False),
            (Output('processing-modal', 'cancelButtonProps'), {'disabled': True},
             {'disabled': False}),
            (Output('processing-modal', 'confirmLoading'), True, False),
        ],
        progress=[
            Output("processing-progress", "percent", allow_duplicate=True),
            Output("processing-progress-stage", "children", allow_duplicate=True),
            Output("processing-progress-detail", "children", allow_duplicate=True),
        ],
        cancel=[
            Input('cancel-processing', 'nClicks')
        ],
        prevent_initial_call=True
    )
    def compute_results(set_progress, okCounts, recompute, n_cpus, ram, batch_size, bookmarked, wdir):

        if not okCounts:
            logger.debug("compute_results: PreventUpdate because okCounts is None")
            raise PreventUpdate

        activate_workspace_logging(wdir)
        start = time.perf_counter()
        
        def progress_adapter(percent, stage="", detail=""):
            if set_progress:
                set_progress((percent, stage or "", detail or ""))

        try:
            logger.info('Starting full processing run.')
            logger.info('Computing chromatograms...')
            progress_adapter(0, "Chromatograms", "Preparing batches...")
            compute_chromatograms_in_batches(wdir, use_for_optimization=False, batch_size=batch_size,
                                             set_progress=progress_adapter, recompute_ms1=False,
                                             recompute_ms2=False, n_cpus=n_cpus, ram=ram, use_bookmarked=bookmarked)
            
            logger.info('Computing results...')
            progress_adapter(0, "Results", "Preparing batches...")
            compute_results_in_batches(wdir=wdir,
                               use_bookmarked= bookmarked,
                               recompute = recompute,
                               batch_size = batch_size,
                               checkpoint_every = 10,
                               set_progress=progress_adapter,
                               n_cpus=n_cpus,
                               ram=ram)


            # Persist the results table to a workspace folder for resilience
            progress_adapter(100, "Results", "Backing up results...")
            try:
                results_dir = Path(wdir) / "results"
                results_dir.mkdir(parents=True, exist_ok=True)
                backup_path = results_dir / "results_backup.csv"
                
                # Use DuckDB's native COPY for much faster CSV export
                with duckdb_connection(wdir) as conn:
                    conn.execute(
                        "COPY (SELECT * FROM results) TO ? (HEADER, DELIMITER ',')",
                        (str(backup_path),)
                    )
                logger.info(f"Backed up results to {backup_path}")
            except Exception:
                logger.warning("Failed to backup results to CSV.", exc_info=True)


            logger.info(f"Results computed in {time.perf_counter() - start:.2f} seconds")
        except Exception:
             logger.error("Processing failed during computation", exc_info=True)
             raise

        return {'action': 'processing', 'status': 'completed', 'timestamp': time.time()}, False

    @app.callback(
        Output('results-action-store', 'data', allow_duplicate=True),
        Output('processing-progress-container', 'style', allow_duplicate=True),
        Output('processing-options-container', 'style', allow_duplicate=True),
        Output('processing-modal', 'visible', allow_duplicate=True),
        Output('processing-progress', 'percent', allow_duplicate=True),
        Output('processing-progress-stage', 'children', allow_duplicate=True),
        Output('processing-progress-detail', 'children', allow_duplicate=True),
        Input('cancel-processing', 'nClicks'),
        prevent_initial_call=True
    )
    def cancel_results_processing(cancel_clicks):
        if not cancel_clicks:
            logger.debug("cancel_results_processing: PreventUpdate because cancel_clicks is None")
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
        Output("processing-chromatogram-compute-cpu-item", "help", allow_duplicate=True),
        Output("processing-chromatogram-compute-ram-item", "help", allow_duplicate=True),
        Output("processing-chromatogram-compute-batch-size", "value"),
        Input("processing-chromatogram-compute-cpu", "value"),
        Input("processing-chromatogram-compute-ram", "value"),
        prevent_initial_call=True
    )
    def update_processing_resource_help(cpu, ram):
        help_cpu = _get_cpu_help_text(cpu)
        help_ram = _get_ram_help_text(ram)
        # Auto-calculate optimal batch size based on current CPU and RAM
        optimal_batch = calculate_optimal_batch_size(
            int(ram) if ram else 8,
            100000,  # Estimate for total pairs
            int(cpu) if cpu else 4
        )
        return help_cpu, help_ram, optimal_batch
