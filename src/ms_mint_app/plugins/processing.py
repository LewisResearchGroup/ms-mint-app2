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

from ..duckdb_manager import duckdb_connection, build_paginated_query_by_peak, create_pivot, \
    duckdb_connection_mint, compute_chromatograms_in_batches, compute_results_in_batches
from ..plugin_interface import PluginInterface

_label = "Processing"


class ProcessingPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 7
        print(f'Initiated {_label} plugin')

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
                                fac.AntdButton(
                                    'Run MINT',
                                    id='processing-btn',
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
                html.Div(
                    fac.AntdDropdown(
                        id='processing-options',
                        title='Options',
                        buttonMode=True,
                        arrow=True,
                        menuItems=[
                            {'title': fac.AntdText('Download', strong=True), 'key': 'processing-download'},
                            {'isDivider': True},
                            {'title': fac.AntdText('Delete selected', strong=True, type='warning'),
                             'key': 'processing-delete-selected'},
                            {'title': fac.AntdText('Clear table', strong=True, type='danger'),
                             'key': 'processing-delete-all'},
                        ],
                        buttonProps={'style': {'textTransform': 'uppercase'}},
                    ),
                    id='processing-options-wrapper',
                ),
            ],
            justify="space-between",
            align="center",
            gap="middle",
        ),
        html.Div(
            [
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
                                'content': 'Raw file format (e.g., mzML, mzXML).',
                            },
                            'run_order': {
                                'title': 'Run Order',
                                'content': 'Injection order for the MS file.',
                            },
                            'plate': {
                                'title': 'Plate',
                                'content': 'Plate identifier if provided.',
                            },
                            'plate_row': {
                                'title': 'Plate Row',
                                'content': 'Plate row (e.g., A, B, C).',
                            },
                            'plate_column': {
                                'title': 'Plate Column',
                                'content': 'Plate column number.',
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
                                        'filterCustomItems': ['ms1', 'ms2']},
                        },
                        sortOptions={'sortDataIndexes': ['peak_label', 'peak_area', 'peak_area_top3', 'peak_mean',
                                                         'peak_median', 'peak_n_datapoints', 'peak_min', 'peak_max',
                                                         'peak_rt_of_max', 'total_intensity']},
                        pagination={
                            'position': 'bottomCenter',
                            'pageSize': 1,
                            'current': 1,
                            'showSizeChanger': True,
                            'pageSizeOptions': [1, 5, 10, 25, 50, 100],
                            'showQuickJumper': True,
                        },
                        tableLayout='fixed',
                        maxWidth="calc(100vw - 250px - 4rem)",
                        maxHeight="calc(100vh - 140px - 2rem)",
                        locale='en-us',
                        showSorterTooltip=False,
                        rowSelectionType='checkbox',
                        size='small',
                        defaultExpandedRowKeys=['row-0'],
                        expandRowByClick=True,
                        mode='server-side',
                    ),
                    text='Loading data...',
                    size='small',
                )
            ],
            id='results-table-container',
            style={'paddingTop': '1rem'},
        ),
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
                                    help=f"Selected {cpu_count() // 2} / {cpu_count()} cpus"

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
                        fac.AntdProgress(
                            id='processing-progress',
                            percent=0,
                        ),
                        fac.AntdButton(
                            'Cancel',
                            id='cancel-processing',
                            style={
                                'alignText': 'center',
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
                                            tooltip='Enter your username information',
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
                    'title': 'Processing Tour',
                    'description': 'Quick guide to processing results.',
                },
                {
                    'title': 'Run MINT',
                    'description': 'Launch processing for selected files and targets.',
                    'targetSelector': '#processing-btn'
                },
                {
                    'title': 'Options',
                    'description': 'Download results or delete selected/all rows.',
                    'targetSelector': '#processing-options-wrapper'
                },
                {
                    'title': 'Filter & sort',
                    'description': 'Use header controls to search, filter, and sort results.',
                    'targetSelector': '#results-table-container'
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
        ),
        dcc.Store('results-action-store'),
        dcc.Store(id='processing-tour-hint-store', data={'open': True}, storage_type='session'),
        dcc.Download('download-csv'),
    ]
)


def layout():
    return _layout


def callbacks(app, fsc, cache):
    @app.callback(
        Output("results-table", "data"),
        Output("results-table", "selectedRowKeys"),
        Output("results-table", "pagination"),

        Input('section-context', 'data'),
        Input("results-action-store", "data"),
        Input('results-table', 'pagination'),
        Input('results-table', 'filter'),
        Input('results-table', 'sorter'),
        State('results-table', 'filterOptions'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def results_table(section_context, results_actions, pagination, filter_, sorter, filterOptions, wdir):
        if section_context and section_context['page'] != 'Processing':
            raise PreventUpdate

        # Autosave results table on tab load/refresh for durability
        try:
            if wdir:
                results_dir = Path(wdir) / "results"
                results_dir.mkdir(parents=True, exist_ok=True)
                with duckdb_connection(wdir) as conn:
                    conn.execute("COPY (SELECT * FROM results) TO ? (HEADER, DELIMITER ',')",
                                 (str(results_dir / "results_backup.csv"),))
        except Exception:
            pass

        start_time = time.perf_counter()

        f = ['processing-delete-all', 'processing-delete-selected']
        if pagination:
            page_size = pagination['pageSize']
            current = pagination['current']

            with duckdb_connection(wdir) as conn:

                offset = (current - 1) * page_size

                sql, params = build_paginated_query_by_peak(
                    conn=conn,
                    filter_=filter_,
                    filterOptions=filterOptions,
                    sorter=sorter,
                    limit=page_size,
                    offset=offset
                )
                df = conn.execute(sql, params).df()

                # total rows:
                number_records = conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT peak_label FROM "
                                              "results)").fetchone()[0]

            # fix page if it underflowed:
            current = max(current if number_records > (current - 1) * page_size else current - 1, 1)

            data = []
            for r, (peak_label, group_df) in enumerate(df.groupby('peak_label')):
                target = dict(
                    key=f'row-{r}',
                    peak_label=peak_label,
                    children=[],
                )
                for ri, row in enumerate(group_df.itertuples(index=False)):
                    target['children'].append({
                        'key': f'row-{r}{ri}',
                        'ms_file_label': row.ms_file_label,
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
                    })
                data.append(target)

            end_time = time.perf_counter()
            time.sleep(max(0.0, min(len(data), 0.5 - (end_time - start_time))))

            return [
                data,
                [],
                {**pagination, 'total': number_records, 'current': current},
            ]

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
            raise PreventUpdate

        if clickedKey == "processing-delete-selected":
            if not selectedRows:
                raise PreventUpdate

        children = fac.AntdFlex(
            [
                fac.AntdText("This action will delete selected results and cannot be undone?",
                             strong=True),
                fac.AntdText("Are you sure you want to delete the selected results?")
            ],
            vertical=True,
        )
        if clickedKey == "processing-delete-all":
            children = fac.AntdFlex(
                [
                    fac.AntdText("This action will delete ALL results and cannot be undone?",
                                 strong=True, type="danger"),
                    fac.AntdText("Are you sure you want to delete the ALL results?")
                ],
                vertical=True,
            )
        return True, children

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("results-action-store", "data", allow_duplicate=True),

        Input("processing-delete-confirmation-modal", "okCounts"),
        State('results-table', 'selectedRows'),
        State("processing-options", "clickedKey"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def confirm_and_delete(okCounts, selectedRows, clickedKey, wdir):

        if okCounts is None:
            raise PreventUpdate
        if clickedKey == "processing-delete-selected" and not selectedRows:
            results_action_store = {'action': 'delete', 'status': 'failed'}
            total_removed = [0, 0]
        elif clickedKey == "processing-delete-selected":
            remove_results = [row["peak_label"] for row in selectedRows]

            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate

                total_removed_q = conn.execute("""
                                               SELECT COUNT(DISTINCT peak_label)    as unique_peaks,
                                                      COUNT(DISTINCT ms_file_label) as unique_files
                                               FROM results
                                               WHERE peak_label IN ?
                                               """, (remove_results,)).fetchone()
                total_removed = [0, 0]
                if total_removed_q:
                    total_removed = list(total_removed_q)
                    conn.execute("DELETE FROM results WHERE peak_label IN ?", (remove_results,))
                    results_action_store = {'action': 'delete', 'status': 'success'}
        else:
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
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
        return (fac.AntdNotification(message="Delete Results",
                                     description=f"Deleted {total_removed[0]} targets with {total_removed[1]} samples "
                                                 "from results",
                                     type="success" if total_removed != [0, 0] else "error",
                                     duration=3,
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True
                                     ),
                results_action_store)

    @app.callback(
        Output("download-results-modal", "visible"),

        Input("processing-options", "nClicks"),
        State("processing-options", "clickedKey"),
        prevent_initial_call=True,
    )
    def open_download_results(n_clicks, clickedKey):
        if n_clicks is None or clickedKey != 'processing-download':
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
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        ws_key = Path(wdir).stem
        with duckdb_connection_mint(Path(wdir).parent.parent) as mint_conn:
            if mint_conn is None:
                raise PreventUpdate
            ws_name = mint_conn.execute("SELECT name FROM workspaces WHERE key = ?", [ws_key]).fetchone()[0]

        if prop_id == 'download-all-results-btn':
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                cols = ', '.join(d_options_value)
                df = conn.execute(f"""
                    SELECT 
                        r.peak_label, 
                        r.ms_file_label, 
                        s.ms_type,
                        {cols} 
                    FROM results r 
                    JOIN samples s ON s.ms_file_label = r.ms_file_label 
                    ORDER BY s.ms_type, r.peak_label, r.ms_file_label
                """).df()
                filename = f"{ws_name}_all_results.csv"

        else:
            with duckdb_connection(wdir) as conn:
                df = create_pivot(conn, d_dm_rows[0], d_dm_cols[0], d_dm_value[0], table='results')
                filename = f"{ws_name}_{d_dm_value[0]}_results.csv"
        return dcc.send_data_frame(df.to_csv, filename, index=False)

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
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'processing-tour-icon':
            return {'open': False}

        if close_counts:
            return {'open': False}

        return store_data or {'open': True}

    @app.callback(
        Output("processing-modal", "visible"),
        Output("processing-warning", "style"),

        Input("processing-btn", "nClicks"),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def open_run_mint_modal(nClicks, wdir):
        if not nClicks:
            raise PreventUpdate

        computed_results = 0
        # check if some results was computed
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update

            results = conn.execute("SELECT COUNT(*) FROM results").fetchone()
            if results:
                computed_results = results[0]

        style = {'display': 'block'} if computed_results else {'display': 'none'}

        return True, style

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
            Output("processing-progress", "percent"),
        ],
        prevent_initial_call=True
    )
    def compute_results(set_progress, okCounts, recompute, n_cpus, ram, batch_size, bookmarked, wdir):

        print(f"{okCounts = }")

        if not okCounts:
            raise PreventUpdate

        start = time.perf_counter()
        print('Computing chromatograms...')
        compute_chromatograms_in_batches(wdir, use_for_optimization=False, batch_size=batch_size,
                                         set_progress=set_progress, recompute_ms1=False,
                                         recompute_ms2=False, n_cpus=n_cpus, ram=ram, use_bookmarked=bookmarked)
        print('Computing results...')
        compute_results_in_batches(wdir=wdir,
                           use_bookmarked= bookmarked,
                           recompute = recompute,
                           batch_size = batch_size,
                           checkpoint_every = 10,
                           set_progress=set_progress,
                           n_cpus=n_cpus,
                           ram=ram)

        # Persist the results table to a workspace folder for resilience
        try:
            with duckdb_connection(wdir) as conn:
                results_df = conn.execute("SELECT * FROM results").df()
            results_dir = Path(wdir) / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(results_dir / "results_backup.csv", index=False)
        except Exception:
            pass

        print(f"Results computed in {time.perf_counter() - start:.2f} seconds")
        return True, False
