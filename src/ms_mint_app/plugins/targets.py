import logging
import base64

import dash
import feffery_antd_components as fac
import polars as pl
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..duckdb_manager import duckdb_connection, build_where_and_params, build_order_by
from ..plugin_interface import PluginInterface

_label = "Targets"
# Template column headers and descriptions for quick downloads
TARGET_TEMPLATE_COLUMNS = [
    'peak_label', 'mz_mean', 'mz_width', 'mz', 'rt', 'rt_min', 'rt_max', 'rt_unit',
    'intensity_threshold', 'polarity', 'filterLine', 'ms_type', 'category',
    'peak_selection', 'score', 'bookmark', 'source', 'notes'
]
TARGET_TEMPLATE_DESCRIPTIONS = [
    'Unique metabolite/feature name',
    'Mean m/z (centroid)',
    'm/z window or tolerance',
    'Precursor m/z (MS2)',
    'Retention time (default: in seconds)',
    'Lower RT bound (default: in seconds)',
    'Upper RT bound (default: in seconds)',
    'RT unit (e.g. s or min; default: in seconds)',
    'Intensity cutoff (anything lower than this value is considered zero)',
    'Polarity (Positive or Negative)',
    'Filter ID for MS2 scans',
    'ms1 or ms2',
    'Category',
    'True if selected for analysis',
    'Score',
    'True if bookmarked',
    'Data source or file',
    'Free-form notes'
]
TARGET_TEMPLATE_CSV = ",".join(TARGET_TEMPLATE_COLUMNS) + "\n" + ",".join(TARGET_TEMPLATE_DESCRIPTIONS) + "\n"
TARGET_DESCRIPTION_MAP = dict(zip(TARGET_TEMPLATE_COLUMNS, TARGET_TEMPLATE_DESCRIPTIONS))


class TargetsPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 4
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
                            'Targets', level=4, style={'margin': '0'}
                        ),
                        fac.AntdIcon(
                            id='targets-tour-icon',
                            icon='pi-info',
                            style={"cursor": "pointer", 'paddingLeft': '10px'},
                        ),
                        fac.AntdButton(
                            'Load Targets',
                            id={
                                'action': 'file-explorer',
                                'type': 'targets',
                            },
                            style={'textTransform': 'uppercase', "margin": "0 50px"},
                        ),
                    ],
                    align='center',
                ),
                fac.AntdFlex(
                    [
                        fac.AntdButton(
                            'Download template',
                            id='download-target-template-btn',
                            icon=fac.AntdIcon(icon='antd-download'),
                            iconPosition='end',
                            style={'textTransform': 'uppercase'},
                        ),
                        fac.AntdButton(
                            'Download targets',
                            id='download-target-list-btn',
                            icon=fac.AntdIcon(icon='antd-download'),
                            iconPosition='end',
                            style={'textTransform': 'uppercase'},
                        ),
                        html.Div(
                            fac.AntdDropdown(
                                id='targets-options',
                                title='Options',
                                buttonMode=True,
                                arrow=True,
                                menuItems=[
                                    {'title': fac.AntdText('Delete selected', strong=True, type='warning'),
                                     'key': 'delete-selected'},
                                    {'title': fac.AntdText('Clear table', strong=True, type='danger'), 'key': 'delete-all'},
                                ],
                                buttonProps={'style': {'textTransform': 'uppercase'}},
                            ),
                            id='targets-options-wrapper',
                        ),
                    ],
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
                fac.AntdSpin(
                    fac.AntdTable(
                        id='targets-table',
                        containerId='targets-table-container',
                        columns=[
                            {
                                'title': 'Target',
                                'dataIndex': 'peak_label',
                                'width': '260px',
                                'fixed': 'left'
                            },
                            {
                                'title': 'Selection',
                                'dataIndex': 'peak_selection',
                                'width': '120px',
                                'renderOptions': {'renderType': 'switch'},
                            },
                            {
                                'title': 'MZ-Mean',
                                'dataIndex': 'mz_mean',
                                'width': '150px',
                            },
                            {
                                'title': 'MZ-Width',
                                'dataIndex': 'mz_width',
                                'width': '150px',
                            },
                            {
                                'title': 'MZ',
                                'dataIndex': 'mz',
                                'width': '150px',
                            },
                            {
                                'title': 'RT',
                                'dataIndex': 'rt',
                                'width': '150px',
                            },
                            {
                                'title': 'RT-min',
                                'dataIndex': 'rt_min',
                                'width': '150px',
                                'editable': True,
                            },
                            {
                                'title': 'RT-max',
                                'dataIndex': 'rt_max',
                                'width': '150px',
                            },
                            {
                                'title': 'RT-Unit',
                                'dataIndex': 'rt_unit',
                                'width': '120px',
                            },
                            {
                                'title': 'Intensity Threshold',
                                'dataIndex': 'intensity_threshold',
                                'width': '200px',
                                'editable': True,
                            },
                            {
                                'title': 'MS-Type',
                                'dataIndex': 'ms_type',
                                'width': '120px',
                            },
                            {
                                'title': 'Polarity',
                                'dataIndex': 'polarity',
                                'width': '120px',
                            },
                            {
                                'title': 'filterLine',
                                'dataIndex': 'filterLine',
                                'width': '300px',
                            },
                            {
                                'title': 'Proton',
                                'dataIndex': 'proton',
                                'width': '120px',
                            },
                            {
                                'title': 'Category',
                                'dataIndex': 'category',
                                'width': '120px',
                            },
                            {
                                'title': 'Notes',
                                'dataIndex': 'notes',
                                'width': '240px',
                            },
                            {
                                'title': 'Source',
                                'dataIndex': 'source',
                                'width': '400px',
                            },
                        ],
                        titlePopoverInfo={
                            'peak_label': {
                                'title': 'Target',
                                'content': TARGET_DESCRIPTION_MAP['peak_label'],
                            },
                            'peak_selection': {
                                'title': 'Selection',
                                'content': TARGET_DESCRIPTION_MAP['peak_selection'],
                            },
                            'mz_mean': {
                                'title': 'MZ-Mean',
                                'content': TARGET_DESCRIPTION_MAP['mz_mean'],
                            },
                            'mz_width': {
                                'title': 'MZ-Width',
                                'content': TARGET_DESCRIPTION_MAP['mz_width'],
                            },
                            'mz': {
                                'title': 'mz',
                                'content': TARGET_DESCRIPTION_MAP['mz'],
                            },
                            'rt': {
                                'title': 'rt',
                                'content': TARGET_DESCRIPTION_MAP['rt'],
                            },
                            'rt_min': {
                                'title': 'rt_min',
                                'content': TARGET_DESCRIPTION_MAP['rt_min'],
                            },
                            'rt_max': {
                                'title': 'rt_max',
                                'content': TARGET_DESCRIPTION_MAP['rt_max'],
                            },
                            'rt_unit': {
                                'title': 'rt_unit',
                                'content': TARGET_DESCRIPTION_MAP['rt_unit'],
                            },
                            'intensity_threshold': {
                                'title': 'Intensity Threshold',
                                'content': TARGET_DESCRIPTION_MAP['intensity_threshold'],
                            },
                            'ms_type': {
                                'title': 'MS-Type',
                                'content': TARGET_DESCRIPTION_MAP['ms_type'],
                            },
                            'polarity': {
                                'title': 'polarity',
                                'content': TARGET_DESCRIPTION_MAP['polarity'],
                            },
                            'filterLine': {
                                'title': 'filterLine',
                                'content': TARGET_DESCRIPTION_MAP['filterLine'],
                            },
                            'proton': {
                                'title': 'proton',
                                'content': 'Ion charge state (Neutral, Positive, or Negative)',
                            },
                            'category': {
                                'title': 'category',
                                'content': TARGET_DESCRIPTION_MAP['category'],
                            },
                            'notes': {
                                'title': 'notes',
                                'content': TARGET_DESCRIPTION_MAP['notes'],
                            },
                            'source': {
                                'title': 'source',
                                'content': TARGET_DESCRIPTION_MAP['source'],
                            },
                        },
                        filterOptions={
                            'peak_label': {'filterMode': 'keyword'},
                            'mz_mean': {'filterMode': 'keyword'},
                            'mz_width': {'filterMode': 'keyword'},
                            'mz': {'filterMode': 'keyword'},
                            'rt': {'filterMode': 'keyword'},
                            'rt_min': {'filterMode': 'keyword'},
                            'rt_max': {'filterMode': 'keyword'},
                            'rt_unit': {'filterMode': 'checkbox',
                                        'filterCustomItems': ['s', 'min']},
                            'intensity_threshold': {'filterMode': 'keyword'},
                            'ms_type': {'filterMode': 'checkbox',
                                        'filterCustomItems': ['ms1', 'ms2']
                                        },
                            'polarity': {'filterMode': 'checkbox',
                                         'filterCustomItems': ['Negative', 'Positive']
                                         },
                            'filterLine': {'filterMode': 'keyword'},
                            'proton': {'filterMode': 'checkbox',
                                       'filterCustomItems': ['Neutral', 'Positive', 'Negative']},
                            'category': {'filterMode': 'checkbox',
                                         # 'filterCustomItems': ['True', 'False']
                                         },
                            'notes': {'filterMode': 'keyword'},
                            'source': {'filterMode': 'keyword'},

                        },
                        sortOptions={
                            'sortDataIndexes': ['peak_label', 'mz_mean', 'mz_width', 'mz', 'rt', 'rt_min', 'rt_max',
                                                'intensity_threshold', 'category']},
                        pagination={
                            'position': 'bottomCenter',
                            'pageSize': 15,
                            'current': 1,
                            'showSizeChanger': True,
                            'pageSizeOptions': [5, 10, 15, 25, 50, 100],
                            'showQuickJumper': True,
                        },
                        tableLayout='fixed',
                        maxWidth="calc(100vw - 250px - 4rem)",
                        maxHeight="calc(100vh - 140px - 2rem)",
                        locale='en-us',
                        rowSelectionType='checkbox',
                        size='small',
                        mode='server-side',
                    ),
                    text='Loading data...',
                    size='small',
                )
            ],
            id='targets-table-container',
            style={'paddingTop': '1rem'},
        ),

        fac.AntdModal(
            "Are you sure you want to delete the selected targets?",
            title="Delete target",
            id="delete-table-targets-modal",
            renderFooter=True,
            okButtonProps={"danger": True},
            locale='en-us',
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Targets Tour',
                    'description': 'This is a tour of the targets plugin.',
                },
                {
                    'title': 'Download template',
                    'description': 'Download the targets template CSV to start with the right columns and descriptions.',
                    'targetSelector': '#download-target-template-btn'
                },
                {
                    'title': 'Download targets',
                    'description': 'Export the current targets table (with filters/sort applied on server) for review or sharing.',
                    'targetSelector': '#download-target-list-btn'
                },
                {
                    'title': 'Options',
                    'description': 'Use these options to delete selected rows and clear the table.',
                    'targetSelector': '#targets-options-wrapper'
                },
                {
                    'title': 'Filter & sort',
                    'description': 'Use header controls to search, filter by value, and sort columns.',
                    'targetSelector': '#targets-table-container'
                },
            ],
            id='targets-tour',
            open=False,
            current=0,
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Need help?',
                    'description': 'Click the info icon to open a quick tour of the Targets table.',
                    'targetSelector': '#targets-tour-icon',
                },
            ],
            mask=False,
            placement='rightTop',
            open=False,
            current=0,
            id='targets-tour-hint',
            className='targets-tour-hint',
            style={
                'background': '#ffffff',
                'border': '0.5px solid #1677ff',
                'boxShadow': '0 6px 16px rgba(0,0,0,0.15), 0 0 0 1px rgba(22,119,255,0.2)',
                'opacity': 1,
            },
        ),
        dcc.Store(id="targets-action-store"),
        dcc.Store(id="targets-tour-hint-store", data={'open': True}, storage_type='session'),
        dcc.Download('download-targets-csv'),
    ],
)


def layout():
    return _layout


def callbacks(app, fsc=None, cache=None):
    @app.callback(
        Output("targets-table", "data"),
        Output("targets-table", "selectedRowKeys"),
        Output("targets-table", "pagination"),
        Output("targets-table", "filterOptions"),

        Input('section-context', 'data'),
        Input("targets-action-store", "data"),
        Input("processed-action-store", "data"),  # from explorer
        Input('targets-table', 'pagination'),
        Input('targets-table', 'filter'),
        Input('targets-table', 'sorter'),
        State('targets-table', 'filterOptions'),
        State("processing-type-store", "data"),  # from explorer
        State("wdir", "data"),
    )
    def targets_table(section_context, processing_output, processed_action, pagination, filter_, sorter, filterOptions,
                      processing_type, wdir):
        # processing_type also store info about ms-files and metadata selection since it is the same modal for all of
        # them
        if section_context and section_context['page'] != 'Targets':
            raise PreventUpdate

        if pagination:
            page_size = pagination['pageSize']
            current = pagination['current']

            with duckdb_connection(wdir) as conn:
                schema = conn.execute("DESCRIBE targets").pl()
            column_types = {r["column_name"]: r["column_type"] for r in schema.to_dicts()}
            where_sql, params = build_where_and_params(filter_, filterOptions)
            order_by_sql = build_order_by(sorter, column_types, tie=('peak_label', 'ASC'))

            sql = f"""
                        WITH filtered AS (
                          SELECT *
                          FROM targets
                          {where_sql}
                        ),
                        paged AS (
                          SELECT *, COUNT(*) OVER() AS __total__
                          FROM filtered
                          {(' ' + order_by_sql) if order_by_sql else ''}
                          LIMIT ? OFFSET ?
                        )
                        SELECT * FROM paged;
                        """

            params_paged = params + [page_size, (current - 1) * page_size]

            with duckdb_connection(wdir) as conn:
                dfpl = conn.execute(sql, params_paged).pl()

            data = dfpl.with_columns(
                pl.col('peak_selection').map_elements(
                    lambda value: {'checked': value},
                    return_dtype=pl.Object
                ).alias('peak_selection'),
            )

            # total rows:
            number_records = int(data["__total__"][0]) if len(data) else 0

            # fix page if it underflowed:
            current = max(current if number_records > (current - 1) * page_size else current - 1, 1)

            with (duckdb_connection(wdir) as conn):
                st_custom_items = filterOptions['category'].get('filterCustomItems')
                category_filters = conn.execute("SELECT DISTINCT category "
                                                "FROM targets "
                                                "ORDER BY category ASC").df()['category'].to_list()
                if st_custom_items != category_filters:
                    output_filterOptions = filterOptions.copy()
                    output_filterOptions['category']['filterCustomItems'] = (category_filters
                                                                             if category_filters != [None] else
                                                                             [])
                else:
                    output_filterOptions = dash.no_update

            return [
                data.to_dicts(),
                [],
                {**pagination, 'total': number_records, 'current': current, 'pageSizeOptions': sorted([5, 10, 15, 25, 50,
                100, number_records])},
                output_filterOptions
            ]
        return dash.no_update

    @app.callback(
        Output('delete-table-targets-modal', 'visible'),
        Output('delete-table-targets-modal', 'children'),
        Input("targets-options", "nClicks"),
        State("targets-options", "clickedKey"),
        State('targets-table', 'selectedRows'),
        prevent_initial_call=True
    )
    def show_delete_modal(nClicks, clickedKey, selectedRows):
        ctx = dash.callback_context
        if not ctx.triggered or clickedKey not in ['delete-selected', 'delete-all']:
            raise PreventUpdate

        if clickedKey == "delete-selected":
            if not bool(selectedRows):
                raise PreventUpdate

        children = fac.AntdFlex(
            [
                fac.AntdText("This action will delete selected targets and cannot be undone?",
                             strong=True),
                fac.AntdText("Are you sure you want to delete the selected targets?")
            ],
            vertical=True,
        )
        if clickedKey == "delete-all":
            children = fac.AntdFlex(
                [
                    fac.AntdText("This action will delete ALL targets and cannot be undone?",
                                 strong=True, type="danger"),
                    fac.AntdText("Are you sure you want to delete the ALL targets?")
                ],
                vertical=True,
            )
        return True, children

    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('targets-action-store', "data", allow_duplicate=True),

        Input('delete-table-targets-modal', 'okCounts'),
        State('targets-table', 'selectedRows'),
        State("targets-options", "clickedKey"),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def target_delete(okCounts, selectedRows, clickedKey, wdir):
        """
        Delete targets from the table.

        Triggered by the delete button in the target table, this function will delete the selected targets from the
        table and write the updated table to the targets file.

        Parameters
        ----------
        okCounts : int
            The number of times the ok button was clicked.
        cancelCounts : int
            The number of times the cancel button was clicked.
        closeCounts : int
            The number of times the close button was clicked.
        selected_rows : list
            A list of dictionaries, where each dictionary represents a selected row in the table.
        wdir : str
            The working directory.

        Returns
        -------
        notifications : list
            A list of notifications to be displayed in the notification container.
        drop_table_output : boolean
            A boolean indicating whether the delete button was clicked.
        """
        if okCounts is None or clickedKey not in ['delete-selected', 'delete-all']:
            raise PreventUpdate
        if clickedKey == "delete-selected" and not selectedRows:
            targets_action_store = {'action': 'delete', 'status': 'failed'}
            total_removed = 0
        elif clickedKey == "delete-selected":
            remove_targets = [row['peak_label'] for row in selectedRows]

            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                conn.execute("DELETE FROM targets WHERE peak_label IN ?", (remove_targets,))
                conn.execute("DELETE FROM chromatograms WHERE peak_label IN ?", (remove_targets,))
                # conn.execute("DELETE FROM results WHERE peak_label IN ?", (remove_targets,))
            total_removed = len(remove_targets)
            targets_action_store = {'action': 'delete', 'status': 'success'}
        elif clickedKey == "delete-all":
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                total_removed_q = conn.execute("SELECT COUNT(*) FROM targets").fetchone()
                targets_action_store = {'action': 'delete', 'status': 'failed'}
                total_removed = 0
                if total_removed_q:
                    total_removed = total_removed_q[0]

                    conn.execute("DELETE FROM targets")
                    conn.execute("DELETE FROM chromatograms")
                    # conn.execute("DELETE FROM results")
                    targets_action_store = {'action': 'delete', 'status': 'success'}
        return (fac.AntdNotification(message="Delete Targets",
                                     description=f"Deleted {total_removed} targets",
                                     type="success" if total_removed > 0 else "error",
                                     duration=3,
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True
                                     ),
                targets_action_store)

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("targets-action-store", "data", allow_duplicate=True),

        Input("targets-table", "recentlyChangedRow"),
        State("targets-table", "recentlyChangedColumn"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def save_target_table_on_edit(row_edited, column_edited, wdir):
        """
        This callback saves the table on cell edits.
        This saves some bandwidth.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        if row_edited is None or column_edited is None:
            raise PreventUpdate
        try:
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate

                query = f"UPDATE targets SET {column_edited} = ? WHERE peak_label = ?"
                if column_edited == 'sample_type' and row_edited[column_edited] in [None, '']:
                    conn.execute(query, ['Unset', row_edited['peak_label']])
                else:
                    conn.execute(query, [row_edited[column_edited], row_edited['peak_label']])
                targets_action_store = {'action': 'delete', 'status': 'success'}
            return fac.AntdNotification(message="Successfully edition saved",
                                        type="success",
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        stack=True
                                        ), targets_action_store
        except Exception as e:
            logging.error(f"Error updating metadata: {e}")
            targets_action_store = {'action': 'delete', 'status': 'failed'}
            return fac.AntdNotification(message="Failed to save edition",
                                        description=f"Failing to save edition with: {str(e)}",
                                        type="error",
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        stack=True
                                        ), targets_action_store

    @app.callback(
        Input('targets-table', 'recentlySwitchDataIndex'),
        Input('targets-table', 'recentlySwitchStatus'),
        Input('targets-table', 'recentlySwitchRow'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def save_switch_changes(recentlySwitchDataIndex, recentlySwitchStatus, recentlySwitchRow, wdir):

        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            conn.execute(f"UPDATE targets SET {recentlySwitchDataIndex} = ? WHERE peak_label = ?",
                         (recentlySwitchStatus, recentlySwitchRow['peak_label']))

    @app.callback(
        Output("download-targets-csv", "data"),

        Input("targets-options", "nClicks"),
        Input("download-target-template-btn", "nClicks"),
        Input("download-target-list-btn", "nClicks"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def download_results(options_clicks, template_clicks, list_clicks, wdir):

        from pathlib import Path
        from ..duckdb_manager import duckdb_connection_mint

        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger == 'download-target-template-btn':
            return dcc.send_string(TARGET_TEMPLATE_CSV, "targets_template.csv")

        if trigger == 'download-target-list-btn':
            ws_key = Path(wdir).stem
            with duckdb_connection_mint(Path(wdir).parent.parent) as mint_conn:
                if mint_conn is None:
                    raise PreventUpdate
                ws_name = mint_conn.execute("SELECT name FROM workspaces WHERE key = ?", [ws_key]).fetchone()[0]

            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                df = conn.execute("SELECT * FROM targets").df()
                filename = f"{ws_name}_targets.csv"
        else:
            raise PreventUpdate
        return dcc.send_data_frame(df.to_csv, filename, index=False)

    @app.callback(
        Output('targets-tour', 'current'),
        Output('targets-tour', 'open'),
        Input('targets-tour-icon', 'nClicks'),
        prevent_initial_call=True,
    )
    def targets_tour(n_clicks):
        print(f"{n_clicks = }")
        return 0, True

    @app.callback(
        Output('targets-tour-hint', 'open'),
        Output('targets-tour-hint', 'current'),
        Input('targets-tour-hint-store', 'data'),
    )
    def sync_hint_store(store_data):
        if not store_data:
            raise PreventUpdate
        return store_data.get('open', True), 0

    @app.callback(
        Output('targets-tour-hint-store', 'data'),
        Input('targets-tour-hint', 'closeCounts'),
        Input('targets-tour-icon', 'nClicks'),
        State('targets-tour-hint-store', 'data'),
        prevent_initial_call=True,
    )
    def hide_hint(close_counts, n_clicks, store_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'targets-tour-icon':
            return {'open': False}

        if close_counts:
            return {'open': False}

        return store_data or {'open': True}
