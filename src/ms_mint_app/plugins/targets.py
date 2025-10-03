import logging

import dash
import feffery_antd_components as fac
import polars as pl
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..duckdb_manager import duckdb_connection
from ..plugin_interface import PluginInterface

_label = "Targets"

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
                fac.AntdDropdown(
                    id='targets-options',
                    title='Options',
                    buttonMode=True,
                    arrow=True,
                    menuItems=[
                        {'title': 'Regenerate colors', 'icon': 'pi-broom', 'key': 'regenerate-colors'},
                        {'isDivider': True},
                        {'title': fac.AntdText('Delete selected', strong=True, type='warning'),
                         'key': 'delete-selected'},
                        {'title': fac.AntdText('Clear table', strong=True, type='danger'), 'key': 'delete-all'},
                    ],
                    buttonProps={'style': {'textTransform': 'uppercase'}},
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
                                'title': 'Polarity',
                                'dataIndex': 'polarity',
                                'width': '120px',
                            },
                            {
                                'title': 'filterLine',
                                'dataIndex': 'filterLine',
                                'width': '120px',
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
                                'title': 'Selected',
                                'dataIndex': 'preselected_processing',
                                'width': '120px',
                                'renderOptions': {'renderType': 'switch'},
                            },
                            {
                                'title': 'Source',
                                'dataIndex': 'source',
                                'width': '400px',
                            },
                        ],
                        titlePopoverInfo={
                            'peak_label': {
                                'title': 'peak_label',
                                'content': 'This is peak_label field',
                            },
                            'mz_mean': {
                                'title': 'mz_mean',
                                'content': 'This is mz_mean field',
                            },
                            'mz_width': {
                                'title': 'mz_width',
                                'content': 'This is mz_width field',
                            },
                            'mz': {
                                'title': 'mz',
                                'content': 'This is mz field',
                            },
                            'rt': {
                                'title': 'rt',
                                'content': 'This is rt field',
                            },
                            'rt_min': {
                                'title': 'rt_min',
                                'content': 'This is rt_min field',
                            },
                            'rt_max': {
                                'title': 'rt_max',
                                'content': 'This is rt_max field',
                            },
                            'rt_unit': {
                                'title': 'rt_unit',
                                'content': 'This is rt_unit field',
                            },
                            'intensity_threshold': {
                                'title': 'intensity_threshold',
                                'content': 'This is intensity_threshold field',
                            },
                            'polarity': {
                                'title': 'polarity',
                                'content': 'This is polarity field',
                            },
                            'filterLine': {
                                'title': 'filterLine',
                                'content': 'This is filterLine field',
                            },
                            'proton': {
                                'title': 'proton',
                                'content': 'This is proton field',
                            },
                            'category': {
                                'title': 'category',
                                'content': 'This is category field',
                            },
                            'preselected_processing': {
                                'title': 'preselected_processing',
                                'content': 'This is preselected_processing field',
                            },
                            'source': {
                                'title': 'source',
                                'content': 'This is source field',
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
                            'polarity': {'filterMode': 'checkbox',
                                         'filterCustomItems': ['Negative', 'Positive']
                                         },
                            'preselected_processing': {'filterMode': 'checkbox',
                                                       'filterCustomItems': ['True', 'False']},
                            'filterLine': {'filterMode': 'keyword'},
                            'proton': {'filterMode': 'checkbox',
                                       'filterCustomItems': ['Neutral', 'Positive', 'Negative']},
                            'category': {'filterMode': 'checkbox',
                                         # 'filterCustomItems': ['True', 'False']
                                         },
                            'source': {'filterMode': 'keyword'},

                        },
                        sortOptions={
                            'sortDataIndexes': ['peak_label', 'mz_mean', 'mz_width', 'mz', 'rt', 'rt_min', 'rt_max',
                                                'rt_unit', 'intensity_threshold', 'polarity', 'filterLine', 'proton',
                                                'category', 'preselected_processing', 'source']},
                        pagination={
                            'position': 'bottomCenter',
                            'pageSize': 10,
                            'current': 1,
                            'showSizeChanger': True,
                            'pageSizeOptions': [5, 10, 25, 50, 100],
                            'showQuickJumper': True,
                        },
                        tableLayout='fixed',
                        maxWidth="calc(100vw - 250px - 4rem)",
                        maxHeight="calc(100vh - 140px - 4rem)",
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
            style={'padding': '1rem 0'},
        ),

        fac.AntdModal(
            "Are you sure you want to delete the selected targets?",
            id="delete-table-targets-modal",
            title="Delete target",
            visible=False,
            closable=False,
            width=400,
            renderFooter=True,
            okText="Delete",
            okButtonProps={"danger": True},
            cancelText="Cancel"
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Targets Tour',
                    'description': 'This is a tour of the targets plugin.',
                },
                {
                    'title': 'Table Columns info',
                    'description': 'To access the information related to each column, I can position the mouse over '
                                   'the column and an information box will be displayed with the most important '
                                   'aspects of the column, such as description, definition, types, etc.',
                    # 'targetId': 'upload-btn-tour-demo-1',
                    'targetSelector': "#targets-table-container div.tabulator-col:nth-of-type(2)"
                },
            ],
            id='targets-tour',
        )
    ],
    style={"padding": "3rem"}
)

        dcc.Store(id="targets-action-store"),
    ],
)


def layout():
    return _layout


def callbacks(app, fsc=None, cache=None):
    @du.callback(
        output=Output("uploaded-targets-store", "data"),
        id="targets-uploader",
    )
    def targets_upload_completed(status):
        logging.warning(f"Upload status: {status} ({type(status)})")
        return [status.latest_file.as_posix(), status.n_uploaded, status.n_total]

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("processed-targets-store", "data"),

        Input("uploaded-targets-store", "data"),
        Input("pkl-ms-mode", "value"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def process_targets_files(uploaded_data, ms_mode, wdir):

        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        latest_file, n_uploaded, n_total = uploaded_data if uploaded_data is not None else (None, 0, 0)

        # at the moment, only the uploaded targets are processed
        if trigger_id == "pkl-ms-mode":
            raise PreventUpdate

        if not latest_file or n_uploaded == 0:
            raise PreventUpdate

        targets_df, failed = T.get_targets_from_upload(latest_file, ms_mode)

        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate

            if targets_df.empty:
                return (fac.AntdNotification(message="Failed to add targets.",
                                             description="No targets found in the uploaded file.",
                                             type="error", duration=3, placement='bottom', showProgress=True,
                                             stack=True),
                        dash.no_update)
            n_registered_before = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
            conn.execute("INSERT INTO targets SELECT * FROM targets_df ON CONFLICT DO NOTHING;")
            n_registered_after = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]

            failed_targets = n_registered_after - n_registered_before - len(targets_df)
            new_targets = len(targets_df) - failed_targets

            if failed_targets != 0 and new_targets != 0:
                notification = fac.AntdNotification(message="Targets added.",
                                     description=f"{new_targets} targets added successfully, "
                                                 f"but {failed_targets} targets failed.",
                                     type="warning", duration=3, placement='bottom', showProgress=True,
                                     stack=True)
            elif failed_targets == 0:
                notification = fac.AntdNotification(message="Targets added successfully.",
                                                    description=f"{new_targets} targets added successfully.",
                                                    type="success", duration=3, placement='bottom', showProgress=True,
                                                    stack=True
                                                    )
            else:
                notification = fac.AntdNotification(message="Targets added failed.",
                                         description=f"{failed_targets} targets failed to add.",
                                         type="error", duration=3, placement='bottom', showProgress=True,
                                         stack=True)
            return notification, True

    @app.callback(
        Output("targets-table", "data"),
        Input("tab", "value"),
        Input("processed-targets-store", "data"),
        Input("removed-targets-store", "data"),
        State("wdir", "data"),
    )
    def targets_table(tab, processed_targets, removed_targets, wdir):

        print(f"{tab = }")

        if tab != "Targets":
            raise PreventUpdate

        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            targets_df = conn.execute("SELECT * FROM targets").df()
            print(f"{targets_df.head() = }")
        return targets_df.to_dict('records')

    @app.callback(
        Output('delete-table-targets-modal', 'visible'),
        Input("pkl-clear", 'n_clicks'),
        State('targets-table', 'multiRowsClicked'),
        prevent_initial_call=True
    )
    def show_delete_modal(delete_clicks, selected_rows):
        if not selected_rows:
            raise PreventUpdate
        return True

    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('removed-targets-store', "data"),

        Input('delete-table-targets-modal', 'okCounts'),
        Input('delete-table-targets-modal', 'cancelCounts'),
        Input('delete-table-targets-modal', 'closeCounts'),
        State('targets-table', 'multiRowsClicked'),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def target_delete(okCounts, cancelCounts, closeCounts, selected_rows, wdir):
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
        if not okCounts or cancelCounts or closeCounts or not selected_rows:
            raise PreventUpdate

        remove_targets = [tr['peak_label'] for tr in selected_rows]

        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            conn.execute("DELETE FROM targets WHERE peak_label IN ?", (remove_targets,))
            # conn.execute("DELETE FROM results WHERE peak_label IN ?", (remove_targets,))

        return (fac.AntdNotification(message="Delete Targets",
                                    description=f"Deleted {len(remove_targets)} targets",
                                    type="success",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    ),
                True)

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("targets-table", "cellEdited"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def save_target_table_on_edit(cell_edited, wdir):
        """
        This callback saves the table on cell edits.
        This saves some bandwidth.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        if cell_edited is None:
            raise PreventUpdate

        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            _column = cell_edited['column']
            _value = cell_edited['value']
            _peak_label = cell_edited['row']['peak_label']
            query = f"UPDATE targets SET {_column} = ? WHERE peak_label = ?"
            conn.execute(query, [_value, _peak_label])
            if _column == 'preselected_processing':
                conn.execute("UPDATE targets SET bookmark = ? WHERE peak_label = ?", [False, _peak_label])

        return fac.AntdNotification(message="Successfully saved target data.",
                                    type="success",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    )

    # @app.callback(
    #     Output("targets-table", "downloadButtonType"),
    #     Input("tab", "value"),
    #     State("active-workspace", "children"),
    # )
    # def table_export_fn(tab, ws_name):
    #     fn = f"{T.today()}-MINT__{ws_name}__targets"
    #     downloadButtonType = {
    #         "css": "btn btn-primary",
    #         "text": "Export",
    #         "type": "csv",
    #         "filename": fn,
    #     }
    #     return downloadButtonType

    @app.callback(
        Output('targets-tour', 'current'),
        Output('targets-tour', 'open'),
        Input('targets-tour-icon', 'nClicks'),
        prevent_initial_call=True,
    )
    def targets_tour(n_clicks):
        print(f"{n_clicks = }")
        return 0, True
