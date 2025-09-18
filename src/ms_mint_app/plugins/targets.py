import logging
import uuid
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_tabulator import DashTabulator

import dash_bootstrap_components as dbc

from .. import tools as T
from ..duckdb_manager import duckdb_connection
from ..plugin_interface import PluginInterface
import dash_uploader as du
import feffery_antd_components as fac

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
        return _outputs
    

from ms_mint.standards import TARGETS_COLUMNS

INFO = dcc.Markdown(
"""
---    

-   **peak_label**: string, Label of the peak (must be unique).
-   **mz_mean**: numeric value. **MS1:** theoretical m/z value of the target ion to extract. **MS2:** m/z value of the 
precursor.
-   **mz_width**: numeric value, width of the peak in \[ppm\]. It is used to calculate the width of the mass window  according to the formula: `Δm = m/z * 1e-6 * mz_width`.
-   **rt**: numeric value, (optional), expected time of the peak. This value is not used during processing, but it can inform the peak optimization procedure.
-   **rt_min**: numeric value, starting time for peak integration.
-   **rt_max**: numeric value, ending time for peak integration.
-   **rt_unit**: one of `s` or `min` for seconds or minutes respectively.
-   **intensity_threshold**: numeric value (>=0), minimum intensity value to include, serves as a noise filter. We recommend setting this to 0. 
-   **target_filename**: string (optional), name of the target list file. It is not used for processing, just to keep track of what files were used.
"""
)

columns = [{"name": i, "id": i, "selectable": True} for i in TARGETS_COLUMNS]

tabulator_options = {
    "groupBy": "ms_type",
    "selectable": True,
    "headerFilterLiveFilterDelay": 3000,
    "layout": "fitDataFill",
    # "height": "900px",
}

downloadButtonType = {
    "css": "btn btn-primary",
    "text": "Export",
    "type": "csv",
    "filename": "Targets",
}

clearFilterButtonType = {"css": "btn btn-outline-dark", "text": "Clear Filters"}

table_columns = [
    {
        "formatter": "rowSelection",
        "titleFormatter": "rowSelection",
        "titleFormatterParams": {"rowRange": "active"},  # only toggle the values of the active filtered rows,
        "hozAlign": "center",
        "headerSort": False,
        "width": "1px",
        "frozen": True,
    },
    {
        "title": "peak_label",
        "field": "peak_label",
        "editor": True,
        "headerTooltip": "This is a tooltip",
        "frozen": True,
    },
    {
        "title": "mz_mean",
        "field": "mz_mean",
        "editor": True,
        "hozAlign": "right",
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "mz_width",
        "field": "mz_width",
        "editor": True,
        "hozAlign": "right",
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "mz",
        "field": "mz",
        "editor": True,
        "hozAlign": "right",
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "rt",
        "field": "rt",
        "editor": True,
        "hozAlign": "right",
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "rt_min",
        "field": "rt_min",
        "editor": True,
        "hozAlign": "right",
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "rt_max",
        "field": "rt_max",
        "editor": True,
        "hozAlign": "right",
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "rt_unit",
        "field": "rt_unit",
        "hozAlign": "center",
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "intensity_threshold",
        "field": "intensity_threshold",
        "editor": True,
        "hozAlign": "right",
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "polarity",
        "field": "polarity",
        "hozAlign": "center",
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "filterLine",
        "field": "filterLine",
        "hozAlign": "center",
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "ms_type",
        "field": "ms_type",
        "hozAlign": "center",
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "category",
        "field": "category",
        "editor": True,
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "preselected_processing",
        "field": "preselected_processing",
        "editor": True,
        "formatter": "tickCross",
        "hozAlign": "center",
        "headerTooltip": "This is a tooltip",
    },
    {
        "title": "source",
        "field": "source",
        "headerTooltip": "This is a tooltip"
    },
]


pkl_table = html.Div(
    id="targets-table-container",
    style={"minHeight": 100, "padding-bottom": "3rem"},
    children=[
        DashTabulator(
            id="targets-table",
            columns=table_columns,
            options=tabulator_options,
            downloadButtonType=downloadButtonType,
            clearFilterButtonType=clearFilterButtonType,
        ),
        dbc.Button("Remove selected targets", id="pkl-clear", style={"float": "right"}, color='danger'),
    ],
)

_layout = html.Div(
    [
        html.Div(
            [
                html.H4(_label),
                fac.AntdIcon(
                    id='targets-tour-icon',
                    icon='pi-info',
                    style={"cursor": "pointer"},
                    )
            ],
            style={"display": "flex", "alignItems": "center", "gap": "10px"}
        ),

        dbc.Row([
            dbc.Col([
                html.Div([html.Label('Upload options'),
                          dcc.Dropdown(
                              id="pkl-ms-mode",
                              options=[{"value": "positive",
                                        "label": "Add proton mass to formula (positive mode)"},
                                       {"value": "negative",
                                        "label": "Subtract proton mass from formula (negative mode)"}],
                          )],
                         ),

            ], width=4),
            dbc.Col([
                du.Upload(
                    id="targets-uploader",
                    max_file_size=1800,  # 1800 MB
                    max_files=1,
                    filetypes=["csv"],
                    upload_id=str(uuid.uuid4()),  # Unique session id
                    text="Upload TARGETS files.",
                ),
            ]),
        ], style={"marginBottom": "30px"}
        ),
        pkl_table,
        INFO,
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


_outputs = html.Div(
    id="pkl-outputs",
    children=[
        dcc.Store(id="uploaded-targets-store"),
        dcc.Store(id="processed-targets-store"),
        dcc.Store(id="removed-targets-store"),
    ],
)


def layout():
    return _layout


def callbacks(app, fsc=None, cache=None):

    @app.callback(
        Output('targets-tour', 'current'),
        Output('targets-tour', 'open'),
        Input('targets-tour-icon', 'nClicks'),
        prevent_initial_call = True,
    )
    def targets_tour(n_clicks):
        print(f"{n_clicks = }")
        return 0, True

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
        State("wdir", "children"),
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
        State("wdir", "children"),
    )
    def targets_table(tab, processed_targets, removed_targets, wdir):

        print(f"{tab = }")

        if tab != "Targets":
            raise PreventUpdate

            targets_df, target_failed, dropped_targets = T.get_targets_from_upload(files, ms_mode)
            T.write_targets(targets_df, wdir)

            notifications = [fac.AntdNotification(message="Load targets.",
                                                  description="Targets file added successfully.", duration=3,
                                                  type="success", placement='bottom', showProgress=True, stack=True)]
            if target_failed:
                notifications.append(fac.AntdNotification(message="Failed to add targets.",
                                                          description='\n, -'.join(target_failed),
                                                          type="error", duration=3, placement='bottom',
                                                          showProgress=True, stack=True))
            if dropped_targets:
                notifications.append(fac.AntdNotification(message="Dropped targets.",
                                                          description=f"Dropped {dropped_targets} targets",
                                                          type="warning", duration=3, placement='bottom',
                                                          showProgress=True, stack=True))
            return notifications, targets_df.to_dict('records')


        targets_df = T.get_targets(wdir)

        if trigger_id == "pkl-ms-mode" and ms_mode:
            targets_df = T.standardize_targets(targets_df, ms_mode)
            T.write_targets(targets_df, wdir)
            return (
                [fac.AntdNotification(message="Changed MS mode.", description=f"MS mode changed to {ms_mode}",
                                      duration=3, placement='bottom', type="success", showProgress=True, stack=True)],
                targets_df.to_dict('records')
            )

        return dash.no_update, targets_df.to_dict('records')

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
        State('wdir', 'children'),
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

        targets_df = T.get_targets(wdir)
        remove_targets = [tr['peak_label'] for tr in selected_rows]
        targets_df = targets_df[~targets_df['peak_label'].isin(remove_targets)]
        T.write_targets(targets_df, wdir)

        return (fac.AntdNotification(message=f"Delete Targets",
                                    description=f"Deleted {len(remove_targets)} targets",
                                    type="success",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    ),
                True)


    @app.callback(
        Output("pkl-table", "downloadButtonType"),
        Input("tab", "value"),
        State("active-workspace", "children"),
    )
    def table_export_fn(tab, ws_name):
        fn = f"{T.today()}-MINT__{ws_name}__targets"
        downloadButtonType = {
            "css": "btn btn-primary",
            "text": "Export",
            "type": "csv",
            "filename": fn,
        }
        return downloadButtonType
