import os
import uuid
import logging

import dash
import tempfile

from pathlib import Path as P, Path

import pandas as pd

from dash import html, dcc, set_props, Patch
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc

from ms_mint.io import convert_mzxml_to_parquet

from dash_tabulator import DashTabulator

import dash_uploader as du

from ..colors import make_palette, make_palette_hsv
from ..plugin_interface import PluginInterface
import feffery_antd_components as fac

from ..duckdb_manager import duckdb_connection
import concurrent.futures

_label = "MS-Files"


class MsFilesPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 2
        print(f"Initiated {_label} plugin")

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(self, app, fsc, cache)

    def outputs(self):
        return _outputs


upload_root = os.getenv("MINT_DATA_DIR", tempfile.gettempdir())
upload_dir = str(P(upload_root) / "MINT-Uploads")
UPLOAD_FOLDER_ROOT = upload_dir

options = {
    "groupBy": ["file_type"],
    "selectable": True,
    "headerFilterLiveFilterDelay": 3000,
    "layout": "fitColumns",
    # "layout":"fitDataStretch",
    "height": "calc(100vh - 22rem - 86px)",
    "reactiveData": True,  # Solo actualiza lo necesario
    # "dataChangedTest": True,
    # "pagination": "local",
    # "paginationSize": 10,
    # "movableColumns": True,
    # "resizableColumns": True,

}

clearFilterButtonType = {"css": "btn btn-outline-dark", "text": "Clear Filters"}

columns = [
    {
        "formatter": "rowSelection",
        "titleFormatter": "rowSelection",
        "titleFormatterParams": {
            "rowRange": "active"  # only toggle the values of the active filtered rows
        },
        "hozAlign": "center",
        "headerSort": False,
        "frozen": True,
        "width": 20,
    },
    {
        "title": "MS-Files",
        "field": "ms_file_label",
        "headerFilter": True,
        "headerSort": True,
        "editor": None,
        # "width": "80%",
        # "sorter": "string",
        "frozen": True,
        "widthGrow": 3,
        "headerTooltip": "This is a tooltip"
    },
    {
        "title": "Color",
        "field": "color",
        "headerFilter": False,
        "editor": None,
        "formatter": "color",
        "width": 50,
        "headerSort": False,
    },
    {
        "title": "Use for Optimization",
        "field": "use_for_optimization",
        "headerFilter": True,
        "formatter": "tickCross",
        "width": 120,
        "headerSort": True,
        "hozAlign": "center",
        "editor": "tickCross",
        "widthGrow": 2,
    },
    {
        "title": "In Analysis",
        "field": "use_for_analysis",
        "headerFilter": True,
        "formatter": "tickCross",
        # "width": "6px",
        "headerSort": True,
        "hozAlign": "center",
        "editor": True,
        "headerTooltip": "This is a tooltip",
    },
    {
        "title": "Label",
        "field": "label",
        "headerFilter": True,
        "headerSort": True,
        "hozAlign": "center",
        "editor": True,
        "headerTooltip": "This is a tooltip",
    },
    {
        "title": "Sample Type",
        "field": "sample_type",
        "headerFilter": True,
        "headerSort": True,
        "hozAlign": "center",
        "editor": True,
        "headerTooltip": "This is a tooltip",
        "widthGrow": 2
    },
    {
        "title": "Run Order",
        "field": "run_order",
        "headerFilter": True,
        "headerSort": True,
        "hozAlign": "center",
        "editor": True,
        "headerTooltip": "This is a tooltip",
    },
    {
        "title": "Plate",
        "field": "plate",
        "headerFilter": True,
        "headerSort": True,
        "hozAlign": "center",
        "editor": True,
        "headerTooltip": "This is a tooltip",
    },
    {
        "title": "Plate Row",
        "field": "plate_row",
        "headerFilter": True,
        "headerSort": True,
        "hozAlign": "center",
        "editor": True,
        "headerTooltip": "This is a tooltip",
    },
    {
        "title": "Plate Column",
        "field": "plate_column",
        "headerFilter": True,
        "headerSort": True,
        "hozAlign": "center",
        "editor": True,
        "headerTooltip": "This is a tooltip",
    },
]

ms_files_table = html.Div(
    id="ms-files-table-container",
    style={"padding": "3rem 0"},
    children=[
        DashTabulator(
            id="ms-files-table",
            columns=columns,
            options=options,
        ),
        dbc.Button("Delete selected file", id="ms-delete", color="danger",
                   style={"text-align": "right"}, className="float-end"),
    ],
)

modal_confirmation = dbc.Modal(
    [
        dbc.ModalHeader("Delete confirmation"),
        dbc.ModalBody("Are you sure you want to delete the selected files?"),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="ms-mc-cancel", className="ml-auto"),
            dbc.Button("Delete", id="ms-mc-confirm", color="danger"),
        ]),
    ],
    id="modal-confirmation",
    is_open=False,
)

_layout = html.Div(
    [
        html.H4("Upload Mass Spec / Metadata files"),
        dbc.Row([
            dbc.Col(
                du.Upload(
                    id="ms-uploader",
                    max_file_size=1800,  # 1800
                    max_files=10000,
                    filetypes=["tar", "zip", "mzxml", "mzml", "mzXML", "mzML", "mzMLb", "feather", "parquet"],
                    upload_id=str(uuid.uuid4()),  # Unique session id
                    text="Upload mzXML/mzML files.",
                    text_completed="Upload completed.",
                ),
            ),
            dbc.Col(
                du.Upload(
                    id="metadata-uploader",
                    max_file_size=1800,  # 1800 MB
                    max_files=1,
                    filetypes=["csv"],
                    upload_id=str(uuid.uuid4()),  # Unique session id
                    text="Upload METADATA files.",
                    text_completed="Upload completed.",
                ),
            )]
        ),
        dcc.Store(id="ms-processed-output"),
        dcc.Store(id="metadata-uploader-input"),
        dcc.Store(id="metadata-processed-store"),
        html.Div(
            id="progress-container",
            style={"display": "none"},
            children=[
                html.P("Processing files..."),
                fac.AntdProgress(
                    id="ms-progress-bar",
                    showInfo=True,

                )
            ]
        ),
        dcc.Interval(id="ms-poll-interval", interval=1000, n_intervals=0, disabled=True),
        modal_confirmation,
        dcc.Store(id="ms-delete-store"),
        dcc.Loading(ms_files_table),
        dcc.Store(id="ms-uploader-input")
    ],
    style={"padding": "3rem"}
)

_outputs = html.Div(
    id="ms-outputs",
    children=[
        html.Div(id={"index": "ms-delete-output", "type": "output"}),
        html.Div(id={"index": "ms-save-output", "type": "output"}),
        html.Div(id={"index": "ms-import-from-url-output", "type": "output"}),
        html.Div(id={"index": "ms-new-target-output", "type": "output"}),
    ],
)


def layout():
    return _layout


def callbacks(cls, app, fsc, cache):
        State("wdir", "data"),
    @app.callback(
        Output("ms-files-table", "data"),
        Input("ms-processed-output", "data"),
        Input("metadata-processed-store", "data"),
        Input("ms-delete-store", "data"),
        Input("wdir", "children"),
        Input("wdir", "data"),
    )
    def ms_files_table(value, value2, files_deleted, wdir):

        if wdir is None:
            raise PreventUpdate

        with duckdb_connection(wdir) as conn:
            if conn is None:
                return pd.DataFrame().to_dict('records')
            data = conn.execute("SELECT * FROM samples_metadata").df()
        return data.to_dict("records")

    @app.callback(
        Output("modal-confirmation", "is_open"),
        Input("ms-delete", "n_clicks"),
        Input("ms-mc-cancel", "n_clicks"),
        Input("ms-mc-confirm", "n_clicks"),
        State("modal-confirmation", "is_open"),
        State("ms-files-table", "multiRowsClicked"),
    )
    def toggle_modal(n_delete, n_cancel, n_confirm, is_open, rows):
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open
        if not rows:
            return False

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == "ms-delete":
            return True  # open modal
        elif trigger_id in ["ms-mc-cancel", "ms-mc-confirm"]:
            return False  # Close modal

        return is_open

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("ms-delete-store", "data"),
        Input("ms-mc-confirm", "n_clicks"),
        State("ms-files-table", "multiRowsClicked"),
        State("wdir", "children"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def confirm_and_delete(n_confirm, rows, wdir):
        if n_confirm is None or not rows:
            raise PreventUpdate

        remove_ms_file = [row["ms_file_label"] for row in rows]

        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            #TODO: remove data from ms_data, and results as well
            conn.execute("DELETE FROM samples_metadata WHERE ms_file_label IN ?", (remove_ms_file,))
            conn.execute("DELETE FROM ms_data WHERE ms_file_label IN ?", (remove_ms_file,))
            # conn.execute("DELETE FROM results WHERE ms_file_label = ?", (filename,))

        return (fac.AntdNotification(message="Delete files",
                                     description=f"{len(rows)} files deleted successful",
                                     type="success",
                                     duration=3,
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True
                                     ),
                len(rows))

    @du.callback(
        output=Output("ms-uploader-input", "data"),
        id="ms-uploader",
    )
    def ms_upload_completed(status):
        logging.warning(f"Upload status: {status} ({type(status)})")
        if status.n_uploaded == status.n_total:
            set_props("ms-progress-bar", {"percent": 0})
            set_props("progress-container", {"style": {"display": "block"}})
            return [[f.as_posix() for f in status.uploaded_files], status.n_total]
        raise PreventUpdate

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("ms-files-table", "cellEdited"),
        State("wdir", "children"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def save_table_on_edit(cell_edited, wdir):
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
            _ms_file_label = cell_edited['row']['ms_file_label']
            query = f"UPDATE samples_metadata SET {_column} = ? WHERE ms_file_label = ?"
            conn.execute(query, [_value, _ms_file_label])

        return fac.AntdNotification(message="Successfully saved metadata.",
                                    type="success",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    )

    @app.callback(
        Output('notifications-container', "children", allow_duplicate=True),
        Output('ms-processed-output', 'data'),
        Output("progress-container", "style", allow_duplicate=True),

    Input('ms-uploader-input', 'data'),
        State("wdir", "children"),
        State("wdir", "data"),
        background=True,
        running=[
            (Output("metadata-uploader", "disabled"), True, False),
        ],
        progress=[Output("ms-progress-bar", "percent")],
        prevent_initial_call=True
    )
    def background_ms_processing(set_progress, uploader_data, wdir):

        prop_id = dash.callback_context.triggered[0]['prop_id']
        print(f'background_ms_processing {prop_id = }')

        if not dash.callback_context.triggered or uploader_data is None:
            raise PreventUpdate

        uploaded_files, n_total = uploader_data
        duplicated_files = []

        # get the ms_file_label data as df to avoid multiple queries
        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            data = conn.execute("SELECT ms_file_label FROM samples_metadata").df()

        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            for i, file_path in enumerate(uploaded_files):
                if Path(file_path).stem in data['ms_file_label'].values:
                    duplicated_files.append(file_path)
                    set_progress(round(len(duplicated_files) / n_total * 100, 2))
                else:
                    futures.append(executor.submit(convert_mzxml_to_parquet, file_path))

            batch_ms = []
            batch_ms_data = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                _ms_file_label, _ms_level, _polarity, _parquet_df = future.result()
                batch_ms.append((_ms_file_label, _ms_file_label, _ms_level, f'ms{_ms_level}', _polarity))
                batch_ms_data.append(_parquet_df)

                if len(batch_ms) == 20:
                    with duckdb_connection(wdir) as conn:
                        if conn is None:
                            raise PreventUpdate
                        try:
                            conn.executemany(
                                "INSERT INTO samples_metadata(ms_file_label, label, ms_level, file_type, polarity) "
                                "VALUES (?, ?, ?, ?, ?)",
                                batch_ms
                            )
                            conn.execute(
                                "INSERT INTO ms_data SELECT * FROM read_parquet(?)", [batch_ms_data])
                        except Exception as e:
                            logging.error(f"DB error: {e}")
                    batch_ms = []
                    batch_ms_data = []
                    set_progress(round(i + len(duplicated_files) / n_total * 100, 2))

            if batch_ms:
                with duckdb_connection(wdir) as conn:
                    if conn is None:
                        raise PreventUpdate
                    try:
                        conn.executemany(
                            "INSERT INTO samples_metadata(ms_file_label, label, ms_level, file_type, polarity) "
                            "VALUES (?, ?, ?, ?, ?)",
                            batch_ms
                        )
                        conn.execute(
                            "INSERT INTO ms_data SELECT ms_file_label, scan_id, mz, intensity, scan_time, mz_precursor, "
                            "filterLine, filterLine_ELMAVEN FROM read_parquet(?)", [batch_ms_data])
                    except Exception as e:
                        logging.error(f"DB error: {e}")

                set_progress(round(n_total / n_total * 100, 2))

        notifications = [
            fac.AntdNotification(message="Files processed",
                                                description=f"Successful processed {n_total - len(duplicated_files)} files",
                                                type="success", duration=3,
                                                placement='bottom', showProgress=True)]
        if duplicated_files:
            notifications.append(
                fac.AntdNotification(message="Duplicated files",
                                     description=f"There are {len(duplicated_files)} files that were "
                                                 f"ignored",
                                     type="warning", duration=3,
                                     placement='bottom', showProgress=True)
            )
        return notifications, True, {'display': 'none'}

    @du.callback(
        output=Output("metadata-uploader-input", "data"),
        id="metadata-uploader",
    )
    def metadata_upload_completed(status):
        logging.warning(f"Upload status: {status} ({type(status)})")
        return [status.latest_file.as_posix(), status.n_uploaded, status.n_total]

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("metadata-processed-store", "data"),
        Input("metadata-uploader-input", "data"),
        State("wdir", "children"),
        prevent_initial_call=True,
    )
    def process_metadata_files(uploaded_data, wdir):
        if not uploaded_data:
            raise PreventUpdate

        latest_file, n_uploaded, n_total = uploaded_data
        # TODO: check if file contains correct columns and minimum data
        df = pd.read_csv(latest_file)
        # rename in_analysis to use_for_analysis
        df.rename(columns={"in_analysis": "use_for_analysis"}, inplace=True)

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)

        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate

            conn.register('metadata_df', df)

            columns_to_update = {"use_for_optimization": "BOOLEAN", "use_for_analysis": "BOOLEAN", "color": "VARCHAR",
                                 "label": "VARCHAR", "sample_type": "VARCHAR", "run_order": "INTEGER",
                                 "plate": "VARCHAR", "plate_row": "VARCHAR", "plate_column": "INTEGER"}

            set_clauses = []
            for col, cast in columns_to_update.items():
                set_clauses.append(f"""
                        {col} = CASE 
                            WHEN metadata_df.{col} IS NOT NULL 
                            THEN CAST(metadata_df.{col} AS {cast}) 
                            ELSE samples_metadata.{col} 
                        END
                    """)
            set_clause = ", ".join(set_clauses)
            stmt = f"""
                    UPDATE samples_metadata 
                    SET {set_clause}
                    FROM metadata_df 
                    WHERE samples_metadata.ms_file_label = metadata_df.ms_file_label
                """

            conn.execute(stmt)
            conn.unregister('metadata_df')

            ms_colors = conn.execute("SELECT ms_file_label, color FROM samples_metadata").df()
            assigned_colors = {row['ms_file_label']: row['color'] for _, row in ms_colors.iterrows() if row['color']}

            if len(assigned_colors) != len(ms_colors):
                colors = make_palette_hsv(ms_colors['ms_file_label'], existing_map=assigned_colors,
                                          s_range=(0.90, 0.95), v_range=(0.90, 0.95)
                                          )
                colors_df = pd.DataFrame(colors.items(), columns=['ms_file_label', 'color'])
                conn.register('colors_df', colors_df)
                conn.execute("""
                             UPDATE samples_metadata
                             SET color = colors_df.color
                             FROM colors_df
                             WHERE samples_metadata.ms_file_label = colors_df.ms_file_label"""
                             )
                conn.unregister('colors_df')
        # TODO:
        # 1. check if any file is marked as use_for_optimization with not color
        # 2. check for duplicate labels or colors

        return (fac.AntdNotification(message="Saved metadata.", description="Metadata file added successfully.",
                                     type="success", duration=3,
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True
                                     ),
                1)
