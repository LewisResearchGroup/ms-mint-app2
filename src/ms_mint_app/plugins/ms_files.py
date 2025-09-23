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

from ..colors import make_palette
from ..plugin_interface import PluginInterface
import feffery_antd_components as fac

from ..duckdb_manager import duckdb_connection
import concurrent.futures

_label = "MS-Files"


class MsFilesPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 2
        self.executor: concurrent.futures.ProcessPoolExecutor = None
        self.futures: list[concurrent.futures.Future] = []
        self.processed = 0
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
                    cancel_button=False,
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
                    cancel_button=False
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
    @app.callback(
        Output("ms-files-table", "data"),
        Input("ms-processed-output", "data"),
        Input("metadata-processed-store", "data"),
        Input("ms-delete-store", "data"),
        Input("wdir", "children"),
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
        set_props("ms-progress-bar", {"max": status.n_total})
        return [status.latest_file.as_posix(), status.n_uploaded, status.n_total]

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("ms-files-table", "cellEdited"),
        State("wdir", "children"),
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
        Output("notifications-container", "children", allow_duplicate=True),
        Output("ms-poll-interval", "disabled", allow_duplicate=True),

        Input("ms-uploader-input", "data"),
        State("wdir", "children"),
        prevent_initial_call=True
    )
    def check_ms_files(fns, wdir):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        latest_file, n_uploaded, n_total = fns if fns is not None else (None, 0, 0)
        if not latest_file or n_uploaded == 0:
            raise PreventUpdate

        latest_file = Path(latest_file)
        # check if latest_file is in the db
        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            duplicated_file_sele = conn.execute("SELECT COUNT(*) FROM samples_metadata WHERE ms_file_label = ?", (latest_file.stem,)).fetchone()

        if duplicated_file_sele is not None and duplicated_file_sele[0] > 0:
            cls.processed += 1

            return fac.AntdNotification(message="Duplicated file",
                                        description=f"{latest_file.stem} already loaded. Skipping...",
                                        type="error",
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        stack=True
                                        ), False
        else:
            if not cls.executor:
                cls.executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
            cls.futures.append(cls.executor.submit(convert_mzxml_to_parquet, latest_file))

        return dash.no_update, False


    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("ms-processed-output", "data"),
        Output("ms-poll-interval", "disabled", allow_duplicate=True),
        Output("progress-container", "style"),
        Output("metadata-uploader", "disabled"),

        Input("ms-poll-interval", "n_intervals"),
        Input("ms-uploader-input", "data"),
        State('ms-progress-bar', 'value'),

        State("wdir", "children"),
        prevent_initial_call=True
    )
    def get_processed_files(n_interval, uploaded_data, current_progress, wdir):

        notification = dash.no_update
        uploader_output = dash.no_update
        ms_poll_interval_disabled = False
        progress_container_style = {"display": "block"}
        metadata_uploader_disabled = True


        latest_file, n_uploaded, n_total = uploaded_data if uploaded_data is not None else (None, 0, 0)

        futures_done = sum(future.done() for future in cls.futures)
        
        
        processed_files = cls.processed + futures_done

        # check if not futures
        if not cls.futures:
            raise PreventUpdate

        if futures_done:
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                # Iterate over a copy of the futures list to allow safe removal
                for future in cls.futures[:]:
                    if future.done():
                        try:
                            _ms_file_label, _ms_level, _polarity, _ms_data_df = future.result()
                            try:
                                conn.execute(
                                    "INSERT INTO samples_metadata(ms_file_label, label, ms_level, file_type, "
                                    'use_for_analysis, polarity) VALUES (?, ?, ?, ?, True, ?)',
                                    [_ms_file_label, _ms_file_label, _ms_level, f'ms{_ms_level}', _polarity]
                                )
                                conn.execute(
                                    "INSERT INTO ms_data SELECT * FROM _ms_data_df")
                            except Exception as e:
                                logging.error(f"DB error: {e}")

                            cls.futures.remove(future)
                            cls.processed += 1
                        except Exception as e:
                            logging.error(f"Error processing future: {e}")

        if processed_files == current_progress:
            raise PreventUpdate
        elif processed_files == 0:
            set_props("ms-progress-bar",
                      {"value": n_total, "label": "Processing MS files..."})
        elif processed_files < n_total:
            set_props("ms-progress-bar",
                      {"value": processed_files, "label": f"Processed {processed_files}/{n_total} files..."})
        else:
            notification = fac.AntdNotification(message="Files processed",
                                                description=f"Successful processed {n_total} files",
                                                type="success", duration=3,
                                                placement='bottom', showProgress=True, stack=True)
            uploader_output = True
            ms_poll_interval_disabled = True
            progress_container_style = {"display": "none"}
            metadata_uploader_disabled = False
            if cls.executor:
                try:
                    cls.executor.shutdown()
                    cls.executor = None
                except Exception as e:
                    pass
        with duckdb_connection(wdir) as conn:
            sm_df = conn.execute("SELECT * FROM samples_metadata").df()
            print(f"{sm_df = }")
        return notification, uploader_output, ms_poll_interval_disabled, progress_container_style, metadata_uploader_disabled

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
                colors = make_palette(ms_colors['ms_file_label'], existing_map=assigned_colors)
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
