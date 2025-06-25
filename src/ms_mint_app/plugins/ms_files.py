import os
import shutil
import uuid
import logging

import dash
import numpy as np
import tempfile

from pathlib import Path as P, Path

import pandas as pd

from dash import html, dcc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc

from ms_mint.io import convert_ms_file_to_feather

from dash_tabulator import DashTabulator

import dash_uploader as du

from .utils import create_toast
from .. import tools as T
from ..plugin_interface import PluginInterface

import concurrent.futures

_label = "MS-Files"

class MsFilesPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 2
        self.executor: concurrent.futures.ProcessPoolExecutor = None
        self.futures: list[concurrent.futures.Future] = []
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
    # "height": "500px",
    # "reactiveData": True,  # Solo actualiza lo necesario
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
        "widthGrow": 3
    },
    {
        "title": "Color",
        "field": "color",
        "headerFilter": False,
        "editor": None,
        "formatter": "color",
        # "width": "3px",
        "headerSort": False,
    },
    {
        "title": "Use for Optimization",
        "field": "use_for_optimization",
        "headerFilter": False,
        "formatter": "tickCross",
        # "width": "6px",
        "headerSort": True,
        "hozAlign": "center",
        "editor": "tickCross",
        "widthGrow": 2,
    },
    {
        "title": "In Analysis",
        "field": "in_analysis",
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
        "widthGrow": 3
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
    # style={"Height": 0, "marginTop": "20px"},
    children=[
        DashTabulator(
            id="ms-files-table",
            columns=columns,
            options=options,
            # clearFilterButtonType=clearFilterButtonType,
        )
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


def get_upload_component(uid, file_stypes, text, disabled=False, max_files=10000):
    return du.Upload(
        id=uid,
        max_file_size=1800,  # 1800 MB
        max_files=max_files,
        filetypes=file_stypes,
        upload_id=str(uuid.uuid1()),  # Unique session id
        text=text,
        disabled=disabled
    )

_layout = html.Div(
    [
        html.H4("Upload Mass Spec / Metadata files"),
        dbc.Row(
            [dbc.Col(
                get_upload_component(
                    uid="ms-uploader",
                    file_stypes=["tar", "zip", "mzxml", "mzml", "mzXML", "mzML", "mzMLb", "feather", "parquet"],
                    text="Upload mzXML/mzML files.",
                ),
            ),
            dbc.Col(
                get_upload_component(
                    uid="metadata-uploader",
                    file_stypes=["csv"],
                    text="Upload METADATA files.",
                    # disabled=True,
                    max_files=1
                ),
            )]
        ),
        dcc.Store(id="ms-uploader-store"),
        dcc.Store(id="metadata-uploader-store"),
        dcc.Store(id="metadata-processed-store"),
        html.Div(
                id="progress-container",
                style={"display": "none"},
                children=[
                    html.P("Progreso del procesamiento"),
                    dbc.Progress(id="ms-progress-bar", animated=True, striped=True, label="Processing files...")
                ]
            ),
        dcc.Interval(id="ms-poll-interval", interval=1000, n_intervals=0, disabled=True),
        modal_confirmation,
        dcc.Store(id="ms-delete-store"),
        html.Div([
            dbc.Row([dbc.Col([dbc.Button("Delete selected file", id="ms-delete", color="danger")],
                             style={"text-align": "right"}), ]),
            dcc.Loading(ms_files_table)]
        ),
        html.Div(id="ms-n-files", style={"max-width": "300px"}),
        html.Div(id="ms-uploader-fns", style={"visibility": "hidden"}),
    ]
)


_outputs = html.Div(
    id="ms-outputs",
    children=[
        html.Div(id={"index": "ms-delete-output", "type": "output"}),
        html.Div(id={"index": "ms-save-output", "type": "output"}),
        html.Div(id={"index": "ms-import-from-url-output", "type": "output"}),
        dcc.Store(id="ms-uploader-output"),
        html.Div(id={"index": "metadata-uploader-output", "type": "output"}),
        html.Div(id={"index": "ms-new-target-output", "type": "output"}),
    ],
)


def layout():
    return _layout

#
def process_file(file_path: Path, output_dir):
    # move converted file to processed folder
    file_path = Path(file_path)
    output_file = Path(output_dir).joinpath(file_path.name).with_suffix(".feather")
    ff = convert_ms_file_to_feather(file_path, output_file)
    # remove original file
    if os.path.isfile(ff):
        os.remove(file_path)
    return ff


def callbacks(cls, app, fsc, cache):
    @app.callback(
        Output("ms-files-table", "data"),
        Input("ms-uploader-output", "data"),
        Input("metadata-processed-store", "data"),
        Input("wdir", "children"),
        Input("ms-delete-store", "data"),
        State("active-workspace", "children"),
        State("ms-files-table", "data"),
    )
    def ms_files_table(value, value2, wdir, files_deleted, workspace, current_data):

        ms_files = T.get_ms_fns(wdir)
        logging.info(f"# Files in {wdir} {workspace} {len(ms_files)}")

        ms_files_names = []
        files_type = []
        for fn in ms_files:
            fn_p = Path(fn)
            if fn_p.stem[-4:] not in ['_ms1', '_ms2']:
                ms_files_names.append(fn_p.stem)
            else:
                ms_files_names.append(fn_p.stem[:-4])
            files_type.append(T.get_ms_level_from_filename(fn))

        df = pd.DataFrame(
            {
                "ms_file_label": ms_files_names,
                "file_type": files_type,
            }
        )
        mdf = T.get_metadata(wdir)

        if value2 is not None or not mdf.empty:
            df = T.merge_metadata(df, mdf)

        if df.empty and files_deleted is None:
            raise PreventUpdate

        if current_data is not None:
            prev_df = pd.DataFrame(current_data)
            if df.reset_index(drop=True).equals(prev_df.reset_index(drop=True)):
                raise PreventUpdate
        return df.to_dict("records")

    @app.callback(
        Output("ms-data-table", "data"),
        Input("wdir", "children"),
        Input("ms-files-table", "rowClicked"),
    )
    def content_table(wdir, row):
        ff = {fo.stem: fo for fo in P(wdir).joinpath("ms_files").glob("*.feather")}
        if row is None or f'{row["ms_file_label"]}_{row["file_type"]}' not in ff:
            raise PreventUpdate
        fn = f'{row["ms_file_label"]}_{row["file_type"]}'
        df = pd.read_feather(ff[fn])
        return df.to_dict("records")


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
        Output("global-toast-container", "children", allow_duplicate=True),
        Output("ms-delete-store", "data"),
        Input("ms-mc-confirm", "n_clicks"),
        State("ms-files-table", "multiRowsClicked"),
        State("wdir", "children"),
        State("global-toast-container", "children"),
        prevent_initial_call=True,
    )
    def confirm_and_delete(n_confirm, rows, wdir, current_toasts):
        if n_confirm is None or not rows:
            raise PreventUpdate

        target_dir = os.path.join(wdir, "ms_files")

        removed_files = []
        failed_files = []
        for row in rows:
            filename = row["ms_file_label"]
            ft = row["file_type"]
            fn = f"{filename}_{ft}.feather"
            file_path = Path(target_dir) / fn
            try:
                if file_path.exists():
                    os.remove(file_path)
                    removed_files.append(fn)
            except Exception as e:
                logging.error(f"Error al eliminar {fn}: {str(e)}")
                failed_files.append(fn)

        dfl = "\n".join(f"- {m}" for m in removed_files)
        ffl = "\n".join(f"- {m}" for m in failed_files)
        msd = dcc.Markdown(
            f"Successfully deleted {len(rows)} files.\n"
            f"{dfl}\n"
        )
        msf = dcc.Markdown(
            f"Failed to delete {len(rows)} files.\n"
            f"{ffl}"
        )
        new_toasts = []
        if removed_files:
            new_toasts.append(create_toast(msd,"Success deletion", "success"))
        if failed_files:
            new_toasts.append(create_toast(msf,"Failed deletion", "danger"))
        updated_toasts = current_toasts + new_toasts
        return updated_toasts, len(rows)

    @du.callback(
        output=[Output("ms-uploader-fns", "children"),
                Output("ms-progress-bar", "max"),
                Output("ms-uploader-store", "data")]                ,
        id="ms-uploader",
    )
    def ms_upload_completed(status):
        logging.warning(f"Upload status: {status} ({type(status)})")
        return [str(fn) for fn in status.uploaded_files], status.n_total, status.n_total

    @du.callback(
        output=Output("metadata-uploader-store", "data"),
        id="metadata-uploader",
    )
    def metadata_upload_completed(status):
        logging.warning(f"Upload status: {status} ({type(status)})")
        return [str(fn) for fn in status.uploaded_files], status.n_total, status.n_total

    @app.callback(
        Output("global-toast-container", "children", allow_duplicate=True),
        Input("ms-files-table", "cellEdited"),
        State("ms-files-table", "data"),
        State("wdir", "children"),
        State("global-toast-container", "children"),
        prevent_initial_call=True,
    )
    def save_table_on_edit(cell_edited, data, wdir, current_toasts):
        """
        This callback saves the table on cell edits.
        This saves some bandwidth.
        """
        if data is None or cell_edited is None:
            raise PreventUpdate
        df = pd.DataFrame(data)
        T.write_metadata(df, wdir)
        new_toast = create_toast("Metadata saved.", "Success metadata saved", "success")
        updated_toasts = current_toasts + [new_toast]
        return updated_toasts

    @app.callback(
        Output("global-toast-container", "children", allow_duplicate=True),
        Output("ms-uploader-output", "data"),
        Output("ms-poll-interval", "disabled"),
        Output('ms-progress-bar', 'value'),
        Output('ms-progress-bar', 'label'),
        Output("progress-container", "style"),
        Output("metadata-uploader", "disabled"),
        Input("ms-poll-interval", "n_intervals"),
        Input("ms-uploader-store", "data"),
        Input("ms-uploader-fns", "children"),
        State("global-toast-container", "children"),
        State("wdir", "children"),
        prevent_initial_call=True
    )
    def process_ms_files(n_interval, n_total, fns, current_toasts, wdir):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "ms-uploader-fns":
            if fns is None or len(fns) == 0:
                raise PreventUpdate
            if len(fns) == 1:
                cls.executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
            ms_dir = T.get_ms_dirname(wdir)
            for fn in fns[len(cls.futures):]:
                T.fix_first_emtpy_line_after_upload_workaround(fn)
                cls.futures.append(cls.executor.submit(process_file, fn, ms_dir))
            return [], dash.no_update, False, 0, '', {"display": "block"}, True
        elif trigger_id == "ms-poll-interval":
            value = sum(future.done() for future in cls.futures)
            ms_poll_interval_disabled = False
            metadata_uploader_disabled = True
            style = {"display": "block"}
            new_toast = []
            if value == n_total:
                cls.executor.shutdown()
                cls.futures = []
                ms_poll_interval_disabled = True
                style = {"display": "none"}
                metadata_uploader_disabled = False
                # create the metadata file
                metadata_df = T.get_metadata(wdir)
                T.write_metadata(metadata_df, wdir)

                new_toast = create_toast(f"{n_total} files processed", "Success files processing", "success")
            updated_toasts = current_toasts + [new_toast]

            return (updated_toasts, dbc.Alert(f"Processing {n_total} uploaded files...", color="info"),
                    ms_poll_interval_disabled,
                    value,
                    f"Processed {value}/{n_total} files...",
                    style,
                    metadata_uploader_disabled)
        raise PreventUpdate

    @app.callback(
        Output("global-toast-container", "children", allow_duplicate=True),
        Output("metadata-processed-store", "data"),
        Input("metadata-uploader-store", "data"),
        State("wdir", "children"),
        State("global-toast-container", "children"),
        prevent_initial_call=True,
    )
    def process_metadata_files(files, wdir, current_toasts):
        if not files:
            raise PreventUpdate
        df = T.get_metadata(wdir)
        new_df = pd.read_csv(files[0][0])
        df = T.merge_metadata(df, new_df)
        if "index" not in df.columns:
            df = df.reset_index()
        T.write_metadata(df, wdir)
        new_toast = create_toast("Metadata file added successfully.", "Success Metadata Added", "success")
        updated_toasts = current_toasts + [new_toast]
        return updated_toasts, 1


    @app.callback(Output("ms-n-files", "children"),
                  Input("ms-files-table", "data"))
    def n_files(data):
        n_files = len(data)
        return dbc.Alert(f"{n_files} files in current workspace.", color="info")

