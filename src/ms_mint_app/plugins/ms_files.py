import json
import logging
import os
import tempfile
from pathlib import Path as P, Path

import dash
import feffery_antd_components as fac
import feffery_utils_components as fuc
import pandas as pd
import polars as pl
import time
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from ..colors import make_palette_hsv
from ..duckdb_manager import duckdb_connection
from ..plugin_interface import PluginInterface
from ..tools import get_metadata

_label = "MS-Files"

home_path = Path.home()


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

_layout = html.Div(
    [
        fac.AntdFlex(
            [
                fac.AntdFlex(
                    [
                        fac.AntdTitle(
                            'MS-Files', level=4, style={'margin': '0'}
                        ),
                        fac.AntdIcon(
                            id='ms-files-tour-icon',
                            icon='pi-info',
                            style={"cursor": "pointer", 'paddingLeft': '10px'},
                        ),
                        fac.AntdSpace(
                            [
                                fac.AntdButton(
                                    'Load MS-Files',
                                    id={
                                        'action': 'file-explorer',
                                        'type': 'ms-files',
                                    },
                                    style={'textTransform': 'uppercase'},
                                ),
                                fac.AntdButton(
                                    'Load Metadata',
                                    id={
                                        'action': 'file-explorer',
                                        'type': 'metadata',
                                    },
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
            )]
        ),
        dcc.Store(id="ms-processed-output"),
        dcc.Store(id="metadata-uploader-input"),
        dcc.Store(id="metadata-processed-store"),
        html.Div(
            id="progress-container",
            style={"display": "none"},
                fac.AntdDropdown(
                    id='ms-options',
                    title='Options',
                    buttonMode=True,
                    arrow=True,
                    menuItems=[
                        {'title': 'Mark selected for optimization'},
                        {'title': ''},
                        {'isDivider': True},
                        {'title': 'Delete selected', 'key': 'delete-selected'},
                    ],
                    buttonProps={'style': {'textTransform': 'uppercase'}},
                ),
            ],
            justify="space-between",
            align="center",
            gap="middle",
        ),
            children=[
                html.P("Processing files..."),
                fac.AntdProgress(
                    id="ms-progress-bar",
                    showInfo=True,

        fac.AntdModal(
            "Are you sure you want to delete the selected files?",
            title="Delete confirmation",
            id="delete-confirmation-modal",
            okButtonProps={"danger": True},
            renderFooter=True,
            locale='en-us',
        ),
                )
            ]
        html.Div(
            [
                fac.AntdSpin(
                    fac.AntdTable(
                        id='ms-files-table',
                        containerId='table-container',
                        columns=[
                            {
                                'title': 'MS-File Label',
                                'dataIndex': 'ms_file_label',
                                'width': '260px',
                            },
                            {
                                'title': 'Label',
                                'dataIndex': 'label',
                                'width': '260px',
                                'editable': True,
                                'editOptions': {
                                    'mode': 'text-area',
                                    'autoSize': {'minRows': 1, 'maxRows': 3},
                                },
                            },
                            {
                                'title': 'Color',
                                'dataIndex': 'color',
                                'width': '80px',
                                'renderOptions': {'renderType': 'button'},
                            },
                            {
                                'title': 'For Optimization',
                                'dataIndex': 'use_for_optimization',
                                'renderOptions': {'renderType': 'switch'},
                                'width': '170px',
                            },
                            {
                                'title': 'For Analysis',
                                'dataIndex': 'use_for_analysis',
                                'renderOptions': {'renderType': 'switch'},
                                'width': '150px',
                            },
                            {
                                'title': 'Sample Type',
                                'dataIndex': 'sample_type',
                                'width': '150px',
                                'editable': True,
                            },
                            {
                                'title': 'Polarity',
                                'dataIndex': 'polarity',
                                'width': '100px',
                            },
                            {
                                'title': 'MS Type',
                                'dataIndex': 'ms_type',
                                'width': '120px',
                            },
                            {
                                'title': 'Run Order',
                                'dataIndex': 'run_order',
                                'width': '120px',
                                'editable': True,
                            },
                            {
                                'title': 'Plate',
                                'dataIndex': 'plate',
                                'width': '100px',
                                'editable': True,
                            },
                            {
                                'title': 'Plate Row',
                                'dataIndex': 'plate_row',
                                'width': '110px',
                                'editable': True,
                            },
                            {
                                'title': 'Plate Col.',
                                'dataIndex': 'plate_column',
                                'width': '110px',
                                'editable': True,
                            },
                        ],
                        titlePopoverInfo={
                            'ms_file_label': {
                                'title': 'ms_file_label',
                                'content': 'This is ms_file_label field',
                            },
                            'label': {
                                'title': 'label',
                                'content': 'This is label field',
                            },
                            'dash_component': {
                                'title': 'dash_component',
                                'content': 'This is dash_component field',
                            },
                            'use_for_optimization': {
                                'title': 'use_for_optimization',
                                'content': 'This is use_for_optimization field',
                            },
                            'use_for_analysis': {
                                'title': 'use_for_analysis',
                                'content': 'This is use_for_analysis field',
                            },
                            'sample_type': {
                                'title': 'sample_type',
                                'content': 'This is sample_type field',
                            },
                            'polarity': {
                                'title': 'polarity',
                                'content': 'This is polarity field',
                            },
                            'file_type': {
                                'title': 'file_type',
                                'content': 'This is file_type field',
                            },
                            'run_order': {
                                'title': 'run_order',
                                'content': 'This is run_order field',
                            },
                            'plate': {
                                'title': 'plate',
                                'content': 'This is plate field',
                            },
                            'plate_row': {
                                'title': 'plate_row',
                                'content': 'This is plate_row field',
                            },
                            'plate_column': {
                                'title': 'plate_column',
                                'content': 'This is plate_column field',
                            },
                        },
                        filterOptions={
                            'ms_file_label': {'filterSearch': True},
                            'ms_type': {'filterMode': 'checkbox'},
                            'use_for_optimization': {'filterMode': 'checkbox'},
                            'use_for_analysis': {'filterMode': 'checkbox'},
                            'sample_type': {'filterSearch': True},
                            'polarity': {'filterMode': 'checkbox'},
                        },
                        sortOptions={'sortDataIndexes': ['run_order']},
                        pagination={'position': 'bottomCenter'},
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
            id='ms-files-table-container',
            style={'padding': '1rem 0'},
        ),
        dcc.Store(id="processing-output-store"),
        dcc.Store(id="selected-folder-path"),
        dcc.Store(id="selected-files", data={}),
        dcc.Store(id='processing-type-store', data={}),
        dcc.Store(id="color-changed-store"),
        dcc.Store(id="ms-delete-store"),
    ]
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
