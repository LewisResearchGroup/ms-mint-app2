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
        fac.AntdModal(
            children=[
                fac.AntdFlex(
                    [
                        html.Div(
                            [
                                fac.AntdBreadcrumb(
                                    id='current-path-modal', items=[]
                                )
                            ],
                            style={'margin': '10px 0', 'flexGrow': 1},
                        ),
                        fac.AntdTable(
                            id="dir-content-table",
                            maxHeight='350px',
                            maxWidth='450px',
                            locale='en-us',
                            columns=[
                                {
                                    'title': '',
                                    'dataIndex': 'selection',
                                    'width': 35,
                                    'renderOptions': {'renderType': 'button'},
                                },
                                {
                                    'title': 'Name',
                                    'dataIndex': 'file_name',
                                    "align": 'left',
                                    'renderOptions': {'renderType': 'button'},
                                },
                                {
                                    'title': 'Files',
                                    'dataIndex': 'files',
                                    'width': 80,
                                    'renderOptions': {'renderType': 'tags'},
                                },
                            ],
                            pagination=False,
                            style={'flexGrow': 4, 'minHeight': '400px'},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        fac.AntdFlex(
                                            [
                                                html.Div(
                                                    html.Strong(
                                                        "Selected files..."
                                                    )
                                                ),
                                                html.Div(
                                                    fac.AntdSelect(
                                                        id='selected-files-extensions',
                                                        size="small",
                                                        mode="multiple",
                                                        placeholder='extensions',
                                                        style={
                                                            "width": "100%",
                                                            'minWidth': '200px',
                                                        },
                                                        locale="en-us",
                                                        allowClear=False,
                                                        disabled=True,
                                                    ),
                                                ),
                                            ],
                                            justify='space-between',
                                        ),
                                        fac.AntdSpace(
                                            id='selected-files-display',
                                            direction='horizontal',
                                            wrap=True,
                                            align='start',
                                            style={
                                                'maxHeight': '180px',
                                                'overflowY': 'auto',
                                            },
                                        ),
                                    ],
                                )
                            ],
                            id="selected-files-area",
                            style={'margin': '10px 0', 'flexGrow': 2},
                        ),
                    ],
                    id='selection-container',
                    vertical=True,
                ),
                html.Div(
                    [
                        html.H4("Processing files..."),
                        fac.AntdProgress(
                            id='sm-processing-progress',
                            percent=0,
                        ),
                        fac.AntdButton(
                            'Cancel',
                            id='cancel-ms-processing',
                            style={
                                'alignText': 'center',
                            },
                        ),
                    ],
                    id='sm-processing-progress-container',
                    style={'display': 'none'},
                ),
            ],
            id="selection-modal",
            title='Load MS-Files',
            width=700,
            renderFooter=True,
            locale='en-us',
            confirmAutoSpin=True,
            loadingOkText='Processing Files...',
            okClickClose=False,
            closable=False,
            maskClosable=False,
            destroyOnClose=True,
            okText="Process Files",
            centered=True,
            styles={'body': {'height': "75vh"}},
        ),
        fac.AntdModal(
            "Are you sure you want to delete the selected files?",
            title="Delete confirmation",
            id="delete-confirmation-modal",
            okButtonProps={"danger": True},
            renderFooter=True,
            locale='en-us',
        ),
        fac.AntdModal(
            [
                fac.AntdCenter(
                    fuc.FefferyHexColorPicker(
                        id='hex-color-picker', showAlpha=True
                    )
                )
            ],
            id='color-picker-modal',
            renderFooter=True,
            width=300,
            styles={
                'body': {
                    'height': 230,
                    'alignItems': 'center',
                    'alignContent': 'end',
                }
            },
            locale='en-us',
        ),
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


def get_content_list(path, extensions):
    """Generate the folder and files list"""
    allow_folder = extensions != ['.csv']

    current_path = Path(path)
    data = []
    c = 0
    for item in sorted(current_path.iterdir(), key=lambda item: item.name):
        if item.name.startswith('.'):
            continue

        if item.is_file() and item.suffix not in extensions:
            continue
        dash_comp = {}
        if item.is_dir():
            subfolders = sum(bool(si.is_dir() and not si.name.startswith('.'))
                             for si in item.iterdir())

            files_for_selecting = [file for ext in extensions for file in item.glob(f"*{ext}")]

            if subfolders or files_for_selecting:
                dash_comp['file_name'] = {
                    'content': item.name,
                    'type': 'link',
                    'custom': {'path': item.as_posix(), 'type': 'folder', 'is_link': True},
                    'icon': 'antd-folder'
                }
                dash_comp['files'] = {
                    'tag': f"{len(files_for_selecting)}" if len(files_for_selecting) else '',
                }

                dash_comp['selection'] = {
                    'content': '',
                    'type': 'button',
                    'color': 'green',
                    'variant': 'filled',
                    'custom': {'path': item.as_posix(), 'type': 'folder', 'is_link': False},
                    'icon': 'antd-plus',
                    'disabled': not allow_folder
                }
        else:
            dash_comp['file_name'] = {
                'content': item.name,
                'type': 'link',
                'custom': {'path': item.as_posix(), 'type': 'file', 'is_link': False},
                'style': {'pointer-events': 'none'},
                'icon': 'antd-file'
            }
            dash_comp['files'] = {
                'content': "",
            }
            dash_comp['selection'] = {
                'content': '',
                'type': 'button',
                'color': 'green',
                'variant': 'filled',
                'custom': {'path': item.as_posix(), 'type': 'file', 'is_link': False},
                'icon': 'antd-plus'
            }
        if dash_comp:
            c += 1
            data.append(dash_comp)
    return data


def process_ms_files(wdir, set_progress, selected_files):
    file_list = [file for folder in selected_files.values() for file in folder]
    n_total = len(file_list)
    failed_files = {}
    total_processed = 0
    import concurrent.futures
    from ms_mint.io import convert_mzxml_to_parquet_pl

    # get the ms_file_label data as df to avoid multiple queries
    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        data = conn.execute("SELECT ms_file_label FROM samples_metadata").pl()
    t1 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:

        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            for file_path in file_list:
                if Path(file_path).stem in data['ms_file_label'].to_list():
                    failed_files[file_path] = 'duplicate'
                    set_progress(round(len(failed_files) / n_total * 100, 2))
                else:
                    futures.append(executor.submit(convert_mzxml_to_parquet_pl, file_path, tmp_dir=tmpdir))

            batch_ms = []
            batch_ms_data = []
            batch_size = 4

            for future in concurrent.futures.as_completed(futures):
                _file_path, _ms_file_label, _ms_level, _polarity, _parquet_df = future.result()
                batch_ms.append((_ms_file_label, _ms_file_label, f'ms{_ms_level}', _polarity))
                batch_ms_data.append(_parquet_df)

                if len(batch_ms) == batch_size:
                    with duckdb_connection(wdir) as conn:
                        if conn is None:
                            raise PreventUpdate
                        try:
                            pldf = pl.DataFrame(batch_ms, schema=['ms_file_label', 'label', 'ms_type', 'polarity'],
                                                orient="row")
                            conn.execute(
                                "INSERT INTO samples_metadata(ms_file_label, label, ms_type, polarity) "
                                "SELECT ms_file_label, label, ms_type, polarity FROM pldf"
                            )
                            conn.execute(
                                "INSERT INTO ms_data (ms_file_label, scan_id, mz, intensity, scan_time, mz_precursor, "
                                "filterLine, filterLine_ELMAVEN) "
                                "SELECT ms_file_label, scan_id, mz, intensity, scan_time, mz_precursor, "
                                "filterLine, filterLine_ELMAVEN FROM read_parquet(?)", [batch_ms_data])
                            total_processed += len(batch_ms_data)
                        except Exception as e:
                            failed_files[_file_path] = str(e)
                            logging.error(f"DB error: {e}")
                    batch_ms = []
                    batch_ms_data = []
                    set_progress(round(total_processed + len(failed_files) / n_total * 100, 1))
            if batch_ms:
                with duckdb_connection(wdir) as conn:
                    if conn is None:
                        raise PreventUpdate
                    try:
                        pldf = pl.DataFrame(batch_ms, schema=['ms_file_label', 'label', 'ms_type', 'polarity'],
                                            orient="row")
                        conn.execute(
                            "INSERT INTO samples_metadata(ms_file_label, label, ms_type, polarity) "
                            "SELECT ms_file_label, label, ms_type, polarity FROM pldf"
                        )
                        conn.execute(
                            "INSERT INTO ms_data SELECT ms_file_label, scan_id, mz, intensity, scan_time, mz_precursor, "
                            "filterLine, filterLine_ELMAVEN FROM read_parquet(?)", [batch_ms_data])
                        total_processed += len(batch_ms_data)
                    except Exception as e:
                        failed_files[_file_path] = str(e)
                        logging.error(f"DB error: {e}")
            set_progress(round(100, 1))
    print(f"{time.time() - t1 = }")
    return total_processed, failed_files


def process_metadata(wdir, set_progress, selected_files):
    file_list = [file for folder in selected_files.values() for file in folder]
    n_total = len(file_list)

    metadata_df, failed_files = get_metadata(file_list)

    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
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

        ms_colors = conn.execute("SELECT ms_file_label, color FROM samples_metadata").df()
        valid = ms_colors[ms_colors["color"].notna() & (ms_colors["color"].str.strip() != "")]
        assigned_colors = dict(zip(valid["ms_file_label"], valid["color"]))

        if len(assigned_colors) != len(ms_colors):
            colors_map = make_palette_hsv(
                ms_colors["ms_file_label"].to_list(),
                existing_map=assigned_colors,
                s_range=(0.90, 0.95),
                v_range=(0.90, 0.95),
            )
            colors_pd = pd.DataFrame({"ms_file_label": list(colors_map.keys()), "color": list(colors_map.values())})
            conn.execute("""
                         UPDATE samples_metadata
                         SET color = colors_pd.color
                         FROM colors_pd
                         WHERE samples_metadata.ms_file_label = colors_pd.ms_file_label"""
                         )
    set_progress(100)
    return len(metadata_df), failed_files


def callbacks(cls, app, fsc, cache, args_namespace):
    @app.callback(
        Output('color-picker-modal', 'visible'),
        Output('hex-color-picker', 'color'),
        Input('ms-files-table', 'nClicksButton'),
        State('ms-files-table', 'clickedCustom'),
        prevent_initial_call=True
    )
    def open_color_picker(nClicksButton, clickedCustom):
        return True, clickedCustom['color']

    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('color-changed-store', 'data'),
        Input('color-picker-modal', 'okCounts'),
        State('hex-color-picker', 'color'),
        State('ms-files-table', 'recentlyButtonClickedRow'),
        State('ms-files-table', 'data'),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def set_color(okCounts, color, recentlyButtonClickedRow, data, wdir):

        print(f"{recentlyButtonClickedRow = }")
        if recentlyButtonClickedRow is None:
            return dash.no_update, dash.no_update

        index = recentlyButtonClickedRow['key']
        previous_color = recentlyButtonClickedRow['color']['content']
        try:
            with duckdb_connection(wdir) as conn:
                conn.execute("UPDATE samples_metadata SET color = ? WHERE ms_file_label = ?",
                             [color, recentlyButtonClickedRow['ms_file_label']])
            data[int(index)]['color'] = {'content': color,
                                         'variant': 'filled',
                                         'custom': {'color': color},
                                         'style': {'background': color, 'width': '70px'}}
            return (fac.AntdNotification(message='Color changed successfully',
                                         description=f'Color changed from {previous_color} to {color}',
                                         type='success',
                                         duration=3,
                                         placement='bottom',
                                         showProgress=True,
                                         stack=True
                                         ),
                    data)
        except Exception as e:
            logging.error(f"DB error: {e}")

            return (fac.AntdNotification(message='Failed to change color',
                                         description=f'Color change failed with {str(e)}',
                                         type='error',
                                         duration=3,
                                         placement='bottom',
                                         showProgress=True,
                                         stack=True
                                         ),
                    dash.no_update)

    @app.callback(
        Output("selection-modal", "visible"),
        Output("selection-modal", 'title'),
        Output('selected-files-extensions', 'options'),
        Output('selected-files-extensions', 'value'),
        Output('processing-type-store', 'data'),

        Input({'action': 'file-explorer', 'type': ALL}, 'nClicks'),
        prevent_initial_call=True
    )
    def open_selection_modal(n_clicks):
        ctx = dash.callback_context
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        prop_data = json.loads(prop_id)
        if prop_data['type'] == "ms-files":
            title = "Load MS Files"
            file_extensions = [".mzXML"]
        elif prop_data['type'] == "metadata":
            title = "Load Metadata"
            file_extensions = [".csv"]
        else:
            title = "Load Targets"
            file_extensions = [".csv"]
        processing_type_store = {'type': prop_data['type'], 'extensions': file_extensions}
        return True, title, file_extensions, file_extensions, processing_type_store

    @app.callback(
        Output('selected-folder-path', 'data', allow_duplicate=True),
        Output('selected-files', 'data', allow_duplicate=True),
        Output('selected-files-display', 'children', allow_duplicate=True),
        Output("sm-processing-progress", "percent"),
        Input('selection-modal', 'visible'),
        prevent_initial_call=True
    )
    def on_modal_close(modal_visible):
        if modal_visible:
            raise PreventUpdate
        return None, {}, [], 0

    @app.callback(
        Output("current-path-modal", "items"),
        Output("dir-content-table", "data"),
        Output('selected-folder-path', 'data'),

        Input('selection-modal', 'visible'),
        Input('dir-content-table', 'nClicksButton'),
        State('dir-content-table', 'clickedCustom'),

        Input('current-path-modal', 'clickedItem'),
        State('selected-folder-path', 'data'),
        State('processing-type-store', 'data'),

        prevent_initial_call=True
    )
    def navigate_folders(modal_visible, nClicksButton, clickedCustom, bc_clicked_item, current_path, processing_type):
        ctx = dash.callback_context
        if not ctx.triggered or not modal_visible:
            return dash.no_update, dash.no_update, dash.no_update

        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if prop_id == 'selection-modal':
            current_modal_path = Path(current_path or home_path)
        elif prop_id == 'current-path-modal':
            current_modal_path = Path(bc_clicked_item['itemKey'] or home_path)
        elif clickedCustom['is_link']:
            current_modal_path = Path(clickedCustom['path'] or home_path)
        else:
            raise PreventUpdate

        all_paths = [path for path in reversed(current_modal_path.parents) if path >= home_path] + [current_modal_path]
        current_path_items = []
        for i, path in enumerate(all_paths):
            if i == 0:
                current_path_items.append({'title': str(path), 'key': str(path)})
            else:
                current_path_items.append({'title': path.name, 'key': str(path)})

        content_items = get_content_list(current_modal_path, processing_type['extensions'])
        return current_path_items, content_items, str(current_modal_path)

    @app.callback(
        Output("selected-files-display", "children"),
        Output('selected-files', 'data'),

        Input("dir-content-table", "nClicksButton"),
        State('dir-content-table', 'clickedCustom'),
        State('selected-files', 'data'),
        State('processing-type-store', 'data'),

        prevent_initial_call=True
    )
    def add_selection(nClicksButton, clickedCustom, selected_files, processing_type):
        if not nClicksButton or clickedCustom['is_link']:
            raise PreventUpdate
        unique_selected_files = {k: set(v) for k, v in selected_files.items()}
        if clickedCustom['type'] == 'folder':
            folder_path = clickedCustom['path']
            ms_files = [file.as_posix() for ext in processing_type['extensions']
                        for file in Path(clickedCustom['path']).rglob(f'*{ext}')]
        else:
            folder_path = Path(clickedCustom['path']).parent.as_posix()
            ms_files = [clickedCustom['path']]

        if folder_path in unique_selected_files:
            unique_selected_files[folder_path].update(ms_files)
        else:
            unique_selected_files[folder_path] = set(ms_files)

        children = [
            fac.AntdTag(
                content=f"{folder_path}: {len(folder_content)} files",
                closeIcon=True,
                id={'type': 'tag-ms-files', 'path': folder_path},
                style={
                    'fontSize': 14,
                    'display': 'flex',
                    'alignItems': 'center',
                },
            )
            for folder_path, folder_content in unique_selected_files.items() if len(folder_content)
        ]
        if not children:
            return dash.no_update, dash.no_update, None
        # set selected_files serializable
        selected_files = {k: list(v) for k, v in unique_selected_files.items()}
        return children, selected_files

    @app.callback(
        Output("selected-files-display", "children", allow_duplicate=True),
        Output('selected-files', 'data', allow_duplicate=True),
        Input({'type': 'tag-ms-files', 'path': ALL}, "closeCounts"),
        State("selected-files-display", "children"),
        State('selected-files', 'data'),
        prevent_initial_call=True
    )
    def delete_tags(closeCounts, children, selected_files):
        for i in closeCounts:
            if i is None:
                continue
            trigger_id = dash.ctx.triggered_id
            for i, child in enumerate(children.copy()):
                if 'id' in child['props'] and trigger_id == child['props']['id']:
                    children.pop(i)
                    try:
                        selected_files.pop(trigger_id['path'])
                    except KeyError:
                        pass
        return children, selected_files

    @app.callback(
        Output("ms-files-table", "data"),
        Output("ms-files-table", "selectedRowKeys"),

        Input("processing-output-store", "data"),
        Input("processing-type-store", "data"),
        Input("ms-delete-store", "data"),
        Input('color-changed-store', 'data'),
        Input("wdir", "data"),
    )
    def ms_files_table(processing_output, processing_type, files_deleted, color_changed, wdir):

        # processing_type also store info about targets selections since it is the same modal for all of them
        if wdir is None or processing_type.get('type') == 'targets':
            raise PreventUpdate

        with duckdb_connection(wdir) as conn:
            if conn is None:
                return []
            dfpl = conn.execute("SELECT * FROM samples_metadata").pl()

        if len(dfpl) == 0:
            raise PreventUpdate
        data = dfpl.with_columns(
            pl.col('color').map_elements(
                lambda value: {
                    'content': value,
                    'variant': 'filled',
                    'custom': {'color': value},
                    'style': {'background': value, 'width': '70px'}
                },
                return_dtype=pl.Struct({
                    'content': pl.String,
                    'variant': pl.String,
                    'custom': pl.Struct({'color': pl.String}),
                    'style': pl.Struct({'background': pl.String, 'width': pl.String})
                }),
                skip_nulls=False,
            ).alias('color'),
            pl.col('use_for_optimization').map_elements(
                lambda value: {'checked': value},
                return_dtype=pl.Object  # Specify that the result is a Python object
            ).alias('use_for_optimization'),
            pl.col('use_for_analysis').map_elements(
                lambda value: {'checked': value},
                return_dtype=pl.Object
            ).alias('use_for_analysis'),
        )

        print(f"{data.to_dicts() = }")
        return data.to_dicts(), []

    @app.callback(
        Output("delete-confirmation-modal", "visible"),

        Input("ms-options", "nClicks"),
        State("ms-options", "clickedKey"),
        State('ms-files-table', 'selectedRows'),
    )
    def toggle_modal(nClicks, clickedKey, selectedRows):
        ctx = dash.callback_context
        return bool(
            ctx.triggered
            and len(selectedRows) != 0
            and clickedKey == "delete-selected"
        )

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("ms-delete-store", "data"),

        Input("delete-confirmation-modal", "okCounts"),
        State('ms-files-table', 'selectedRows'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def confirm_and_delete(okCounts, selectedRows, wdir):

        if okCounts is None or not selectedRows:
            raise PreventUpdate

        remove_ms_file = [row["ms_file_label"] for row in selectedRows]

        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            # TODO: remove data from  results as well
            conn.execute("DELETE FROM samples_metadata WHERE ms_file_label IN ?", (remove_ms_file,))
            conn.execute("DELETE FROM ms_data WHERE ms_file_label IN ?", (remove_ms_file,))
            conn.execute("DELETE FROM chromatograms WHERE ms_file_label IN ?", (remove_ms_file,))
            # conn.execute("DELETE FROM results WHERE ms_file_label = ?", (filename,))

        return (fac.AntdNotification(message="Delete files",
                                     description=f"{len(selectedRows)} files deleted successful",
                                     type="success",
                                     duration=3,
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True
                                     ),
                len(selectedRows))

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("ms-files-table", "recentlyChangedRow"),
        State("ms-files-table", "recentlyChangedColumn"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def save_table_on_edit(row_edited, column_edited, wdir):
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
                query = f"UPDATE samples_metadata SET {column_edited} = ? WHERE ms_file_label = ?"
                conn.execute(query, [row_edited[column_edited], row_edited['ms_file_label']])
            return fac.AntdNotification(message="Successfully edition saved",
                                        type="success",
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        stack=True
                                        )
        except Exception as e:
            logging.error(f"Error updating metadata: {e}")
            return fac.AntdNotification(message="Failed to save edition",
                                        description=f"Failing to save edition with: {str(e)}",
                                        type="error",
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
