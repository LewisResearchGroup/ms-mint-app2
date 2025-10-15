import json
from collections import Counter
from pathlib import Path

import dash
import feffery_antd_components as fac
import psutil
from dash import Output, Input, State, ALL, html, dcc
from dash.exceptions import PreventUpdate

from ms_mint_app.tools import process_ms_files, process_metadata, process_targets

home_path = Path.home()


class FileExplorer:
    """
    FileExplores component. Creates a Modal to select files from the system.
    """

    def outputs(self):
        return None

    def layout(self):
        return html.Div(
            [
                fac.AntdModal(
                    [
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
                                    size='small',
                                    pagination=False,
                                    style={'flexGrow': 4, 'minHeight': '400px'},
                                ),
                                fac.AntdFlex(
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

                                        html.Div(
                                            [
                                                fac.AntdEmpty(
                                                    description='No files selected',
                                                    id='files-selection-empty',
                                                    style={'height': '100%'}
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
                                            style={'height': 180, 'marginBottom': 10}
                                        ),
                                        fac.AntdForm(
                                            [
                                                fac.AntdFormItem(
                                                    fac.AntdInputNumber(
                                                        id='processing-cpu-input',
                                                        placeholder='New workspace name',
                                                        defaultValue=4,
                                                        min=1,
                                                        max=psutil.cpu_count(),
                                                    ),
                                                    label='Number of CPUs:',
                                                ),
                                            ],
                                            layout='inline',
                                            id='processing-cpu-form',
                                        ),
                                    ],
                                    id="selected-files-area",
                                    vertical=True,
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
                            id='processing-progress-container',
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

                dcc.Store(id="selected-folder-path"),
                dcc.Store(id="selected-files", data={}),
                dcc.Store(id='processing-type-store', data={}),
                dcc.Store(id="processed-action-store"),
            ]
        )

    def get_content_list(self, path, extensions):
        """
        Generate content list for the directory
        """
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
                    'style': {'pointerEvents': 'none'},
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

    def callbacks(self, app, fsc, cache, args=None):
        @app.callback(
            Output("selection-modal", "visible", allow_duplicate=True),
            Output("selection-modal", 'title'),
            Output('selected-files-extensions', 'options'),
            Output('selected-files-extensions', 'value'),
            Output('processing-type-store', 'data'),
            Output('processing-cpu-form', 'style'),

            Input({'action': 'file-explorer', 'type': ALL}, 'nClicks'),
            prevent_initial_call=True
        )
        def open_selection_modal(n_clicks):

            if n_clicks is None or not any(n_clicks):
                raise PreventUpdate

            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
            prop_data = json.loads(prop_id)

            if prop_data['type'] == "ms-files":
                title = "Load MS Files"
                file_extensions = [".mzXML"]
                style = {'display': 'block'}
            elif prop_data['type'] == "metadata":
                title = "Load Metadata"
                file_extensions = [".csv"]
                style = {'display': 'none'}
            else:
                title = "Load Targets"
                file_extensions = [".csv"]
                style = {'display': 'none'}


            processing_type_store = {'type': prop_data['type'], 'extensions': file_extensions}
            return True, title, file_extensions, file_extensions, processing_type_store, style

        @app.callback(
            Output('selected-folder-path', 'data', allow_duplicate=True),
            Output('selected-files', 'data', allow_duplicate=True),
            Output('selected-files-display', 'children', allow_duplicate=True),
            Output('files-selection-empty', 'style', allow_duplicate=True),
            Output("sm-processing-progress", "percent"),

            Input('selection-modal', 'visible'),
            prevent_initial_call=True
        )
        def on_modal_close(modal_visible):
            if modal_visible:
                raise PreventUpdate
            return None, {}, [], {'height': 180, 'display': 'block'}, 0

        # Callback para navegar por carpetas
        @app.callback(
            Output("current-path-modal", "items"),
            Output("dir-content-table", "data"),
            Output('selected-folder-path', 'data', allow_duplicate=True),

            Input('selection-modal', 'visible'),
            Input('dir-content-table', 'nClicksButton'),
            State('dir-content-table', 'clickedCustom'),

            Input('current-path-modal', 'clickedItem'),
            State('selected-folder-path', 'data'),
            State('processing-type-store', 'data'),

            prevent_initial_call=True
        )
        def navigate_folders(modal_visible, nClicksButton, clickedCustom, bc_clicked_item, current_path,
                             processing_type):
            ctx = dash.callback_context
            if not ctx.triggered or not modal_visible:
                return dash.no_update, dash.no_update, dash.no_update

            prop_id = ctx.triggered[0]['prop_id']

            # Extraer el subcomponent del prop_id
            if isinstance(prop_id, str) and 'selection-modal' in prop_id:
                current_modal_path = Path(current_path or home_path)
            elif isinstance(prop_id, str) and 'current-path-modal' in prop_id:
                current_modal_path = Path(bc_clicked_item['itemKey'] or home_path)
            elif clickedCustom and clickedCustom.get('is_link'):
                current_modal_path = Path(clickedCustom['path'] or home_path)
            else:
                raise PreventUpdate

            all_paths = [path for path in reversed(current_modal_path.parents) if path >= home_path] + [
                current_modal_path]
            current_path_items = []
            for i, path in enumerate(all_paths):
                if i == 0:
                    current_path_items.append({'title': str(path), 'key': str(path)})
                else:
                    current_path_items.append({'title': path.name, 'key': str(path)})

            content_items = self.get_content_list(current_modal_path, processing_type['extensions'])
            return current_path_items, content_items, str(current_modal_path)

        @app.callback(
            Output("selected-files-display", "children"),
            Output('files-selection-empty', 'style'),
            Output('selected-files', 'data'),

            Input("dir-content-table", "nClicksButton"),
            State('dir-content-table', 'clickedCustom'),
            State('selected-files', 'data'),
            State('processing-type-store', 'data'),

            prevent_initial_call=True
        )
        def add_selection(nClicksButton, clickedCustom, selected_files, processing_type):
            if not nClicksButton or clickedCustom.get('is_link'):
                raise PreventUpdate

            unique_selected_files = {k: set(v) for k, v in selected_files.items()}

            if clickedCustom['type'] == 'folder':
                folder_path = clickedCustom['path']
                files = [file.as_posix() for ext in processing_type['extensions']
                         for file in Path(clickedCustom['path']).rglob(f"*{ext}")]
            else:
                folder_path = Path(clickedCustom['path']).parent.as_posix()
                files = [clickedCustom['path']]

            if folder_path in unique_selected_files:
                unique_selected_files[folder_path].update(files)
            else:
                unique_selected_files[folder_path] = set(files)

            children = [
                fac.AntdTag(
                    content=f"{folder_path}: {len(folder_content)} files",
                    closeIcon=True,
                    id={'type': 'tag-files', 'path': folder_path},
                    style={
                        'fontSize': 14,
                        'display': 'flex',
                        'alignItems': 'center',
                    },
                )
                for folder_path, folder_content in unique_selected_files.items() if len(folder_content)
            ]

            if not children:
                return dash.no_update, dash.no_update

            # Convertir sets a listas para serialización
            selected_files = {k: list(v) for k, v in unique_selected_files.items()}
            return children, {'height': 180, 'display': 'none'}, selected_files

        @app.callback(
            Output("selected-files-display", "children", allow_duplicate=True),
            Output('files-selection-empty', 'style', allow_duplicate=True),
            Output('selected-files', 'data', allow_duplicate=True),

            Input({'type': 'tag-files', 'path': ALL}, "closeCounts"),
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
            style = {'display': 'none'} if children else {'height': 180, 'display': 'block'}
            return children, style, selected_files

        @app.callback(
            Output('notifications-container', "children", allow_duplicate=True),
            Output('processed-action-store', 'data', allow_duplicate=True),
            Output('selection-modal', 'visible', allow_duplicate=True),

            Input('selection-modal', 'okCounts'),
            State('processing-type-store', 'data'),
            State("selected-files", "data"),
            State('processing-cpu-input', 'value'),
            State("wdir", "data"),
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
                (Output("selection-container", "style"), {'display': 'none'}, {'display': 'block'}),

                (Output('selection-modal', 'confirmAutoSpin'), True, False),
                (Output('selection-modal', 'cancelButtonProps'), {'disabled': True},
                 {'disabled': False}),
                (Output('selection-modal', 'confirmLoading'), True, False),
            ],
            progress=[Output("sm-processing-progress", "percent")],
            cancel=[
                Input('cancel-ms-processing', 'nClicks')
            ],
            prevent_initial_call=True
        )
        def background_processing(set_progress, okCounts, processing_type, selected_files, cpu_input, wdir):
            if not okCounts or not selected_files or not dash.callback_context.triggered:
                raise PreventUpdate

            if processing_type['type'] == "ms-files":
                total_processed, failed_files = process_ms_files(wdir, set_progress, selected_files, cpu_input)
                message = "MS Files processed"
            elif processing_type['type'] == "metadata":
                total_processed, failed_files = process_metadata(wdir, set_progress, selected_files)
                message = "Metadata processed"
            else:
                total_processed, failed_files = process_targets(wdir, set_progress, selected_files)
                message = "Targets processed"

            if total_processed:
                if failed_files:
                    f_map = Counter([ff.values() for ff in failed_files])
                    description = (f"Successful processed {total_processed} files with {len(failed_files)} failed "
                                   f" {list(f_map.items())}")
                    mss_type = "warning"
                else:
                    description = f"Successful processed {total_processed} files"
                    mss_type = "success"
            else:
                f_map = Counter([ff.values() for ff in failed_files])
                description = f"Failed processing {len(failed_files)} files {list(f_map.items())}"
                mss_type = "error"
            notification = fac.AntdNotification(message=message,
                                                description=description,
                                                type=mss_type,
                                                duration=3,
                                                placement='bottom',
                                                showProgress=True)
            processed_action_store = {'action': 'processing', 'status': 'success'}

            return notification, processed_action_store, False
