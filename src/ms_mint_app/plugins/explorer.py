import datetime
import json
import platform
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

    def get_selected_tree_data(self, selected_files):
        """Generates the tree of selected files organized by folder."""
        tree_data = []

        # Group files by folder
        folders = {}
        for file_path in selected_files:
            path = Path(file_path)
            folder = path.parent.as_posix()
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(file_path)

        # Create tree nodes
        for folder, files in sorted(folders.items()):
            folder_node = {
                'key': folder,
                'title': f"{Path(folder).name} ({len(files)} files)",
                'icon': 'antd-folder',
                'checkable': True,
                'children': [
                    {
                        'key': file,
                        'title': Path(file).name,
                        'isLeaf': True,
                        'checkable': True,
                    }
                    for file in sorted(files)
                ]
            }
            tree_data.append(folder_node)

        return tree_data

    def outputs(self):
        return None

    def layout(self):
        return html.Div(
            [
                fac.AntdModal(
                    [
                        html.Div(
                            [
                                fac.AntdSpace(
                                    [
                                        html.Div(
                                            [
                                                # Breadcrumb for current path
                                                fac.AntdBreadcrumb(
                                                    id='current-path-modal',
                                                    items=[],
                                                    style={'margin': '10px 0', 'height': '24px'}
                                                ),
                                                # Table of files and directories
                                                html.Div(
                                                    [
                                                        fac.AntdTable(
                                                            id='file-table',
                                                            data=[],
                                                            columns=[
                                                                {
                                                                    'title': 'Name',
                                                                    'dataIndex': 'name',
                                                                    'align': 'left',
                                                                    'width': '70%',
                                                                    'renderOptions': {'renderType': 'link'},
                                                                },
                                                                {
                                                                    'title': 'T',
                                                                    'dataIndex': 'type',
                                                                    'width': '5%',
                                                                },
                                                                {
                                                                    'title': 'Modified',
                                                                    'dataIndex': 'modified',
                                                                    'width': '25%',
                                                                },
                                                                {
                                                                    'title': 'Fc',
                                                                    'dataIndex': 'file_count',
                                                                    'width': '10%',
                                                                },
                                                            ],
                                                            sortOptions={
                                                                'sortDataIndexes': ['modified', 'name', 'file_count']},
                                                            rowSelectionType='checkbox',
                                                            rowSelectionWidth=50,
                                                            size='small',
                                                            bordered=True,
                                                            pagination={
                                                                'pageSize': 50,
                                                                'showSizeChanger': True,
                                                                'pageSizeOptions': [20, 50, 100, 200],
                                                                'showQuickJumper': True,
                                                                'position': 'bottomCenter',
                                                                'size': 'small'
                                                            },
                                                            maxHeight=418,
                                                            showSorterTooltip=False,
                                                            locale='en-us',
                                                            enableCellClickListenColumns=['name']
                                                        )
                                                    ],
                                                    style={
                                                        'maxWidth': '800px', 'height': '524px'}
                                                ),
                                                # Filter by extension
                                                fac.AntdSelect(
                                                    id='selected-files-extensions',
                                                    size="small",
                                                    mode="multiple",
                                                    placeholder='Filter by extension',
                                                    style={'width': "100%", 'margin': '10px 0'},
                                                    locale="en-us",
                                                    allowClear=True,
                                                    disabled=True,
                                                ),
                                            ],
                                        ),
                                        fac.AntdFlex(
                                            [
                                                fac.AntdFlex(
                                                    [
                                                        html.Strong("Selected files"),
                                                        fac.AntdSpace(
                                                            [
                                                                fac.AntdButton(
                                                                    id='remove-marked-btn',
                                                                    size='small',
                                                                    danger=True,
                                                                    type='text',
                                                                    icon=fac.AntdIcon(
                                                                        icon='md-remove-circle-outline'),
                                                                ),
                                                                fac.AntdButton(
                                                                    id='clear-selection-btn',
                                                                    size='small',
                                                                    danger=True,
                                                                    type='primary',
                                                                    icon=fac.AntdIcon(icon='antd-delete'),
                                                                ),
                                                            ],
                                                            size='small'
                                                        ),
                                                    ],
                                                    justify='space-between',
                                                    align='center',
                                                    style={'margin': '10px 0'}
                                                ),
                                                html.Div(
                                                    [
                                                        fac.AntdTree(
                                                            id='selected-files-tree',
                                                            treeData=[],
                                                            checkable=True,
                                                            showIcon=True,
                                                            showLine=True,
                                                            defaultExpandAll=True,
                                                            style={'display': 'none'},
                                                        ),
                                                        fac.AntdEmpty(
                                                            description='No files selected',
                                                            id='files-selection-empty',
                                                            locale='en-us',
                                                            image='simple',
                                                            style={'display': 'block'}
                                                        )
                                                    ],
                                                    style={
                                                        'minHeight': '524px',
                                                        'maxHeight': '524px',
                                                        'maxWidth': '340px',
                                                        'width': '340px',
                                                        'flex': '1',
                                                        'overflow': 'auto'
                                                    }
                                                ),
                                                # File counter
                                                html.Div(
                                                    'Total: 0 files selected',
                                                    id='selection-counter',
                                                    style={'marginTop': '10px', 'fontSize': '12px',
                                                           'color': '#666'}
                                                ),
                                            ],
                                            vertical=True,
                                        ),
                                    ],
                                    align='start',
                                ),
                                # CPUs configuration
                                fac.AntdForm(
                                    [
                                        fac.AntdFormItem(
                                            fac.AntdInputNumber(
                                                id='processing-cpu-input',
                                                defaultValue=4,
                                                min=1,
                                                max=psutil.cpu_count(),
                                                size='small',
                                            ),
                                            label='CPUs:',
                                            tooltip='Number of CPUs to use for processing in parallel',
                                        ),
                                    ],
                                    layout='inline',
                                    id='processing-cpu-form',
                                    style={'marginTop': '20px'}
                                ),
                            ],
                            id='selection-container',
                        ),
                        # Processing progress container
                        html.Div(
                            [
                                html.H4("Processing files..."),
                                fac.AntdProgress(
                                    id='sm-processing-progress',
                                    percent=0,
                                ),
                                fac.AntdText(
                                    id='ms-files-progress-detail',
                                    type='secondary',
                                    style={'marginTop': '0.5rem', 'marginBottom': '0.75rem'},
                                ),
                                fac.AntdButton(
                                    'Cancel',
                                    id='cancel-ms-processing',
                                    danger=True,
                                    style={
                                        'alignText': 'center',
                                    },
                                ),
                            ],
                            id='processing-progress-container',
                            style={'display': 'none', 'textAlign': 'center', 'padding': '20px'},
                        ),
                    ],
                    id="selection-modal",
                    title='Load Files',
                    width=1200,
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
                    styles={'body': {'height': "70vh"}},
                ),
                dcc.Store(id="current-path-store", data=str(home_path)),
                dcc.Store(id="selected-files-store", data=[]),
                dcc.Store(id='processing-type-store', data={}),
                dcc.Store(id="processed-action-store"),
                dcc.Store(id='table-data-store', data=[]),
                dcc.Store(id='marked-for-removal', data=[]),
                dcc.Store(id='is-at-root', data=False),  # Para saber si estamos en el nivel de drives
            ]
        )

    def get_table_data(self, path, extensions, is_root=False):
        """Generates table data for the specified directory."""

        # If we are at the root of Windows, show the drives
        if is_root and platform.system() == 'Windows':
            import string
            table_data = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if Path(drive).exists():
                    table_data.append({
                        'key': drive,
                        'name': {
                            'content': f"{letter}:",
                        },
                        'type': 'Unidad',
                        'file_count': len([f for ext in extensions for f in Path(drive).glob(f'*{ext}')]) or '-',
                        'path': drive,
                        'is_dir': True,
                    })
            return table_data

        # Normal navigation
        current_path = Path(path)

        def safe_iterdir(p):
            try:
                return list(p.iterdir())
            except (PermissionError, OSError):
                return []

        def safe_is_dir(p):
            try:
                return p.is_dir()
            except (PermissionError, OSError):
                return False

        items = safe_iterdir(current_path)
        table_data = []

        # Sort directories and files
        dirs = sorted([i for i in items if safe_is_dir(i) and not i.name.startswith('.')], key=lambda x: x.name)
        files = sorted([i for i in items if i.is_file() and i.suffix in extensions and not i.name.startswith('.')],
                       key=lambda x: x.name)

        # Add directories (show all, without checking content)
        for folder in dirs:
            folder_stats = folder.stat()

            table_data.append({
                'key': folder.as_posix(),
                'name': {
                    'content': folder.name,
                },
                'type': fac.AntdIcon(icon='antd-folder'),
                'file_count': len([f for ext in extensions for f in folder.glob(f'*{ext}')]) or '-',
                'modified': datetime.datetime.fromtimestamp(folder_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                'path': folder.as_posix(),
                'is_dir': True,
            })

        # Add files
        for file in files:
            file_stats = file.stat()
            table_data.append({
                'key': file.as_posix(),
                'name': {
                    'content': file.name,
                },
                'type': fac.AntdIcon(icon='antd-file'),
                'file_count': '-',
                'modified': datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                'path': file.as_posix(),
                'is_dir': False,
            })

        return table_data

    def callbacks(self, app, fsc, cache, args=None):
        @app.callback(
            Output("selection-modal", "visible", allow_duplicate=True),
            Output("selection-modal", 'title'),
            Output('selected-files-extensions', 'options'),
            Output('selected-files-extensions', 'value'),
            Output('processing-type-store', 'data', allow_duplicate=True),
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
            Output('selected-files-store', 'data', allow_duplicate=True),
            Output('processing-type-store', 'data', allow_duplicate=True),
            Output('processed-action-store', 'data', allow_duplicate=True),
            Output('table-data-store', 'data', allow_duplicate=True),
            Output('marked-for-removal', 'data', allow_duplicate=True),
            Output('is-at-root', 'data', allow_duplicate=True),

            Output('selected-files-tree', 'treeData', allow_duplicate=True),
            Output('selection-counter', 'children', allow_duplicate=True),
            Output('file-table', 'selectedRowKeys', allow_duplicate=True),
            Output('selected-files-tree', 'checkedKeys', allow_duplicate=True),
            Output('selected-files-tree', 'style', allow_duplicate=True),
            Output('files-selection-empty', 'style', allow_duplicate=True),

            Input('selection-modal', 'visible'),
            prevent_initial_call=True
        )
        def on_modal_close(modal_visible):
            if modal_visible:
                raise PreventUpdate
            return ([], {}, None, [], [], False, [], 'Total: 0 files selected', [], [], {'display': 'none'},
                    {'display': 'block'})

        @app.callback(
            Output("current-path-modal", "items", allow_duplicate=True),
            Output("file-table", "data", allow_duplicate=True),
            Output('current-path-store', 'data', allow_duplicate=True),
            Output('table-data-store', 'data', allow_duplicate=True),

            Input('selection-modal', 'visible'),
            Input('current-path-modal', 'clickedItem'),
            Input('file-table', 'recentlyCellClickRecord'),

            State('current-path-store', 'data'),
            State('processing-type-store', 'data'),

            prevent_initial_call=True
        )
        def navigate_folders(modal_visible, bc_clicked_item, recentlyCellClickRecord, current_path, processing_type):
            if not modal_visible:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update

            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Determine the new path
            if 'current-path-modal' in trigger_id and bc_clicked_item:
                new_path = Path(bc_clicked_item.get('itemKey', current_path))
            elif 'file-table' in trigger_id and recentlyCellClickRecord:
                # Navigate only if is a directory
                if recentlyCellClickRecord.get('is_dir'):
                    new_path = Path(recentlyCellClickRecord['key'])
                else:
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            elif 'selection-modal' in trigger_id:
                new_path = Path(current_path or home_path)
            else:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update

            # Construct breadcrumb
            try:
                if platform.system() == 'Windows':
                    root = Path(new_path.drive + '\\')
                    all_paths = [root] + [p for p in new_path.parents if p != root][::-1]
                    if new_path != root:
                        all_paths.append(new_path)
                else:
                    all_paths = [Path('/')] + [p for p in new_path.parents if p != Path('/')][::-1]
                    if new_path != Path('/'):
                        all_paths.append(new_path)

                breadcrumb_items = []
                for i, path in enumerate(all_paths):
                    if i == 0:
                        breadcrumb_items.append(
                            {'title': 'root' if str(path) == '/' else str(path), 'key': str(path)})
                    else:
                        breadcrumb_items.append({'title': path.name, 'key': str(path)})
            except:
                breadcrumb_items = [{'title': str(new_path), 'key': str(new_path)}]

            # Generate table data
            extensions = processing_type.get('extensions', ['.csv'])
            new_table_data = self.get_table_data(new_path, extensions)

            return breadcrumb_items, new_table_data, str(new_path), new_table_data

        @app.callback(
            Output('selected-files-store', 'data', allow_duplicate=True),
            Output('selected-files-tree', 'treeData', allow_duplicate=True),
            Output('selected-files-tree', 'style', allow_duplicate=True),
            Output('files-selection-empty', 'style', allow_duplicate=True),
            Output('selection-counter', 'children', allow_duplicate=True),
            Output('file-table', 'selectedRowKeys', allow_duplicate=True),
            Output('marked-for-removal', 'data', allow_duplicate=True),
            Output('selected-files-tree', 'checkedKeys', allow_duplicate=True),

            Input('file-table', 'selectedRowKeys'),
            Input('file-table', 'recentlyCellClickRecord'),
            Input('clear-selection-btn', 'nClicks'),
            Input('remove-marked-btn', 'nClicks'),
            Input('selected-files-tree', 'checkedKeys'),

            State('selected-files-store', 'data'),
            State('processing-type-store', 'data'),
            State('table-data-store', 'data'),
            State('marked-for-removal', 'data'),

            prevent_initial_call=True
        )
        def update_selection(selectedRowKeys, recentlyCellClickRecord, clear_clicks, remove_clicks, tree_checked_keys,
                             current_selection, processing_type, table_data, marked_for_removal):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            trigger_action = ctx.triggered[0]['prop_id'].split('.')[-1]

            selected_files = set(current_selection or [])
            marked = set(marked_for_removal or [])

            if trigger_id == 'clear-selection-btn':
                # Clear all selection
                selected_files = set()
                new_selected_keys = []
                marked = set()
                tree_checks = []

            elif trigger_id == 'file-table':
                # Update selection based on selected rows
                extensions = processing_type.get('extensions', [])
                if trigger_action == 'recentlyCellClickRecord':
                    if not recentlyCellClickRecord.get('is_dir'):
                        selected_files.add(recentlyCellClickRecord['key'])
                else:
                    if selectedRowKeys:
                        # Process new selections
                        for row_key in selectedRowKeys:
                            # Search for the item in the table
                            for item in table_data:
                                if item['key'] == row_key:
                                    if item['is_dir']:
                                        # It's a folder, add all its files recursively
                                        folder_path = Path(item['path'])
                                        for ext in extensions:
                                            selected_files.update(str(f) for f in folder_path.rglob(f"*{ext}"))
                                    else:
                                        # It's a file
                                        selected_files.add(item['path'])
                                        break
                new_selected_keys = selectedRowKeys
                tree_checks = list(marked)

            elif trigger_id == 'selected-files-tree':
                # Update marked items for removal
                marked = set(tree_checked_keys or [])
                new_selected_keys = dash.no_update
                tree_checks = list(marked)

            elif trigger_id == 'remove-marked-btn':
                # Remove marked items
                items_to_remove = set()
                selectedRowKeysToDelete = set()
                for item in marked:
                    if item in selectedRowKeys:
                        selectedRowKeysToDelete.add(item)
                    path = Path(item)
                    if path.is_dir() or item in [Path(f).parent.as_posix() for f in selected_files]:
                        # It's a folder, remove all its files
                        items_to_remove.update(f for f in selected_files if Path(f).parent.as_posix() == item)
                    else:
                        # It's a file
                        items_to_remove.add(item)

                selected_files -= items_to_remove
                marked = set()
                new_selected_keys = list(set(selectedRowKeys) - selectedRowKeysToDelete)
                tree_checks = []
            else:
                raise PreventUpdate

            selected_list = sorted(list(selected_files))

            # generate selected tree data
            tree_data = self.get_selected_tree_data(selected_list)

            # styles
            if selected_list:
                tree_style = {'display': 'flex'}
                empty_style = {'display': 'none'}
            else:
                tree_style = {'display': 'none'}
                empty_style = {'display': 'block'}

            counter = f"Total: {len(selected_list)} files selected"
            return selected_list, tree_data, tree_style, empty_style, counter, new_selected_keys, list(
                marked), tree_checks

        @app.callback(
            Output('notifications-container', "children", allow_duplicate=True),
            Output('processed-action-store', 'data', allow_duplicate=True),
            Output('selection-modal', 'visible', allow_duplicate=True),

            Input('selection-modal', 'okCounts'),
            State('processing-type-store', 'data'),
            State("selected-files-store", "data"),
            State('processing-cpu-input', 'value'),
            State("wdir", "data"),
            background=True,
            running=[
                (Output('processing-progress-container', 'style'),
                 {
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
            progress=[
                Output("sm-processing-progress", "percent"),
                Output("ms-files-progress-detail", "children"),
            ],
            cancel=[
                Input('cancel-ms-processing', 'nClicks')
            ],
            prevent_initial_call=True
        )
        def background_processing(set_progress, okCounts, processing_type, selected_files_list,
                                  cpu_input, wdir):
            if not okCounts or not selected_files_list:
                raise PreventUpdate

            # Convert list to dictionary grouped by folder for compatibility
            selected_files = {}
            for file_path in selected_files_list:
                folder = str(Path(file_path).parent)
                if folder not in selected_files:
                    selected_files[folder] = []
                selected_files[folder].append(file_path)

            # defaults for branches
            failed_targets = []
            stats = {}

            def progress_adapter(percent, detail=""):
                if set_progress:
                    set_progress((percent, detail or ""))

            if processing_type['type'] == "ms-files":
                total_processed, failed_files = process_ms_files(wdir, progress_adapter, selected_files, cpu_input)
                message = "MS Files processed"
            elif processing_type['type'] == "metadata":
                total_processed, failed_files = process_metadata(wdir, progress_adapter, selected_files)
                message = "Metadata processed"
            else:
                total_processed, failed_files, failed_targets, stats = process_targets(wdir, progress_adapter, selected_files)
                message = "Targets processed"

            duplicate_targets = stats.get("duplicate_peak_labels", 0) if processing_type['type'] == "targets" else 0
            failed_targets_count = len(failed_targets) if processing_type['type'] == "targets" else 0

            if total_processed:
                details = []
                if failed_files:
                    details.append(f"{len(failed_files)} file(s) failed")
                if failed_targets_count:
                    details.append(f"{failed_targets_count} target row(s) failed")
                if duplicate_targets:
                    details.append(f"{duplicate_targets} duplicate target label(s) deduplicated")

                if details:
                    description = f"Processed {total_processed} files; " + "; ".join(details) + ". See logs for details."
                    mss_type = "warning" if not failed_files else "warning"
                else:
                    description = f"Successfully processed {total_processed} files"
                    mss_type = "success"
            else:
                description = f"Failed processing {len(failed_files)} files. See logs for details."
                mss_type = "error"
            notification = fac.AntdNotification(message=message,
                                                description=description,
                                                type=mss_type,
                                                duration=3,
                                                placement='bottom',
                                                showProgress=True,
                                                style={"maxWidth": 420, "width": "420px"})
            processed_action_store = {'action': 'processing', 'status': 'success'}

            return notification, processed_action_store, False
