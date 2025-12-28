import re
import shutil
import json
import os
import logging
from pathlib import Path

import dash
import feffery_antd_components as fac
import pandas as pd
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..duckdb_manager import duckdb_connection_mint, duckdb_connection
from ..logging_setup import activate_workspace_logging, deactivate_workspace_logging
from ..plugin_interface import PluginInterface

_label = "Workspaces"
pattern = re.compile(r"^[A-Za-z0-9_]+$")



logger = logging.getLogger(__name__)

class WorkspacesPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 0
        logger.info(f'Initiated {_label} plugin')

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)

    def outputs(self):
        return _outputs


_layout = html.Div(
    [
        dcc.Store(id="ws-action-store"),
        fac.AntdFlex(
            [
                fac.AntdTitle('Workspaces', level=4, style={'margin': '0'}),
                fac.AntdIcon(
                    id='workspace-tour-icon',
                    icon='pi-info',
                    style={"cursor": "pointer", 'paddingLeft': '10px'},
                )
            ],
            align='center'
        ),
        
        # --- Configuration Section ---
        fac.AntdFlex(
            [
                fac.AntdText("Current Data Directory:", strong=True),
                fac.AntdTooltip(
                    title="This is the global root directory where all your workspaces and data are stored.",
                    children=fac.AntdIcon(icon="antd-question-circle", style={"color": "#888", "marginRight": "2.5px"})
                ),
                fac.AntdText("Loading...", id="ws-current-data-dir-text", style={"marginRight": "10px", "fontWeight": "bold"}),
                fac.AntdTooltip(
                    fac.AntdButton("Change Location", id="ws-change-data-dir-btn", size="small"),
                    title="Select a different folder on your computer to store your MINT workspaces and data.",
                    placement="bottom"
                )
            ],
            style={"marginBottom": "1px", "marginTop": "100px"},
            gap=10,
            align="center"
        ),
        # -----------------------------
        # -----------------------------

        html.Div([
            fac.AntdTable(
                id='ws-table',
                columns=[
                    {'title': 'Name', 'dataIndex': 'name', 'align': 'left', 'width': '30%'},
                    {'title': 'Description', 'dataIndex': 'description', 'align': 'left', 'editable': True,
                     'width': '50%'},
                    {'title': 'Created at', 'dataIndex': 'created_at', 'align': 'center', 'width': '10%'},
                    {'title': 'Last Activity', 'dataIndex': 'last_activity', 'align': 'center', 'width': '10%'},
                ],
                filterOptions={
                    'name': {'filterSearch': True},
                },
                sortOptions={'sortDataIndexes': ['name', 'last_activity', 'created_at']},
                footer=[
                    fac.AntdFlex([
                        fac.AntdTooltip(
                            fac.AntdButton('Create Workspace', id='ws-create', icon=fac.AntdIcon(icon='antd-plus')),
                            title="Create a new, empty workspace.",
                            placement="top"
                        ),
                        fac.AntdTooltip(
                            fac.AntdButton('Delete Workspace', id='ws-delete', danger=True, icon=fac.AntdIcon(icon='antd-minus')),
                            title="Permanently delete the selected workspace and all its data.",
                            placement="top"
                        )],
                        justify='space-between'
                    )
                ],
                pagination={'pageSize': 10, 'hideOnSinglePage': True},
                locale='en-us',
                rowSelectionType='radio',
                size='small',
            )],
            style={"marginTop": "2rem"},
        ),
        fac.AntdModal(
            [
                fac.AntdForm(
                    [
                        fac.AntdFormItem(
                            fac.AntdInput(id='ws-create-input-name', placeholder='New workspace name', value=None),
                            label='Name:',
                            hasFeedback=True,
                            id='ws-create-form-item'
                        ),
                        fac.AntdFormItem(
                            fac.AntdInput(id='ws-create-input-description', placeholder='Project description.',
                                          value=None, mode='text-area'),
                            label='Description:',
                        ),
                    ],
                ),
            ],
            title='Create Workspace',
            id='ws-create-modal',
            renderFooter=True,
            okText='Create',
            locale='en-us',
            okButtonProps={
                'disabled': True
            }
        ),
        fac.AntdModal([
            html.Div(fac.AntdText("This will delete all files and results in the selected workspace.")),

            html.Div(fac.AntdText('Are you sure you want to delete this workspace?', strong=True))],
            title="Delete Workspace",
            id="ws-delete-modal",
                okText="Delete",
                renderFooter=True,
                locale='en-us',
                okButtonProps={
                    'danger': True
                }
        ),
        
        # --- Change Data Directory Modal ---
        fac.AntdModal(
            [
                fac.AntdText("Enter the absolute path for the new Global Data Directory."),
                fac.AntdAlert(
                    message="Warning: Changing this will reload the application with the new directory. Existing workspaces in the old directory will remain there but won't be visible until you switch back.",
                    type="warning",
                    showIcon=True,
                    style={"marginBottom": "10px", "marginTop": "10px"}
                ),
                fac.AntdForm(
                    [
                        fac.AntdFormItem(
                            fac.AntdInput(id='ws-input-data-dir', placeholder='/absolute/path/to/MINT', value=None),
                            label='New Path:',
                            hasFeedback=True,
                            id='ws-data-dir-form-item'
                        ),
                    ]
                )
            ],
            title='Change Global Data Directory',
            id='ws-change-data-dir-modal',
            renderFooter=True,
            okText='Save & Reload',
            locale='en-us',
        ),
        # -----------------------------------

        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'Use this short tutorial to pick, create, and clean up workspaces.',
                },
                {
                    'title': 'Pick a workspace',
                    'description': 'Click a row to load the workspace; sort or filter the table if you have many.',
                    'targetSelector': '#ws-table'
                },
                {
                    'title': 'Create a new one',
                    'description': 'Use “Create Workspace” to start a fresh project with its own files/results.',
                    'targetSelector': '#ws-create'
                },
                {
                    'title': 'Clean up',
                    'description': 'Delete the selected workspace (and its files/results) when you no longer need it.',
                    'targetSelector': '#ws-delete'
                },
                {
                    'title': 'Tip',
                    'description': 'Changes are saved automatically after you switch workspaces.',
                    'targetSelector': '#ws-table'
                },
            ],
            id='workspace-tour',
            open=False,
            current=0,
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Click here for help',
                    # 'description': 'Click the info icon to open a quick tour of Workspaces.',
                    'targetSelector': '#workspace-tour-icon',
                },
            ],
            mask=False,
            placement='rightTop',
            open=False,
            current=0,
            id='workspace-tour-hint',
            className='targets-tour-hint',
            style={
                'background': '#ffffff',
                'border': '0.5px solid #1677ff',
                'boxShadow': '0 6px 16px rgba(0,0,0,0.15), 0 0 0 1px rgba(22,119,255,0.2)',
                'opacity': 1,
            },
        ),
        dcc.Store(id='workspace-tour-hint-store', data={'open': True}, storage_type='session'),
    ]
)

_outputs = html.Div(
    id="ws-outputs",
    children=[

    ],
)



def _create_ws_input_validation(value, tmpdir):
    if value is None:
        raise PreventUpdate

    if not tmpdir:
        okButtonProps = {'disabled': True}
        validateStatus = 'error'
        help = 'Workspace path not available'
    elif value is not None and bool(pattern.match(value)):
        with duckdb_connection_mint(tmpdir) as mint_conn:
            if mint_conn is None:
                okButtonProps = {'disabled': True}
                validateStatus = 'error'
                help = 'Cannot open workspace database'
            else:
                ws_df = mint_conn.execute("SELECT * FROM workspaces WHERE name = ?", (value,)).df()
                if ws_df.empty:
                    okButtonProps = {'disabled': False}
                    validateStatus = 'success'
                    help = None
                else:
                    okButtonProps = {'disabled': True}
                    validateStatus = 'error'
                    help = 'Workspace already exists!'
    else:
        okButtonProps = {'disabled': True}
        validateStatus = 'error'
        help = 'Workspace name can only contain: a-z, A-Z, 0-9 and _'
    return validateStatus, help, okButtonProps


def _create_workspace(okCounts, tmpdir, ws_name, ws_description):
    if not okCounts:
        raise PreventUpdate
    if not tmpdir:
        raise PreventUpdate
    with duckdb_connection_mint(tmpdir) as mint_conn:
        if mint_conn is None:
            raise PreventUpdate
        previous_active = mint_conn.execute("SELECT key FROM workspaces WHERE active = true").fetchone()

        key = mint_conn.execute("INSERT INTO workspaces (name, description, active, created_at, last_activity) "
                                "VALUES (?, ?, true, NOW(), NOW()) RETURNING key", (ws_name, ws_description)).fetchone()
        if previous_active:
            mint_conn.execute("UPDATE workspaces SET active = false WHERE key = ?", (previous_active[0],))
        if key:
            ws_path = Path(tmpdir, 'workspaces', str(key[0]))
            ws_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created workspace: {ws_name} at {ws_path}")

    return 'create', None, None, None


def _delete_workspace(okCounts, tmpdir, selectedRowKeys):
    if not okCounts:
        raise PreventUpdate
    if not tmpdir or not selectedRowKeys:
        raise PreventUpdate

    ws_key = selectedRowKeys[0]

    with duckdb_connection_mint(tmpdir) as mint_conn:
        try:
            mint_conn.execute("BEGIN")
            ws_row = mint_conn.execute(
                "SELECT name FROM workspaces WHERE key = ?",
                (ws_key,)
            ).fetchone()
            if not ws_row:
                mint_conn.execute("ROLLBACK")
                return fac.AntdNotification(
                    message="Workspace not found",
                    type="error",
                    duration=4,
                    placement='bottom',
                    showProgress=True,
                    stack=True
                ), {'type': 'delete', 'status': 'error'}

            next_active = mint_conn.execute(
                "SELECT key FROM workspaces WHERE key != ? ORDER BY last_activity DESC LIMIT 1",
                (ws_key,)
            ).fetchone()

            ws_path = Path(tmpdir, 'workspaces', str(ws_key))
            deactivate_workspace_logging()

            def _onerror(func, p, exc_info):
                import os
                try:
                    os.chmod(p, 0o700)
                    func(p)
                except Exception:
                    raise

            try:
                shutil.rmtree(ws_path, onerror=_onerror)
            except Exception as fs_err:
                mint_conn.execute("ROLLBACK")
                return fac.AntdNotification(
                    message="Failed to delete workspace files",
                    description=str(fs_err),
                    type="error",
                    duration=5,
                    placement='bottom',
                    showProgress=True,
                    stack=True
                ), {'type': 'delete', 'status': 'error'}

            mint_conn.execute("DELETE FROM workspaces WHERE key = ?", (ws_key,))
            if next_active:
                mint_conn.execute("UPDATE workspaces SET active = true WHERE key = ?", (next_active[0],))
            mint_conn.execute("COMMIT")

            ws_name = ws_row[0]
            logger.info(f"Deleted workspace: {ws_name} (key: {ws_key})")
            return fac.AntdNotification(message=f"Workspace {ws_name} deleted.",
                                        type="success",
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        stack=True), {'type': 'delete', 'status': 'success'}
        except Exception as e:
            logger.error(f"Error deleting workspace {ws_key}: {e}", exc_info=True)
            mint_conn.execute("ROLLBACK")
            raise

    return dash.no_update, {'type': 'delete', 'status': 'error'}


def _ws_activate(selectedRowKeys, tmpdir, ws_action):
    if not selectedRowKeys:
        return dash.no_update, '', '', None

    ws_key = selectedRowKeys[0]

    with duckdb_connection_mint(tmpdir) as mint_conn:
        # Check if this workspace is already active
        is_active = mint_conn.execute("SELECT active FROM workspaces WHERE key = ?", (ws_key,)).fetchone()
        already_active = is_active and is_active[0]

        if already_active:
             # Just update activity time
            mint_conn.execute("UPDATE workspaces SET last_activity = NOW() WHERE key = ?", (ws_key,))
            name = mint_conn.execute("SELECT name FROM workspaces WHERE key = ?", (ws_key,)).fetchone()
            notification = dash.no_update
        else:
            # Full activation switch
            mint_conn.execute("UPDATE workspaces SET active = false WHERE key != ?", (ws_key,))
            name = mint_conn.execute(
                "UPDATE workspaces SET active = true, last_activity = NOW() WHERE key = ? RETURNING name",
                (ws_key,)
            ).fetchone()
            
            if name:
                ws_name = name[0]
                # Only log the explicit activation switch here
                logger.info(f"Activated workspace: {ws_name} (key: {ws_key})")
                notification = fac.AntdNotification(message=f"Workspace {ws_name} activated.",
                                                    type="success",
                                                    duration=3,
                                                    placement='bottom',
                                                    showProgress=True,
                                                    stack=True)
            else:
                 notification = dash.no_update

        if name:
            ws_name = name[0]
            wdir = Path(tmpdir, 'workspaces', ws_key)
        else:
            return dash.no_update, '', '', None

    # This will now be silent if the handler is already attached (idempotent)
    activate_workspace_logging(wdir, workspace_name=ws_name)

    return notification, ws_name, wdir.as_posix(), wdir.as_posix()


def _save_ws_table_on_edit(row_edited, column_edited, tmpdir):
    ctx = dash.callback_context
    # if not ctx.triggered:
    #     raise PreventUpdate

    if row_edited is None or column_edited is None:
        raise PreventUpdate

    if not tmpdir:
        raise PreventUpdate

    allowed_columns = {"description"}
    if column_edited not in allowed_columns:
        return fac.AntdNotification(
            message="Edit not allowed",
            description=f"Column '{column_edited}' cannot be edited.",
            type="error",
            duration=4,
            placement='bottom',
            showProgress=True,
            stack=True
        ), dash.no_update

    ws_key = row_edited.get('key')
    if not ws_key:
        raise PreventUpdate

    logger.info(f"Updating workspace {ws_key}: {column_edited} = {row_edited[column_edited]}")

    try:
        with duckdb_connection_mint(tmpdir, workspace=ws_key) as mint_conn:
            if mint_conn is None:
                raise PreventUpdate
            query = f"UPDATE workspaces SET {column_edited} = ?, last_activity = NOW() WHERE key = ?"
            mint_conn.execute(query, [row_edited[column_edited], ws_key])
        return fac.AntdNotification(message="Successfully edition saved",
                                    type="success",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    ), {'type': 'edit'}
    except Exception as e:
        logger.error(f"Failed to save edit for workspace {ws_key}: {e}", exc_info=True)
        return fac.AntdNotification(message="Failed to save edition",
                                    description=f"Failing to save edition with: {str(e)}",
                                    type="error",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    ), dash.no_update


def _save_new_data_dir(okCounts, new_path):
    if not okCounts:
        raise PreventUpdate
    
    if not new_path or not new_path.strip():
         return dash.no_update, fac.AntdNotification(message="Please enter a valid path.", type="error"), True
    
    new_path = os.path.expanduser(new_path)
    new_path_obj = Path(new_path)

    if not new_path_obj.is_absolute():
        return dash.no_update, fac.AntdNotification(message="Path must be absolute.", type="error"), True
    
    # Try to create directory if it doesn't exist to verify permissions
    try:
        new_path_obj.mkdir(parents=True, exist_ok=True)
        # Create .cache inside
        (new_path_obj / ".cache").mkdir(exist_ok=True)
    except OSError as e:
         return dash.no_update, fac.AntdNotification(message=f"Permission denied or invalid path: {e}", type="error"), True

    # Update Config
    config_path = os.path.expanduser(os.environ.get("MINT_CONFIG_PATH") or "~/.mint_config.json")
    try:
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        else:
            cfg = {}
        
        cfg["data_dir"] = str(new_path_obj)
        
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
            
    except Exception as e:
         return dash.no_update, fac.AntdNotification(message=f"Failed to update config file: {e}", type="error"), True

    # Update Environment Variable for this session (best effort)
    os.environ["MINT_DATA_DIR"] = str(new_path_obj)
    
    # Update global TMPDIR in app.py to ensure page refreshes pick up the new path
    try:
        import ms_mint_app.app as app_module
        app_module.TMPDIR = new_path_obj
        app_module.CACHEDIR = new_path_obj / ".cache"
        logger.info(f"Updated app.TMPDIR to {app_module.TMPDIR}")
    except Exception as e:
        logger.warning(f"Failed to update app.TMPDIR: {e}")

    new_user_path = new_path_obj / "Local"
    new_user_path.mkdir(parents=True, exist_ok=True)
    
    with duckdb_connection_mint(str(new_user_path)) as mint_conn:
        logger.info(f"Initialized DB in {new_user_path}")

    return str(new_user_path), fac.AntdNotification(message="Global Data Directory updated successfully!", type="success"), False


def callbacks(app, fsc, cache):
    @app.callback(
        Output('workspace-tour', 'current'),
        Output('workspace-tour', 'open'),
        Input('workspace-tour-icon', 'nClicks'),
        prevent_initial_call=True,
    )
    def workspace_tour_open(n_clicks):
        return 0, True

    @app.callback(
        Output('workspace-tour-hint', 'open'),
        Output('workspace-tour-hint', 'current'),
        Input('workspace-tour-hint-store', 'data'),
    )
    def workspace_hint_sync(store_data):
        if not store_data:
            raise PreventUpdate
        return store_data.get('open', True), 0

    @app.callback(
        Output('workspace-tour-hint-store', 'data'),

        Input('workspace-tour-icon', 'nClicks'),
        State('workspace-tour-hint-store', 'data'),
        prevent_initial_call=True,
    )
    def workspace_hide_hint(n_clicks, store_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'workspace-tour-icon':
            return {'open': False}

        return store_data or {'open': True}

    @app.callback(
        Output('ws-create-form-item', 'validateStatus'),
        Output('ws-create-form-item', 'help'),
        Output('ws-create-modal', 'okButtonProps'),

        Input('ws-create-input-name', 'value'),
        State("tmpdir", "data"),
        prevent_initial_call=True
    )
    def create_ws_input_validation(value, tmpdir):
        return _create_ws_input_validation(value, tmpdir)

    @app.callback(
        Output('ws-action-store', 'data', allow_duplicate=True),
        Output('ws-create-input-name', 'value'),
        Output('ws-create-input-description', 'value'),
        Output('ws-create-form-item', 'validateStatus', allow_duplicate=True),

        Input('ws-create-modal', 'okCounts'),
        State("tmpdir", "data"),
        State('ws-create-input-name', 'value'),
        State('ws-create-input-description', 'value'),
        prevent_initial_call=True
    )
    def create_workspace(okCounts, tmpdir, ws_name, ws_description):
        return _create_workspace(okCounts, tmpdir, ws_name, ws_description)

    @app.callback(
        Output('ws-create-modal', 'visible'),
        Input("ws-create", "nClicks")
    )
    def create_workspace_modal(nClicks):
        if nClicks is None:
            raise PreventUpdate
        return True

    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('ws-action-store', 'data', allow_duplicate=True),

        Input('ws-delete-modal', 'okCounts'),
        State("tmpdir", "data"),
        State('ws-table', 'selectedRowKeys'),
        prevent_initial_call=True
    )
    def delete_workspace(okCounts, tmpdir, selectedRowKeys):
        return _delete_workspace(okCounts, tmpdir, selectedRowKeys)

    @app.callback(
        Output('ws-delete-modal', 'visible'),
        Input("ws-delete", "nClicks")
    )
    def delete_workspace_modal(nClicks):
        if nClicks is None:
            raise PreventUpdate
        return True

    @app.callback(
        Output("ws-table", "data"),
        Output("ws-table", "expandedRowKeyToContent"),
        Output("ws-table", "selectedRowKeys"),

        Input('section-context', 'data'),
        Input('ws-action-store', 'data'),
        Input("tmpdir", "data"), # Changed from State to Input to auto-update
    )
    def ws_table(section_context, ws_action, tmpdir):

        if section_context and  section_context['page'] != 'Workspaces':
            raise PreventUpdate

        with duckdb_connection_mint(tmpdir) as mint_conn:
            data = mint_conn.execute("SELECT * FROM workspaces ORDER BY last_activity DESC").df()
            logger.debug(f"Loaded workspace table data: {len(data)} rows")

            if not data.empty:
                data["key"] = data["key"].astype(str)

            cols = ['created_at', 'last_activity']
            data[cols] = data[cols].apply(lambda col: col.dt.strftime("%y-%m-%d %H:%M:%S"))

            row_content = mint_conn.execute("SELECT key FROM workspaces").df()
            if not row_content.empty:
                row_content["key"] = row_content["key"].astype(str)

            def row_comp(key):
                _path = Path(tmpdir, 'workspaces', str(key))
                path_info = html.Div(
                    [
                        fac.AntdText('Workspace path:', strong=True, locale='en-us', style={'marginRight': '10px'}),
                        fac.AntdText(_path.as_posix(), copyable=True, locale='en-us')
                    ],
                    style={'minWidth': '200px', 'flexGrow': 1, 'padding': '10px'}
                )

                # Avoid bumping last_activity just for rendering the preview table
                with duckdb_connection(_path, register_activity=False) as conn:
                    summary = conn.execute("""
                                           SELECT * FROM (
                                               SELECT 'samples' AS table_name, COUNT(*) AS rows FROM samples
                                               UNION ALL
                                               SELECT 'ms1_data' AS table_name, COUNT(*) AS rows FROM ms1_data
                                               UNION ALL
                                               SELECT 'ms2_data' AS table_name, COUNT(*) AS rows FROM ms2_data
                                               UNION ALL
                                               SELECT 'targets' AS table_name, COUNT(*) AS rows FROM targets
                                               UNION ALL
                                               SELECT 'chromatograms' AS table_name, COUNT(*) AS rows FROM chromatograms
                                               UNION ALL
                                               SELECT 'results' AS table_name, COUNT(*) AS rows FROM results
                                           ) t
                                           ORDER BY CASE table_name
                                                        WHEN 'samples' THEN 1
                                                        WHEN 'ms1_data' THEN 2
                                                        WHEN 'ms2_data' THEN 3
                                                        WHEN 'targets' THEN 4
                                                        WHEN 'chromatograms' THEN 5
                                                        WHEN 'results' THEN 6
                                                        ELSE 7
                                                        END
                                           """).df()
                    if not summary.empty:
                        summary['rows'] = summary['rows'].apply(
                            lambda x: f"{int(x):,}" if pd.notna(x) else ""
                        )
                    db_info = fac.AntdTable(
                        columns=[
                            {'title': 'Table name', 'dataIndex': 'table_name', 'align': 'left', 'width': '50%'},
                            {'title': 'Rows', 'dataIndex': 'rows', 'align': 'center', 'width': '50%'},
                        ],
                        data=summary.to_dict('records'),
                        pagination=False,
                        locale='en-us',
                        size='small',
                        style={'minWidth': '200px', 'flexGrow': 1}
                    )
                return fac.AntdFlex(
                    [
                        path_info,
                        db_info
                    ],
                    wrap=True
                )

            row_content['content'] = row_content['key'].apply(row_comp)
            selectedRowKeys = mint_conn.execute("SELECT key FROM workspaces WHERE active = true").fetchone()

            sk = [str(selectedRowKeys[0])] if selectedRowKeys else None
        return data.to_dict('records'), row_content.to_dict('records'), sk

    @app.callback(
        Output('ws-action-store', 'data', allow_duplicate=True),
        Input("ws-table", "data"),
        prevent_initial_call=True,
    )
    def reset_ws_action_store(_data):
        # Clear the action flag after the table refreshes so ordering/notifications return to normal
        return None

    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output("ws-wdir-name-text", "children"),
        Output("ws-wdir-name", "text"),
        Output("wdir", "data"),

        Input("ws-table", "selectedRowKeys"),
        State("tmpdir", "data"),
        State("ws-action-store", "data"),
        prevent_initial_call=True
    )
    def ws_activate(selectedRowKeys, tmpdir, ws_action):
        return _ws_activate(selectedRowKeys, tmpdir, ws_action)

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output('ws-action-store', 'data', allow_duplicate=True),

        Input("ws-table", "recentlyChangedRow"),
        State("ws-table", "recentlyChangedColumn"),
        State("tmpdir", "data"),
        prevent_initial_call=True,
    )
    def save_ws_table_on_edit(row_edited, column_edited, tmpdir):
        return _save_ws_table_on_edit(row_edited, column_edited, tmpdir)

    @app.callback(
        Output("ws-change-data-dir-modal", "visible"),
        Input("ws-change-data-dir-btn", "nClicks"),
        prevent_initial_call=True
    )
    def open_change_data_dir_modal(nClicks):
        return True

    @app.callback(
        Output("tmpdir", "data", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("ws-change-data-dir-modal", "visible", allow_duplicate=True),
        
        Input("ws-change-data-dir-modal", "okCounts"),
        State("ws-input-data-dir", "value"),
        prevent_initial_call=True
    )
    def save_new_data_dir(okCounts, new_path):
        return _save_new_data_dir(okCounts, new_path)

    @app.callback(
        Output("ws-current-data-dir-text", "children"),
        Input("tmpdir", "data")
    )
    def update_data_dir_display(tmpdir):
        logger.debug(f"Updating Display with tmpdir: {tmpdir}")
        if not tmpdir:
            return "Unknown"
        # Strip /Local or /User/* suffix to show the root
        p = Path(tmpdir)
        if p.name == "Local":
            return str(p.parent)
        if p.parent.name == "User":
            return str(p.parent.parent)
        return str(p)
