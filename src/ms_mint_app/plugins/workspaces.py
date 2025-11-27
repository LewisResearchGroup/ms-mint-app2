import re
import shutil
from pathlib import Path

import dash
import feffery_antd_components as fac
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..duckdb_manager import duckdb_connection_mint, duckdb_connection
from ..plugin_interface import PluginInterface

_label = "Workspaces"
pattern = re.compile(r"^[A-Za-z0-9_]+$")


class WorkspacesPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 0
        print(f'Initiated {_label} plugin')

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
                        fac.AntdButton('Create Workspace', id='ws-create', icon=fac.AntdIcon(icon='antd-plus')),
                        fac.AntdButton('Delete Workspace', id='ws-delete', danger=True, icon=fac.AntdIcon(
                            icon='antd-minus'))],
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
        )
    ]
)

_outputs = html.Div(
    id="ws-outputs",
    children=[

    ],
)


def callbacks(app, fsc, cache):
    @app.callback(
        Output('ws-create-form-item', 'validateStatus'),
        Output('ws-create-form-item', 'help'),
        Output('ws-create-modal', 'okButtonProps'),

        Input('ws-create-input-name', 'value'),
        State("tmpdir", "data"),
        prevent_initial_call=True
    )
    def create_ws_input_validation(value, tmpdir):
        if value is None:
            raise PreventUpdate

        if value is not None and bool(pattern.match(value)):
            with duckdb_connection_mint(tmpdir) as mint_conn:
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
        if not okCounts:
            raise PreventUpdate
        with duckdb_connection_mint(tmpdir) as mint_conn:
            previous_active = mint_conn.execute("SELECT key FROM workspaces WHERE active = true").fetchone()

            key = mint_conn.execute("INSERT INTO workspaces (name, description, active, created_at, last_activity) "
                                    "VALUES (?, ?, true, NOW(), NOW()) RETURNING key", (ws_name, ws_description)).fetchone()
            if previous_active:
                mint_conn.execute("UPDATE workspaces SET active = false WHERE key = ?", (previous_active[0],))
            if key:
                ws_path = Path(tmpdir, 'workspaces', str(key[0]))
                ws_path.mkdir(parents=True, exist_ok=True)

        return 'create', None, None, None

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
        if not okCounts:
            raise PreventUpdate

        print(f'{selectedRowKeys = }')

        with duckdb_connection_mint(tmpdir) as mint_conn:
            next_active = mint_conn.execute("SELECT key FROM workspaces "
                                            "WHERE active = false ORDER BY last_activity DESC LIMIT 1").fetchone()

            name = mint_conn.execute("DELETE FROM workspaces WHERE key = ? RETURNING name",
                                     (selectedRowKeys[0],)).fetchone()
            if next_active:
                mint_conn.execute("UPDATE workspaces SET active = true WHERE key = ?", (next_active[0],))

            if name:
                ws_path = Path(tmpdir, 'workspaces', str(selectedRowKeys[0]))
                shutil.rmtree(ws_path)
                ws_name = name[0]

                return fac.AntdNotification(message=f"Workspace {ws_name} deleted.",
                                            type="success",
                                            duration=3,
                                            placement='bottom',
                                            showProgress=True,
                                            stack=True), {'type': 'delete', 'status': 'success'}

        return dash.no_update, {'type': 'delete', 'status': 'error'}

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
        State("tmpdir", "data"),
    )
    def ws_table(section_context, ws_action, tmpdir):

        if section_context and  section_context['page'] != 'Workspaces':
            raise PreventUpdate

        with duckdb_connection_mint(tmpdir) as mint_conn:
            if ws_action is None:
                stmt = "SELECT * FROM workspaces ORDER BY last_activity DESC"
            else:
                stmt = "SELECT * FROM workspaces"

            data = mint_conn.execute(stmt).df()
            print(f"{data = }")

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

                with duckdb_connection(_path) as conn:
                    summary = conn.execute("""
                                           SELECT table_name, estimated_size as rows
                                           FROM duckdb_tables()
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

        if not selectedRowKeys:
            return dash.no_update, '', '', None

        with duckdb_connection_mint(tmpdir) as mint_conn:
            mint_conn.execute("UPDATE workspaces SET active = false WHERE key != ?", (selectedRowKeys[0],))
            name = mint_conn.execute("UPDATE workspaces SET active = true WHERE key = ? RETURNING name",
                                     (selectedRowKeys[0],)).fetchone()
            if name:
                ws_name = name[0]
                wdir = Path(tmpdir, 'workspaces', selectedRowKeys[0])

            if ws_action:
                notification = dash.no_update
            else:
                notification = fac.AntdNotification(message=f"Workspace {ws_name} activated.",
                                                    type="success",
                                                    duration=3,
                                                    placement='bottom',
                                                    showProgress=True,
                                                    stack=True)

        return notification, ws_name, wdir.as_posix(), wdir.as_posix()

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),

        Input("ws-table", "recentlyChangedRow"),
        State("ws-table", "recentlyChangedColumn"),
        State("tmpdir", "data"),
        prevent_initial_call=True,
    )
    def save_ws_table_on_edit(row_edited, column_edited, tmpdir):
        """
        This callback saves the table on cell edits.
        This saves some bandwidth.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        if row_edited is None or column_edited is None:
            raise PreventUpdate

        print(f"{row_edited = }")
        print(f"{column_edited = }")

        try:
            with duckdb_connection_mint(tmpdir, workspace=row_edited['key']) as mint_conn:
                if mint_conn is None:
                    raise PreventUpdate
                query = f"UPDATE workspaces SET {column_edited} = ? WHERE key = ?"
                mint_conn.execute(query, [row_edited[column_edited], row_edited['key']])
            return fac.AntdNotification(message="Successfully edition saved",
                                        type="success",
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        stack=True
                                        )
        except Exception as e:
            return fac.AntdNotification(message="Failed to save edition",
                                        description=f"Failing to save edition with: {str(e)}",
                                        type="error",
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        stack=True
                                        )
