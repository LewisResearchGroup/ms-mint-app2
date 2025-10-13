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
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from ..colors import make_palette_hsv
from ..duckdb_manager import duckdb_connection
from ..plugin_interface import PluginInterface

_label = "MS-Files"

home_path = Path.home()


class MsFilesPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 2
        print(f"Initiated {_label} plugin")

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache, args):
        callbacks(self, app, fsc, cache, args)

    def outputs(self):
        return None


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
                        {'title': 'Generate colors', 'icon': 'antd-highlight', 'key': 'generate-colors'},
                        {'title': 'Regenerate colors', 'icon': 'pi-broom', 'key': 'regenerate-colors'},
                        {'isDivider': True},
                        {'title': fac.AntdText('Delete selected', strong=True, type='warning'),
                         'key': 'delete-selected'},
                        {'title': fac.AntdText('Clear table', strong=True, type='danger'), 'key': 'delete-all'},
                    ],
                    buttonProps={'style': {'textTransform': 'uppercase'}},
                ),
            ],
            justify="space-between",
            align="center",
            gap="middle",
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
                        containerId='ms-files-table-container',
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
                            'ms_file_label': {'filterMode': 'keyword'},
                            'label': {'filterMode': 'keyword'},
                            'color': {'filterMode': 'keyword'},
                            'user_for_optimization': {'filterMode': 'checkbox',
                                                      'filterCustomItems': ['True', 'False']},
                            'user_for_analysis': {'filterMode': 'checkbox',
                                                  'filterCustomItems': ['True', 'False']},
                            'sample_type': {'filterMode': 'checkbox'},
                            'polarity': {'filterMode': 'checkbox',
                                         'filterCustomItems': ['Positive', 'Negative']},
                            'ms_type': {'filterMode': 'checkbox',
                                        'filterCustomItems': ['ms1', 'ms2']},
                        },
                        sortOptions={'sortDataIndexes': ['run_order']},
                        pagination={
                            'position': 'bottomCenter',
                            'pageSize': 10,
                            'current': 1,
                            'showSizeChanger': True,
                            'pageSizeOptions': [5, 10, 25, 50, 100],
                            'showQuickJumper': True,
                        },
                        tableLayout='fixed',
                        maxWidth="calc(100vw - 250px - 4rem)",
                        maxHeight="calc(100vh - 140px - 2rem)",
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
            style={'paddingTop': '1rem'},
        ),
        dcc.Store(id="ms-table-action-store", data={}),
    ]
)


def layout():
    return _layout

def generate_colors(wdir, regenerate=False):
    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        ms_colors = conn.execute("SELECT ms_file_label, color FROM samples").df()
        if regenerate:
            assigned_colors = {}
        else:
            valid = ms_colors[ms_colors["color"].notna() &
                              (ms_colors["color"].str.strip() != "") &
                              (ms_colors["color"].str.strip() != "#ffffff")]
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
                         UPDATE samples
                         SET color = colors_pd.color
                         FROM colors_pd
                         WHERE samples.ms_file_label = colors_pd.ms_file_label"""
                         )
        return len(ms_colors) - len(assigned_colors)


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
        Output('ms-table-action-store', 'data', allow_duplicate=True),
        Input('color-picker-modal', 'okCounts'),
        State('hex-color-picker', 'color'),
        State('ms-files-table', 'recentlyButtonClickedRow'),
        State('ms-files-table', 'data'),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def set_color(okCounts, color, recentlyButtonClickedRow, data, wdir):

        if recentlyButtonClickedRow is None or not okCounts:
            return dash.no_update, dash.no_update

        previous_color = recentlyButtonClickedRow['color']['content']
        try:
            with duckdb_connection(wdir) as conn:
                conn.execute("UPDATE samples SET color = ? WHERE ms_file_label = ?",
                             [color, recentlyButtonClickedRow['ms_file_label']])
            ms_table_action_store = {'action': 'color-changed', 'status': 'success'}

            return (fac.AntdNotification(message='Color changed successfully',
                                         description=f'Color changed from {previous_color} to {color}',
                                         type='success',
                                         duration=3,
                                         placement='bottom',
                                         showProgress=True,
                                         stack=True
                                         ),
                    ms_table_action_store
                    )
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
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('ms-table-action-store', 'data', allow_duplicate=True),

        Input("ms-options", "nClicks"),
        State("ms-options", "clickedKey"),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def genere_color_map(nClicks, clickedKey, wdir):
        ctx = dash.callback_context
        if (
                not ctx.triggered or
                not nClicks or
                not clickedKey or
                clickedKey not in ['generate-colors', 'regenerate-colors']
        ):
            raise PreventUpdate
        if clickedKey == "generate-colors":
            n_colors = generate_colors(wdir)
        else:
            n_colors = generate_colors(wdir, regenerate=True)

        if n_colors == 0:
            notification = fac.AntdNotification(message='No colors generated',
                                                type='warning',
                                                duration=3,
                                                placement='bottom',
                                                showProgress=True,
                                                stack=True
                                                )
        else:
            notification = fac.AntdNotification(message='Colors generated successfully',
                                     description=f'{n_colors} colors generated',
                                     type='success',
                                     duration=3,
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True
                                     )
        ms_table_action_store = {'action': 'color-changed', 'status': 'success'}
        return notification, ms_table_action_store

    @app.callback(
        Input('ms-files-table', 'recentlySwitchDataIndex'),
        Input('ms-files-table', 'recentlySwitchStatus'),
        Input('ms-files-table', 'recentlySwitchRow'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def save_switch_changes(recentlySwitchDataIndex, recentlySwitchStatus, recentlySwitchRow, wdir):

        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            conn.execute(f"UPDATE samples SET {recentlySwitchDataIndex} = ? WHERE ms_file_label = ?",
                         (recentlySwitchStatus, recentlySwitchRow['ms_file_label']))


    @app.callback(
        Output("ms-files-table", "data"),
        Output("ms-files-table", "selectedRowKeys"),
        Output("ms-files-table", "pagination"),
        Output("ms-files-table", "filterOptions"),

        Input("ms-table-action-store", "data"),
        Input("processed-action-store", "data"), # from explorer
        Input('ms-files-table', 'pagination'),
        Input('ms-files-table', 'filter'),
        State('ms-files-table', 'filterOptions'),
        State("processing-type-store", "data"), # from explorer
        Input("wdir", "data"),
    )
    def ms_files_table(processing_output, processed_action, pagination, filter_, filterOptions, processing_type, wdir):

        # processing_type also store info about targets selections since it is the same modal for all of them
        if (
                wdir is None or
                processing_type.get('type') == 'targets' or
                (processed_action and processed_action.get('status') != 'success')
        ):
            raise PreventUpdate
        start_time = time.perf_counter()
        if pagination:
            page_size = pagination['pageSize']
            current = pagination['current']

            base_query = "SELECT * FROM samples"
            where_clauses = []
            params = []

            if filter_ and any(v for v in filter_.values()):
                with duckdb_connection(wdir) as conn:
                    if conn is None:
                        raise PreventUpdate

                    schema = conn.execute("DESCRIBE samples").pl()
                    column_types = {row['column_name']: row['column_type'] for row in schema.to_dicts()}

                    for key, value in filter_.items():
                        if value:
                            # if keyword (LIKE)
                            if filterOptions[key].get('filterMode') == 'keyword':
                                column_type = column_types.get(key, '').upper()
                                if any(numeric_type in column_type for numeric_type in
                                       ['DOUBLE', 'FLOAT', 'INTEGER', 'BIGINT', 'DECIMAL', 'REAL', 'SMALLINT',
                                        'TINYINT']):
                                    try:
                                        numeric_value = float(value[0])
                                        where_clauses.append(f'\"{key}\" = ?')
                                        params.append(numeric_value)
                                    except (ValueError, TypeError):
                                        # If casting fails, this filter won't match. Add a clause that is always false.
                                        where_clauses.append("1=0")
                                else:
                                    where_clauses.append(f'\"{key}\" ILIKE ?')
                                    params.append(f"%{value[0]}%")
                            # if multiple selection (IN)
                            else:
                                where_clauses.append(f"\"{key}\" IN ({','.join(['?'] * len(value))})")
                                params.extend(value)

                where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                count_query = f"SELECT COUNT(*) FROM samples{where_sql}"

                with (duckdb_connection(wdir) as conn):
                    if conn is None:
                        raise PreventUpdate
                    number_records = conn.execute(count_query, params).fetchone()[0]
                    # fix current page if the last record is deleted and then the number of pages changes
                    current = max(current if number_records > (current - 1) * page_size else current - 1, 1)
                    data_query = f"{base_query}{where_sql} LIMIT ? OFFSET ?"
                    params_data = params + [page_size, (current - 1) * page_size]

                    dfpl = conn.execute(data_query, params_data).pl()
            else:
                with (duckdb_connection(wdir) as conn):
                    if conn is None:
                        raise PreventUpdate
                    # no filters only pagination
                    number_records = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
                    # fix current page if the last record is deleted and then the number of pages changes
                    current = max(current if number_records > (current - 1) * page_size else current - 1, 1)
                    data_query = f"{base_query} LIMIT ? OFFSET ?"
                    dfpl = conn.execute(data_query, [page_size, (current - 1) * page_size]).pl()

            with duckdb_connection(wdir) as conn:
                st_custom_items = filterOptions['sample_type'].get('filterCustomItems')
                sample_type_filters = conn.execute("SELECT DISTINCT sample_type "
                                                   "FROM samples "
                                                   "ORDER BY sample_type ASC").df()['sample_type'].to_list()
                if st_custom_items != sample_type_filters:
                    output_filterOptions = filterOptions.copy()
                    output_filterOptions['sample_type']['filterCustomItems'] = list(sample_type_filters)
                else:
                    output_filterOptions = dash.no_update

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
            end_time = time.perf_counter()
            time.sleep(max(0.0, min(len(data), 0.5 - (end_time - start_time))))

            return [
                data.to_dicts(),
                [],
                {**pagination, 'total': number_records, 'current': current},
                output_filterOptions
            ]
        return dash.no_update

    @app.callback(
        Output("delete-confirmation-modal", "visible"),
        Output("delete-confirmation-modal", "children"),

        Input("ms-options", "nClicks"),
        State("ms-options", "clickedKey"),
        State('ms-files-table', 'selectedRows'),
        prevent_initial_call=True
    )
    def toggle_modal(nClicks, clickedKey, selectedRows):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        if clickedKey == "delete-selected":
            if not bool(selectedRows):
                raise PreventUpdate

        children = fac.AntdFlex(
            [
                fac.AntdText("This action will delete selected MS-files and cannot be undone?",
                             strong=True),
                fac.AntdText("Are you sure you want to delete the selected MS-files?")
            ],
            vertical=True,
        )
        if clickedKey == "delete-all":
            children = fac.AntdFlex(
                [
                    fac.AntdText("This action will delete ALL MS-files and cannot be undone?",
                                 strong=True, type="danger"),
                    fac.AntdText("Are you sure you want to delete the ALL MS-files?")
                ],
                vertical=True,
            )
        return True, children

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("ms-table-action-store", "data", allow_duplicate=True),

        Input("delete-confirmation-modal", "okCounts"),
        State('ms-files-table', 'selectedRows'),
        State("ms-options", "clickedKey"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def confirm_and_delete(okCounts, selectedRows, clickedKey, wdir):

        if okCounts is None:
            raise PreventUpdate
        if clickedKey == "delete-selected" and not selectedRows:
            ms_table_action_store = {'action': 'delete', 'status': 'failed'}
            total_removed = 0
        elif clickedKey == "delete-selected":
            remove_ms_file = [row["ms_file_label"] for row in selectedRows]

            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                conn.execute("DELETE FROM samples WHERE ms_file_label IN ?", (remove_ms_file,))
                conn.execute("DELETE FROM ms_data WHERE ms_file_label IN ?", (remove_ms_file,))
                conn.execute("DELETE FROM chromatograms WHERE ms_file_label IN ?", (remove_ms_file,))
                # conn.execute("DELETE FROM results WHERE ms_file_label = ?", (filename,))
            total_removed = len(remove_ms_file)
            ms_table_action_store = {'action': 'delete', 'status': 'success'}
        else:
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                total_removed_q = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
                ms_table_action_store = {'action': 'delete', 'status': 'failed'}
                total_removed = 0
                if total_removed_q:
                    total_removed = total_removed_q[0]

                    conn.execute("DELETE FROM samples")
                    conn.execute("DELETE FROM chromatograms")
                    # conn.execute("DELETE FROM results")
                    ms_table_action_store = {'action': 'delete', 'status': 'success'}
        return (fac.AntdNotification(message="Delete MS-files",
                                     description=f"Deleted {total_removed} MS-Files",
                                     type="success" if total_removed > 0 else "error",
                                     duration=3,
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True
                                     ),
                ms_table_action_store)

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("ms-table-action-store", "data", allow_duplicate=True),

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
                query = f"UPDATE samples SET {column_edited} = ? WHERE ms_file_label = ?"
                conn.execute(query, [row_edited[column_edited], row_edited['ms_file_label']])
                ms_table_action_store = {'action': 'delete', 'status': 'success'}
            return fac.AntdNotification(message="Successfully edition saved",
                                        type="success",
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        stack=True
                                        ), ms_table_action_store
        except Exception as e:
            logging.error(f"Error updating metadata: {e}")
            ms_table_action_store = {'action': 'delete', 'status': 'failed'}
            return fac.AntdNotification(message="Failed to save edition",
                                        description=f"Failing to save edition with: {str(e)}",
                                        type="error",
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        stack=True
                                        ), ms_table_action_store
