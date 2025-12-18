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
import math
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from plotly import colors

from .. import tools as T
from ..colors import make_palette_hsv
from ..duckdb_manager import duckdb_connection, build_where_and_params, build_order_by
from ..plugin_interface import PluginInterface
from ..sample_metadata import GROUP_COLUMNS, GROUP_DESCRIPTIONS, GROUP_LABELS

_label = "MS-Files"
MS_METADATA_TEMPLATE_COLUMNS = [
    'ms_file_label',
    'label',
    'color',
    'use_for_optimization',
    'use_for_processing',
    'use_for_analysis',
    'sample_type',
    *GROUP_COLUMNS,
    'polarity',
    'ms_type',
]
MS_METADATA_TEMPLATE_DESCRIPTIONS = [
    'Unique file name; must match the MS file on disk',
    'Friendly label to display in plots and reports',
    'Hex color for visualizations (auto-generated if blank)',
    'True to include in optimization steps (COMPUTE CHROMATOGRAMS)',
    'True to include in processing (RUN MINT)',
    'True to include in analysis outputs',
    'Sample category (e.g.; Sample; QC; Blank; Standard)',
    *[GROUP_DESCRIPTIONS[col] for col in GROUP_COLUMNS],
    'Polarity (Positive or Negative)',
    'Acquisition type (ms1 or ms2)',
]
MS_METADATA_TEMPLATE_CSV = (
    ",".join(MS_METADATA_TEMPLATE_COLUMNS)
    + "\n"
    + ",".join(MS_METADATA_TEMPLATE_DESCRIPTIONS)
    + "\n"
)
MS_METADATA_DESCRIPTION_MAP = dict(zip(MS_METADATA_TEMPLATE_COLUMNS, MS_METADATA_TEMPLATE_DESCRIPTIONS))
NOTIFICATION_COMPACT_STYLE = {"maxWidth": 420, "width": "420px"}

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
                fac.AntdFlex(
                    [
                        fac.AntdButton(
                            'Download template',
                            id='download-ms-template-btn',
                            icon=fac.AntdIcon(icon='antd-download'),
                            iconPosition='end',
                            style={'textTransform': 'uppercase'},
                        ),
                        fac.AntdButton(
                            'Download MS-files',
                            id='download-ms-files-btn',
                            icon=fac.AntdIcon(icon='antd-download'),
                            iconPosition='end',
                            style={'textTransform': 'uppercase'},
                        ),
                        html.Div(
                            fac.AntdDropdown(
                                id='ms-options',
                                title='Options',
                                buttonMode=True,
                                arrow=True,
                                menuItems=[
                                    {'title': 'Generate colors', 'icon': 'antd-highlight', 'key': 'generate-colors'},
                                    {'isDivider': True},
                                    {'title': fac.AntdText('Delete selected', strong=True, type='warning'),
                                     'key': 'delete-selected'},
                                    {'title': fac.AntdText('Clear table', strong=True, type='danger'), 'key': 'delete-all'},
                                ],
                                buttonProps={'style': {'textTransform': 'uppercase'}},
                            ),
                            id='ms-options-wrapper',
                        ),
                    ],
                    align='center',
                    gap='small',
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
                                'width': '300px',
                                'fixed': 'left'
                            },
                            {
                                'title': 'Label',
                                'dataIndex': 'label',
                                'width': '300px',
                                'editable': True,
                                'editOptions': {
                                    'mode': 'text-area',
                                    'autoSize': {'minRows': 1, 'maxRows': 3},
                                },
                            },
                            {
                                'title': 'Color',
                                'dataIndex': 'color',
                                'width': '100px',
                                'renderOptions': {'renderType': 'button'},
                            },
                            {
                                'title': 'For Optimization',
                                'dataIndex': 'use_for_optimization',
                                'renderOptions': {'renderType': 'switch'},
                                'width': '170px',
                            },
                            {
                                'title': 'For Processing',
                                'dataIndex': 'use_for_processing',
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
                            *[
                                {
                                    'title': GROUP_LABELS[col],
                                    'dataIndex': col,
                                    'width': '130px',
                                    'editable': True,
                                }
                                for col in GROUP_COLUMNS
                            ],
                            {
                                'title': 'Polarity',
                                'dataIndex': 'polarity',
                                'width': '150px',
                            },
                            {
                                'title': 'MS Type',
                                'dataIndex': 'ms_type',
                                'width': '120px',
                            },
                        ],
                        titlePopoverInfo={
                            'ms_file_label': {
                                'title': 'ms_file_label',
                                'content': MS_METADATA_DESCRIPTION_MAP['ms_file_label'],
                            },
                            'label': {
                                'title': 'label',
                                'content': MS_METADATA_DESCRIPTION_MAP['label'],
                            },
                            'color': {
                                'title': 'color',
                                'content': MS_METADATA_DESCRIPTION_MAP['color'],
                            },
                            'use_for_optimization': {
                                'title': 'use_for_optimization',
                                'content': MS_METADATA_DESCRIPTION_MAP['use_for_optimization'],
                            },
                            'use_for_processing': {
                                'title': 'use_for_processing',
                                'content': MS_METADATA_DESCRIPTION_MAP['use_for_processing'],
                            },
                            'use_for_analysis': {
                                'title': 'use_for_analysis',
                                'content': MS_METADATA_DESCRIPTION_MAP['use_for_analysis'],
                            },
                            'sample_type': {
                                'title': 'sample_type',
                                'content': MS_METADATA_DESCRIPTION_MAP['sample_type'],
                            },
                            **{
                                col: {
                                    'title': GROUP_LABELS[col],
                                    'content': MS_METADATA_DESCRIPTION_MAP[col],
                                } for col in GROUP_COLUMNS
                            },
                            'polarity': {
                                'title': 'polarity',
                                'content': MS_METADATA_DESCRIPTION_MAP['polarity'],
                            },
                            'ms_type': {
                                'title': 'ms_type',
                                'content': MS_METADATA_DESCRIPTION_MAP['ms_type'],
                            },
                            'file_type': {
                                'title': 'file_type',
                                'content': 'Raw file format (e.g., mzML, mzXML)',
                            },
                        },
                        filterOptions={
                            'ms_file_label': {'filterMode': 'keyword'},
                            'label': {'filterMode': 'keyword'},
                            'color': {'filterMode': 'keyword'},
                            'use_for_optimization': {'filterMode': 'checkbox',
                                                      'filterCustomItems': ['True', 'False']},
                            'use_for_processing': {'filterMode': 'checkbox',
                                                   'filterCustomItems': ['True', 'False']},
                            'use_for_analysis': {'filterMode': 'checkbox',
                                                  'filterCustomItems': ['True', 'False']},
                            'sample_type': {'filterMode': 'checkbox'},
                            **{col: {'filterMode': 'keyword'} for col in GROUP_COLUMNS},
                            'polarity': {'filterMode': 'checkbox',
                                         'filterCustomItems': ['Positive', 'Negative']},
                            'ms_type': {'filterMode': 'checkbox',
                                        'filterCustomItems': ['ms1', 'ms2']},
                        },
                        sortOptions={'sortDataIndexes': []},
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
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'This tutorial shows how to load, review, and export MS files.',
                },
                {
                    'title': 'Load raw files',
                    'description': 'Click “Load MS-Files” to browse and add raw data files to this workspace.',
                    'targetSelector': "[id='{\"action\":\"file-explorer\",\"type\":\"ms-files\"}']"
                },
                {
                    'title': 'Use the metadata template',
                    'description': 'Download the metadata template if you need the expected column names.',
                    'targetSelector': '#download-ms-template-btn'
                },
                {
                    'title': 'Add metadata',
                    'description': '(Optional) Use “Load Metadata” to import a CSV with sample info (labels, types, etc.).',
                    'targetSelector': "[id='{\"action\":\"file-explorer\",\"type\":\"metadata\"}']"
                },
                {
                    'title': 'Options',
                    'description': 'Generate colors or delete selected/all rows from the options menu.',
                    'targetSelector': '#ms-options-wrapper'
                },
                {
                    'title': 'Review and filter',
                    'description': 'Inspect, filter, and sort MS files here; select rows to delete or export.',
                    'targetSelector': '#ms-files-table-container'
                },
                {
                    'title': 'Export',
                    'description': 'Download the current table (with server-side filters) for backup or sharing.',
                    'targetSelector': '#download-ms-files-btn'
                },
            ],
            id='ms-files-tour',
            open=False,
            current=0,
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Need help?',
                    'description': 'Click the info icon to open a quick tour of the MS-Files table.',
                    'targetSelector': '#ms-files-tour-icon',
                },
            ],
            mask=False,
            placement='rightTop',
            open=False,
            current=0,
            id='ms-files-tour-hint',
            className='targets-tour-hint',
            style={
                'background': '#ffffff',
                'border': '0.5px solid #1677ff',
                'boxShadow': '0 6px 16px rgba(0,0,0,0.15), 0 0 0 1px rgba(22,119,255,0.2)',
                'opacity': 1,
            },
        ),
        dcc.Store(id="ms-tour-hint-store", data={'open': False}, storage_type='local'),
        dcc.Download(id='download-ms-files-csv'),
        dcc.Store(id="ms-table-action-store", data={}),
    ]
)


def layout():
    return _layout

def generate_colors(wdir, regenerate=False):
    def _rgb_to_hex(rgb_str: str) -> str:
        if isinstance(rgb_str, str) and rgb_str.startswith("rgb"):
            nums = rgb_str.strip("rgb() ").split(",")
            if len(nums) == 3:
                try:
                    r, g, b = (int(n) for n in nums)
                    return f"#{r:02x}{g:02x}{b:02x}"
                except ValueError:
                    return rgb_str
        return rgb_str

    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate

        ms_colors = conn.execute(
            "SELECT ms_file_label, sample_type, color FROM samples"
        ).df()
        ms_colors["sample_key"] = ms_colors["sample_type"].fillna(ms_colors["ms_file_label"])

        if regenerate:
            assigned_colors = {}
        else:
            valid = ms_colors[
                ms_colors["color"].notna()
                & (ms_colors["color"].str.strip() != "")
                & (ms_colors["color"].str.strip() != "#bbbbbb")
            ].copy()
            valid["sample_key"] = valid["sample_type"].fillna(valid["ms_file_label"])
            assigned_colors = (
                valid.drop_duplicates(subset="sample_key")
                .set_index("sample_key")["color"]
                .to_dict()
            )

        sample_keys = ms_colors["sample_key"].drop_duplicates().to_list()

        if len(assigned_colors) != len(sample_keys):
            pastel_palette = colors.qualitative.Pastel
            pastel_palette_hex = [_rgb_to_hex(c) for c in pastel_palette]
            if len(sample_keys) <= len(pastel_palette):
                colors_map = assigned_colors.copy()
                palette_idx = 0
                for key in sample_keys:
                    if key in colors_map:
                        continue
                    colors_map[key] = pastel_palette_hex[palette_idx]
                    palette_idx += 1
            else:
                colors_map = make_palette_hsv(
                    sample_keys,
                    existing_map=assigned_colors,
                    s_range=(0.90, 0.95),
                    v_range=(0.90, 0.95),
                )

            colors_pd = ms_colors[["ms_file_label"]].copy()
            colors_pd["color"] = ms_colors["sample_key"].map(colors_map)
            conn.register("colors_pd", colors_pd)
            conn.execute(
                """
                UPDATE samples
                SET color = colors_pd.color
                FROM colors_pd
                WHERE samples.ms_file_label = colors_pd.ms_file_label
                """
            )

        return len(sample_keys) - len(assigned_colors)


def callbacks(cls, app, fsc, cache, args_namespace):
    @app.callback(
        Output('color-picker-modal', 'visible'),
        Output('hex-color-picker', 'color'),
        Input('ms-files-table', 'nClicksButton'),
        State('ms-files-table', 'clickedCustom'),
        prevent_initial_call=True
    )
    def open_color_picker(nClicksButton, clickedCustom):
        if not clickedCustom or 'color' not in clickedCustom:
            raise PreventUpdate
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

        if recentlyButtonClickedRow is None or not okCounts or not wdir:
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
                                         stack=True,
                                         style=NOTIFICATION_COMPACT_STYLE
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
                                         stack=True,
                                         style=NOTIFICATION_COMPACT_STYLE
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
                clickedKey != 'generate-colors'
        ):
            raise PreventUpdate
        # Single option: always generate colors by sample type, refreshing missing/placeholder values.
        n_colors = generate_colors(wdir, regenerate=True)

        if n_colors == 0:
            notification = fac.AntdNotification(message='No colors generated',
                                                type='warning',
                                                duration=3,
                                                placement='bottom',
                                                showProgress=True,
                                                stack=True,
                                                style=NOTIFICATION_COMPACT_STYLE
                                                )
        else:
            notification = fac.AntdNotification(message='Colors generated successfully',
                                     description=f'{n_colors} colors generated',
                                     type='success',
                                     duration=3,
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True,
                                     style=NOTIFICATION_COMPACT_STYLE
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

        if not wdir or not recentlySwitchDataIndex or recentlySwitchStatus is None or not recentlySwitchRow:
            raise PreventUpdate

        allowed_switch_columns = {
            "use_for_optimization",
            "use_for_processing",
            "use_for_analysis",
        }
        if recentlySwitchDataIndex not in allowed_switch_columns:
            raise PreventUpdate

        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            conn.execute(f"UPDATE samples SET {recentlySwitchDataIndex} = ? WHERE ms_file_label = ?",
                         (recentlySwitchStatus, recentlySwitchRow['ms_file_label']))

    @app.callback(
        Output("download-ms-files-csv", "data"),
        Input("download-ms-template-btn", "nClicks"),
        Input("download-ms-files-btn", "nClicks"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def download_ms_files(template_clicks, list_clicks, wdir):
        from ..duckdb_manager import duckdb_connection_mint

        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        ws_name = "workspace"
        if wdir:
            try:
                ws_key = Path(wdir).stem
                with duckdb_connection_mint(Path(wdir).parent.parent) as mint_conn:
                    if mint_conn is not None:
                        ws_row = mint_conn.execute("SELECT name FROM workspaces WHERE key = ?", [ws_key]).fetchone()
                        if ws_row is not None:
                            ws_name = ws_row[0]
            except Exception:
                pass

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'download-ms-template-btn':
            filename = f"{T.today()}-MINT__{ws_name}-ms_files_template.csv"
            return dcc.send_string(MS_METADATA_TEMPLATE_CSV, filename)

        if trigger == 'download-ms-files-btn':
            if not wdir:
                raise PreventUpdate
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                df = conn.execute("SELECT * FROM samples").df()

            filename = f"{T.today()}-MINT__{ws_name}-ms_files.csv"
            return dcc.send_data_frame(df.to_csv, filename, index=False)

        raise PreventUpdate


    @app.callback(
        Output("ms-files-table", "data"),
        Output("ms-files-table", "selectedRowKeys"),
        Output("ms-files-table", "pagination"),
        Output("ms-files-table", "filterOptions"),

        Input('section-context', 'data'),
        Input("ms-table-action-store", "data"),
        Input("processed-action-store", "data"), # from explorer
        Input('ms-files-table', 'pagination'),
        Input('ms-files-table', 'filter'),
        Input('ms-files-table', 'sorter'),
        State('ms-files-table', 'filterOptions'),
        State("processing-type-store", "data"), # from explorer
        State("wdir", "data"),
    )
    def ms_files_table(section_context, processing_output, processed_action, pagination, filter_, sorter, filterOptions,
                       processing_type, wdir):

        if section_context and section_context['page'] != 'MS-Files':
            raise PreventUpdate
        if not wdir:
            raise PreventUpdate

        start_time = time.perf_counter()
        if pagination:
            page_size = pagination['pageSize']
            current = pagination['current']

            with duckdb_connection(wdir) as conn:
                schema = conn.execute("DESCRIBE samples").pl()
            column_types = {r["column_name"]: r["column_type"] for r in schema.to_dicts()}
            where_sql, params = build_where_and_params(filter_, filterOptions)
            order_by_sql = build_order_by(sorter, column_types, tie=('ms_file_label', 'ASC'))  # '' if there is no valid sorter

            sql = f"""
            WITH filtered AS (
              SELECT *
              FROM samples
              {where_sql}
            ),
            paged AS (
              SELECT *, COUNT(*) OVER() AS __total__
              FROM filtered
              {(' ' + order_by_sql) if order_by_sql else ''}
              LIMIT ? OFFSET ?
            )
            SELECT * FROM paged;
            """

            params_paged = params + [page_size, (current - 1) * page_size]

            with duckdb_connection(wdir) as conn:
                dfpl = conn.execute(sql, params_paged).pl()

            # total rows:
            number_records = int(dfpl["__total__"][0]) if len(dfpl) else 0
            max_page = max(math.ceil(number_records / page_size), 1)
            current = min(max(current, 1), max_page)

            # If we just removed the page we were on, re-query for the new page index
            if params_paged[-1] != (current - 1) * page_size:
                params_paged = params + [page_size, (current - 1) * page_size]
                with duckdb_connection(wdir) as conn:
                    dfpl = conn.execute(sql, params_paged).pl()
                number_records = int(dfpl["__total__"][0]) if len(dfpl) else 0


            with (duckdb_connection(wdir) as conn):
                st_custom_items = filterOptions['sample_type'].get('filterCustomItems')
                sample_type_filters = conn.execute("SELECT DISTINCT sample_type "
                                                   "FROM samples "
                                                   "ORDER BY sample_type ASC").df()['sample_type'].to_list()
                if st_custom_items != sample_type_filters:
                    output_filterOptions = filterOptions.copy()
                    output_filterOptions['sample_type']['filterCustomItems'] = (sample_type_filters
                                                                                if sample_type_filters != [None] else
                                                                                [])
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
                pl.col('use_for_processing').map_elements(
                    lambda value: {'checked': value},
                    return_dtype=pl.Object
                ).alias('use_for_processing'),
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
                {**pagination,
                 'total': number_records,
                 'current': current,
                 'pageSizeOptions': sorted(set([5, 10, 25, 50, 100] + ([number_records] if number_records else [])))},
                output_filterOptions
            ]
        return dash.no_update

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input('section-context', 'data'),
        Input("wdir", "data"),
        prevent_initial_call=True,
    )
    def warn_missing_workspace(section_context, wdir):
        if not section_context or section_context.get('page') != 'MS-Files':
            raise PreventUpdate
        if wdir:
            raise PreventUpdate
        return fac.AntdNotification(
            message="Activate a workspace",
            description="Select or create a workspace before working with MS files.",
            type="warning",
            duration=4,
            placement='bottom',
            showProgress=True,
            stack=True,
            style=NOTIFICATION_COMPACT_STYLE,
        )

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
        if (
                not ctx.triggered or
                not nClicks or
                not clickedKey or
                clickedKey not in ['delete-selected', 'delete-all']
        ):
            raise PreventUpdate

        if clickedKey == "delete-selected":
            if not selectedRows:
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
        if not wdir:
            raise PreventUpdate
        if clickedKey == "delete-selected" and not selectedRows:
            ms_table_action_store = {'action': 'delete', 'status': 'failed'}
            total_removed = 0
        elif clickedKey == "delete-selected":
            remove_ms1_file = [row["ms_file_label"] for row in selectedRows if row['ms_type'] == 'ms1']
            remove_ms2_file = [row["ms_file_label"] for row in selectedRows if row['ms_type'] == 'ms2']
            remove_ms_files = remove_ms1_file + remove_ms2_file

            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                try:
                    conn.execute("BEGIN")
                    if remove_ms1_file:
                        conn.execute("DELETE FROM ms1_data WHERE ms_file_label IN ?", (remove_ms1_file,))
                    if remove_ms2_file:
                        conn.execute("DELETE FROM ms2_data WHERE ms_file_label IN ?", (remove_ms2_file,))
                    if remove_ms_files:
                        conn.execute("DELETE FROM chromatograms WHERE ms_file_label IN ?", (remove_ms_files,))
                        conn.execute("DELETE FROM results WHERE ms_file_label IN ?", (remove_ms_files,))
                        conn.execute("DELETE FROM samples WHERE ms_file_label IN ?", (remove_ms_files,))
                    conn.execute("COMMIT")
                    conn.execute("CHECKPOINT")
                except Exception as e:
                    conn.execute("ROLLBACK")
                    logging.error(f"Error deleting selected MS files: {e}")
                    return (fac.AntdNotification(
                                message="Delete MS-files failed",
                                description="Could not delete the selected files; no changes were applied.",
                                type="error",
                                duration=4,
                                placement='bottom',
                                showProgress=True,
                                stack=True,
                                style=NOTIFICATION_COMPACT_STYLE
                            ),
                            {'action': 'delete', 'status': 'failed'})

            total_removed = len(remove_ms_files)
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

                    conn.execute("BEGIN")
                    for t in ("ms1_data", "ms2_data", "chromatograms", "results", "samples"):
                        conn.execute(f"TRUNCATE {t}")
                    conn.execute("COMMIT")
                    conn.execute("CHECKPOINT")
                    conn.execute("ANALYZE")

                    ms_table_action_store = {'action': 'delete', 'status': 'success'}
        return (fac.AntdNotification(message="Delete MS-files",
                                     description=f"Deleted {total_removed} MS-Files",
                                     type="success" if total_removed > 0 else "error",
                                     duration=3,
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True,
                                     style=NOTIFICATION_COMPACT_STYLE
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

        if not wdir or row_edited is None or column_edited is None:
            raise PreventUpdate

        allowed_columns = {
            "label",
            "sample_type",
            *GROUP_COLUMNS,
        }
        if column_edited not in allowed_columns:
            raise PreventUpdate
        try:
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                query = f"UPDATE samples SET {column_edited} = ? WHERE ms_file_label = ?"
                conn.execute(query, [row_edited[column_edited], row_edited['ms_file_label']])
                ms_table_action_store = {'action': 'edit', 'status': 'success'}
            return fac.AntdNotification(message="Successfully edition saved",
                                        type="success",
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        stack=True,
                                        style=NOTIFICATION_COMPACT_STYLE
                                        ), ms_table_action_store
        except Exception as e:
            logging.error(f"Error updating metadata: {e}")
            ms_table_action_store = {'action': 'edit', 'status': 'failed'}
            return fac.AntdNotification(message="Failed to save edition",
                                        description=f"Failing to save edition with: {str(e)}",
                                        type="error",
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        stack=True,
                                        style=NOTIFICATION_COMPACT_STYLE
                                        ), ms_table_action_store

    @app.callback(
        Output('ms-files-tour', 'current'),
        Output('ms-files-tour', 'open'),
        Input('ms-files-tour-icon', 'nClicks'),
        prevent_initial_call=True,
    )
    def ms_files_tour(n_clicks):
        return 0, True

    @app.callback(
        Output('ms-files-tour-hint', 'open'),
        Output('ms-files-tour-hint', 'current'),
        Input('ms-tour-hint-store', 'data'),
    )
    def ms_hint_sync(store_data):
        if not store_data:
            raise PreventUpdate
        return store_data.get('open', True), 0

    @app.callback(
        Output('ms-tour-hint-store', 'data'),
        Input('ms-files-tour-hint', 'closeCounts'),
        Input('ms-files-tour-icon', 'nClicks'),
        State('ms-tour-hint-store', 'data'),
        prevent_initial_call=True,
    )
    def ms_hide_hint(close_counts, n_clicks, store_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'ms-files-tour-icon':
            return {'open': False}

        if close_counts:
            return {'open': False}

        return store_data or {'open': True}
