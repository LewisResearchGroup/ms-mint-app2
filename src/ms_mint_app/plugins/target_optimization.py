import json
import uuid
import math
import logging

from plotly_resampler import FigureResampler
from tqdm import tqdm

import dash
from dash import html, dcc, no_update, Patch
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import feffery_antd_components as fac

from .. import tools as T
from .. import duckdb_manager as DDB
from ..duckdb_manager import duckdb_connection
from ..plugin_interface import PluginInterface

_label = "Optimization"

class TargetOptimizationPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 6
        print(f'Initiated {_label} plugin')

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)
    
    def outputs(self):
        return _outputs


info_txt = """
Creating chromatograms from mzXML/mzML files can last 
a long time the first time. Try converting your files to 
_feather_ format first.'
"""

def create_preview_peakshape_plotly(
    chromatogram_data,
    rt,
    rt_min,
    rt_max,
    peak_label,
    log=False,
):
    fig = FigureResampler(go.Figure())

    if chromatogram_data.empty:
        return fig

    intensity_min = 1e20
    intensity_max = -1e20
    for _, row in chromatogram_data.iterrows():
        label, color, scan_time, intensity = row

        temp_df = pd.DataFrame({
            'scan_time': scan_time,
            'intensity': intensity
        })

        temp_df.explode(['scan_time', 'intensity'])

        # check for intensity min and max before update the df
        intensity_min = min(intensity_min, temp_df['intensity'].min())
        intensity_max = max(intensity_max, temp_df['intensity'].max())

        rt_min_s = rt_min - (rt - rt_min)
        rt_max_s = rt_max + (rt_max - rt)
        temp_df = temp_df[(rt_min_s < temp_df["scan_time"]) & (temp_df["scan_time"] < rt_max_s)]

        fig.add_trace(go.Scatter(
                                 mode='lines', name=label, line=dict(color=color)),
            hf_x=temp_df['scan_time'], hf_y=temp_df['intensity'],
        )

    fig.layout.yaxis.range = [intensity_min, intensity_max]

    fig.add_vrect(x0=rt_min, x1=rt_max, fillcolor="green", opacity=0.05, line_width=0)
    if rt is not None:
        fig.add_vline(x=rt, line_dash="dot", line_color="black")

    fig.update_layout(
        xaxis_title="Retention Time",
        yaxis_title="Intensity",
        title=peak_label,
        showlegend=False,
        margin=dict(l=40, r=5, t=40, b=40),
    )
    if log:
        fig.update_yaxes(type="log")
    return fig


def create_plot(*, chromatogram_data, checkedkeys, rt, rt_min, rt_max, title, log):
    """Crear gráfico con rango y línea central"""
    fig = FigureResampler(go.Figure())
    fig.layout.hovermode = "closest"

    fig.update_layout(
        yaxis_title="Intensity",
        xaxis_title="Scan Time [s]",
        title=title
    )
    if log:
        fig.update_yaxes(type="log")

    fig.add_vline(
        rt, line=dict(color='black', width=3), annotation=dict(text="RT", showarrow=True, bgcolor="white", font=dict(
                color="black", size=14, weight="bold",
            ),)
    )
    fig.add_vrect(
        x0=rt_min, x1=rt_max, line_width=0, fillcolor="green", opacity=0.1
    )

    temp_df = chromatogram_data.explode(['scan_time', 'intensity'])
    slider_min = temp_df['scan_time'].min()
    slider_max = temp_df['scan_time'].max()

    for _, row in chromatogram_data.iterrows():
        label, color, scan_time, intensity = row
        fig.add_trace(
            go.Scattergl(
                         mode='lines+markers', name=label, line=dict(color=color),
                         visible='legendonly' if label not in checkedkeys else True,
                         ),
            hf_x=scan_time,
            hf_y=intensity,
        )
    fig.update_layout(hoverlabel=dict(namelength=-1))
    fig.layout.xaxis.range = [slider_min, slider_max]

    return fig, math.floor(slider_min), math.ceil(slider_max)


config = {
    'scrollZoom': True,             # allows scroll wheel zooming
    'displayModeBar': True,         # show toolbar
    'modeBarButtonsToAdd': [],      # no need to add zoom/pan – already present
    'displaylogo': False
}

_layout = dbc.Container([
    dbc.Row([
        # Side Panel for Controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sample Type"),
                dbc.CardBody([
                    html.Div([
                        # Tree que se expande
                        html.Div([
                            fac.AntdTree(
                                id='sample-type-tree',
                                treeData=[],
                                multiple=True,
                                checkable=True,
                                defaultExpandAll=True,
                            )],
                            style={
                                'flex': '1',
                                'overflow': 'auto',
                                'minHeight': '0'
                            }
                        ),
                        html.Div([
                            html.Label("Selection"),
                            dcc.Dropdown(
                                id="card-plot-selection",
                                options={
                                    'all': 'All',
                                    'bookmarked': 'Bookmarked',
                                    'unmarked': 'Unmarked'
                                },
                                value='all',
                                className="mt-2",
                                clearable=False,
                                style={'marginBottom': '80px'}
                            ),
                            dcc.Checklist(
                                id="pko-figure-options",
                                options=[{"value": "log", "label": "  Logarithmic y-scale"}],
                                value=[],
                                className="mt-4 mb-2",
                            )],
                            style={
                                'flex': '0 0 auto',
                                'borderTop': '1px solid #f0f0f0',
                                'paddingTop': '10px'
                            }
                        )],
                        style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'height': '100%'  # Usar toda la altura disponible del CardBody
                        }
                    )],
                    style={
                        'height': '94vh',  # Ajusta según la altura del header
                        'padding': '10px'
                    }
                    # className="overflow-auto",
                )],
                # className="mb-5",
                style={'maxHeight': '98vh'}
            ),
        ], width=2),  # Reduced width for side panel
        
        # Main Content Column
        dbc.Col([
            # Peak Preview Images (Now directly above the main figure)
            html.Div(
                children=[
                    html.Div([
                        html.H6("Getting chromatograms..."),
                        fac.AntdProgress(
                            id='chromatograms-progress',
                            percent=0,
                        )],
                        id='chromatograms-progress-container',
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "flexDirection": "column",
                            "minWidth": "200px",
                            "maxWidth": "500px",
                            "margin": "auto",
                        }
                    ),
                    fac.AntdSpin(
                        html.Div([
                            fac.AntdSpace(
                                id="pko-peak-preview-images",
                                style={
                                    "width": "100%",
                                    'overflowY': 'auto',
                                    "height": "94vh",
                                },
                                wrap=True,
                                align='start'
                            ),
                            fac.AntdPagination(
                                id='plot-preview-pagination',
                                defaultPageSize=30,
                                locale='en-us',
                                align='center',
                                showTotalSuffix='targets',
                                hideOnSinglePage=True
                            )]
                        ),
                        style={"height": "100%", "alignContent": "center"},
                        text="Loading plots...",
                        size="large",
                        delay=500,
                        includeProps=['pko-peak-preview-images.children']
                    ),


                    # main modal
                    fac.AntdModal(
                        id="pko-info-modal",
                        visible=False,
                        width="90vw",
                        centered=True,
                        destroyOnClose=True,
                        closable=False,
                        maskClosable=False,
                        children=[
                            html.Div([
                                dcc.Graph(
                                    id='pko-plot',
                                    config={'displayModeBar': True},
                                    style={'width': '100%', 'height': '600px'}
                                ),
                                html.Div(
                                    [
                                        html.Div([
                                            dcc.RangeSlider(
                                                id='rt-range-slider',
                                                step=1,
                                                tooltip={"placement": "bottom", "always_visible": False},
                                                pushable=1,
                                                allowCross=False,
                                                updatemode="drag",
                                            )],
                                            id='rslider',
                                            style={'alignItems': 'center',
                                                   "alignContent": 'center',
                                                   "width": "100%"
                                                   }
                                        ),
                                        html.Div(
                                            id="rt-values-span",
                                            style={"display": "flex",
                                                   "flexDirection": "column",
                                                   "fontWeight": "bold",
                                                   "fontSize": "16px",
                                                   "minWidth": "100px"
                                                   }
                                        )
                                    ],
                                    style={"display": "flex",}
                                ),
                            ]),
                            html.Hr(style={'margin': '10px 0 10px 0'}),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Row([
                                        dbc.Col(
                                            fac.AntdAlert(
                                                message="RT values have been changed. Save or reset the changes.",
                                                type="warning",
                                                showIcon=True,
                                            ),
                                            width=8
                                        ),
                                        dbc.Col(
                                            fac.AntdButton(
                                                "Reset",
                                                id="reset-btn",
                                                icon=fac.AntdIcon(icon='antd-reload'),
                                            ),
                                            width=2,
                                            style={'textAlign': 'right',
                                                   "alignContent": "center"}
                                        ),
                                        dbc.Col(
                                            fac.AntdButton(
                                                "Save",
                                                id="save-btn",
                                                type="primary",
                                                icon=fac.AntdIcon(icon="antd-save")
                                            ),
                                            width=2,
                                            style={'textAlign': 'left',
                                                   "alignContent": "center"}
                                        )],
                                        id='action-buttons-container',
                                        style={
                                            "visibility": "hidden",
                                            'opacity': '0',
                                            'transition': 'opacity 0.3s ease-in-out',
                                        }
                                    ),
                                    width=11
                                ),
                                dbc.Col(
                                    fac.AntdButton(
                                        "Close",
                                        id="close-modal-btn",
                                    ),
                                    width=1,
                                    style={"alignContent": "center",
                                           "textAlign": "right"}
                                ),
                            ]),
                        ]
                    ),
                    # confirmation modal
                    fac.AntdModal(
                        id="confirm-modal",
                        title="Confirm close without saving",
                        visible=False,
                        width=400,
                        children=[
                            fac.AntdParagraph(
                                "Are you sure you want to close this window without saving your changes?"),
                            html.Div([
                                fac.AntdSpace([
                                    fac.AntdButton(
                                        "Cancel",
                                        id="stay-btn"
                                    ),
                                    fac.AntdButton(
                                        "Close without saving",
                                        id="close-without-save-btn",
                                        type="primary",
                                        danger=True
                                    )
                                ])
                            ], style={'textAlign': 'right', 'marginTop': '20px'})
                        ]
                    ),
                    fac.AntdModal(
                        id="delete-modal",
                        title="Delete target",
                        visible=False,
                        closable=False,
                        width=400,
                        renderFooter=True,
                        okText="Delete",
                        okButtonProps={"danger": True},
                        cancelText="Cancel"
                    )

                ],
                style={
                    'height': '98vh',
                    'margin': '0',
                    "alignItems": "center",
                    "alignContent": "center",
                }
            ),
            # Hidden div for image click tracking
            html.Div(id="pko-image-clicked", style={'display': 'none'}),
            html.Div(id="delete-target-clicked", style={'display': 'none'}),
        ], width=10),  # Expanded width for main content
        
    ]),
], fluid=True)

pko_layout_no_data = html.Div(
    [
        dcc.Markdown(
            """### No targets found.
    You did not genefrate a targets yet.
    """
        )
    ]
)

_outputs = html.Div(
    id="pko-outputs",
    children=[
        html.Div(id={"index": "pko-set-rt-output", "type": "output"}),
        html.Div(id={"index": "pko-confirm-rt-output", "type": "output"}),
        html.Div(id={"index": "pko-detect-rt-for-all-output", "type": "output"}),
        html.Div(id={"index": "pko-detect-rtspan-for-all-output", "type": "output"}),
        html.Div(id={"index": "pko-detect-rt-output", "type": "output"}),
        html.Div(id={"index": "pko-detect-rtspan-output", "type": "output"}),
        html.Div(id={"index": "pko-drop-target-output", "type": "output"}),
        html.Div(id={"index": "pko-remove-low-intensity-output", "type": "output"}),
        dcc.Store(id='saved-range', data=[None, None, None]),
        dcc.Store(id='current-range', data=[None, None, None]),
        dcc.Store(id='has-unsaved-changes', data=False),
        dcc.Store(id='chromatograms', data={}),
    ],
)

def layout():
    return _layout


def callbacks(app, fsc, cache, cpu=None):

    # clientside callback for set width rangeslider == plot bglayer
    app.clientside_callback(
        """
        function(relayoutData) {
            const root = document.getElementById("pko-plot");
            if (!root) return window.dash_clientside.no_update;
            console.log(root);
            // plotly DOM: tres divs -> svg -> g.bglayer -> rect
            const bg = root.querySelector("div > div > div svg > g.draglayer > g.xy > rect");
            console.log(bg);
            if (!bg) return window.dash_clientside.no_update;

            const rw = root.offsetWidth;
            const w = bg.width.baseVal.value;
            const pl = bg.x.baseVal.value - 25;
            const pr = rw - pl - w - 150;
            console.log("rw:", rw, "w:", w, "pl:", pl, "pr:", pr);

            if (!isFinite(w) || w <= 0) return window.dash_clientside.no_update;


            return {"paddingLeft": pl + "px", "paddingRight": pr + "px", "width": "100%"};
        }
        """,
        Output("rslider", "style"),
        Input("pko-plot", "relayoutData"),
        prevent_initial_call=True
    )

    @app.callback(
        Output('sample-type-tree', 'treeData'),
        Output('sample-type-tree', 'checkedKeys'),
        Output('sample-type-tree', 'expandedKeys'),
        Input("chromatograms", "data"),
        State("wdir", "children"),
    )
    def update_sample_type_tree(chromatograms, wdir):
        if not chromatograms:
            raise PreventUpdate

        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update, dash.no_update, dash.no_update
            metadata = conn.execute("SELECT label, sample_type, use_for_optimization FROM samples_metadata").df()

        if metadata.empty:
            return dash.no_update, dash.no_update, dash.no_update

        sample_type_dict = metadata.groupby('sample_type')['label'].apply(list).to_dict()
        checked_keys = metadata[metadata['use_for_optimization']]['label'].tolist()

        tree_data = []
        for k, v in sample_type_dict.items():
            children = [{'title': ms, 'key': ms} for ms in v if ms in checked_keys]
            if children:
                tree_data.append({'title': k, 'key': k, 'children': children})

        expanded_keys = [
            k
            for k, v in sample_type_dict.items()
            if any(ms in checked_keys for ms in v)
        ]
        return tree_data, checked_keys, expanded_keys

    @app.callback(
        Output('pko-info-modal', 'visible'),
        Output('pko-info-modal', 'title'),
        Output('has-unsaved-changes', 'data'),

        Input("pko-image-clicked", "children"),
        Input('close-modal-btn', 'nClicks'), # nClicks (antd) != n_clicks (dash)
        Input('close-without-save-btn', 'nClicks'),
        Input('sample-type-tree', 'checkedKeys'),
        State('has-unsaved-changes', 'data'),
        State('wdir', 'children'),
        prevent_initial_call=True
    )
    def handle_modal_open_close(image_clicked, close_clicks, close_without_save_clicks,
                                checkedkeys, has_changes, wdir):
        """
        Handle opening and closing of the modal dialog for display the selected target plot.
        :param image_clicked:
        :param close_clicks:
        :param close_without_save_clicks:
        :param saved_values:
        :param has_changes:
        :return:
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'pko-image-clicked':

            return (True, image_clicked,
                    # no_update,
                    # saved_values,
                    False)
        elif trigger_id == 'close-without-save-btn':
            # Close modal without saving changes
            return (False, no_update,
                    # no_update,
                    # no_update,
                    False)
        elif trigger_id == 'close-modal-btn':
            # if not has_changes, close it
            if not has_changes:
                return (False, no_update,
                        # no_update,
                        # no_update,
                        False)
            # if it has_changes, don't close it
            return (no_update, no_update,
                    # no_update,
                    # no_update,
                    no_update)
        else:
            print(f"modal {trigger_id = }")

        return (no_update, no_update,
                # no_update,
                # no_update,
                no_update)

    @app.callback(
        Output('confirm-modal', 'visible'),
        Input('close-modal-btn', 'nClicks'),
        State('has-unsaved-changes', 'data'),
        prevent_initial_call=True
    )
    def show_confirm_modal(close_clicks, has_changes):
        return bool(close_clicks and has_changes)

    @app.callback(
        Output('pko-plot', 'figure'),
        Output('action-buttons-container', 'style'),
        Output("rt-range-slider", "min"),
        Output("rt-range-slider", "max"),
        Output("rt-range-slider", "value"),
        Output("rt-range-slider", "marks"),
        Output("rt-range-slider", "tooltip"),
        Output('has-unsaved-changes', 'data', allow_duplicate=True),

        Input("pko-image-clicked", "children"),
        Input("pko-figure-options", "value"),
        Input('rt-range-slider', 'value'),
        Input('sample-type-tree', 'checkedKeys'),
        State('wdir', 'children'),
        prevent_initial_call=True
    )
    def update_from_slider(image_clicked, options, slider_values, checkedkeys, wdir):

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # reset slider values when is not an update to avoid use stored values from other plots
        if trigger_id != 'rt-range-slider':
            slider_values = None
        if image_clicked is None:
            raise PreventUpdate

        # use peak_label instead of image_clicked, more intuitive
        peak_label = image_clicked

        with duckdb_connection(wdir) as conn:

            target_df = conn.execute("SELECT rt, rt_min, rt_max, peak_label FROM targets "
                                     "WHERE peak_label = ?", [peak_label]).df()

            print(f"{target_df = }")
            rt, rt_min, rt_max, label = target_df.iloc[0]

            # round values to int because the slider steps are int
            orig_values = [int(rt_min), int(rt), int(rt_max)]

            if trigger_id == 'pko-image-clicked':
                samples_for_optimization = conn.execute("SELECT ms_file_label, color, label FROM samples_metadata "
                                                        "WHERE use_for_optimization = True"
                                                        ).df()

                rt_slider_min, st_slider, rt_slider_max = slider_values or orig_values

                query = f"""SELECT 
                                sel.label,
                                sel.color,
                                chr.scan_time, 
                                chr.intensity
                            FROM chromatograms AS chr
                            JOIN samples_for_optimization AS sel ON sel.ms_file_label = chr.ms_file_label
                            WHERE chr.peak_label = ?"""

                chrom_df = conn.execute(query, [peak_label]).df()

                fig, slider_min, slider_max = create_plot(
                    chromatogram_data=chrom_df,
                    checkedkeys=checkedkeys,
                    rt=st_slider,
                    rt_min=rt_slider_min,
                    rt_max=rt_slider_max,
                    title=peak_label,
                    log='log' in options,
                )

                slider_marks = {i: str(i) for i in range(slider_min, slider_max, (int(slider_max - slider_min)//5))}
                has_changes = slider_values != orig_values if slider_values else False

                buttons_style = {
                    'visibility': 'visible' if has_changes else 'hidden',
                    'opacity': '1' if has_changes else '0',
                    'transition': 'opacity 0.3s ease-in-out'
                }
                return (fig, buttons_style, slider_min, slider_max,
                        [rt_slider_min, st_slider, rt_slider_max],
                        slider_marks,
                        {"placement": "bottom", "always_visible": False},
                        has_changes,
                        )

            elif trigger_id == 'rt-range-slider':
                patch_fig = Patch()
                if slider_values[0]:
                    patch_fig['layout']['shapes'][1]['x0'] = slider_values[0]
                if slider_values[2]:
                    patch_fig['layout']['shapes'][1]['x1'] = slider_values[2]
                if slider_values[1]:
                    patch_fig['layout']['shapes'][0]['x0'] = slider_values[1]
                    patch_fig['layout']['shapes'][0]['x1'] = slider_values[1]
                    patch_fig['layout']['annotations'][0]['x'] = slider_values[1]

                has_changes = slider_values != orig_values if slider_values else False
                buttons_style = {
                    'visibility': 'visible' if has_changes else 'hidden',
                    'opacity': '1' if has_changes else '0',
                    'transition': 'opacity 0.3s ease-in-out'
                }

                return (patch_fig, buttons_style, no_update, no_update, no_update, no_update, no_update, has_changes)
        return (no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update)

    @app.callback(
        Output('rt-values-span', 'children'),
        Input('rt-range-slider', 'value'),
        prevent_initial_call=True
    )
    def rt_representation(values):
        rt_min, rt, rt_max = values

        return [html.Span(f"RT-min: {rt_min}", className="rt-value"),
                html.Span(f"RT: {rt}", className="rt-value"),
                html.Span(f"RT-max: {rt_max}", className="rt-value")]


    @app.callback(
        Output('saved-range', 'data', allow_duplicate=True),
        Output('pko-info-modal', 'visible', allow_duplicate=True),
        Output('notifications-container', 'children'),
        Output('has-unsaved-changes', 'data', allow_duplicate=True),
        Output('action-buttons-container', 'style', allow_duplicate=True),

        Input("pko-image-clicked", "children"),
        Input('save-btn', 'nClicks'),
        Input("rt-range-slider", "value"),
        State("has-unsaved-changes", "data"),
        State('wdir', 'children'),
        prevent_initial_call=True
    )
    def save_changes(image_clicked, save_clicks, current_values, unsaved_changes, wdir):

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'save-btn' and unsaved_changes:
            targets = T.get_targets(wdir)
            rt_min, rt, rt_max = current_values
            peak_label = targets.loc[targets['peak_label'] == image_clicked, "peak_label"].iloc[0]
            T.update_targets(wdir, peak_label, rt_min, rt_max, rt)

            buttons_style = {
                'visibility': 'hidden',
                'opacity': '0',
                'transition': 'opacity 0.3s ease-in-out'
            }

            return (
                current_values,  # Actualizar valores guardados
                False,  # Cerrar modal
                fac.AntdNotification(message=f"RT values saved...\n"
                                             f"RT-min: {current_values[0]:.2f}<br>"
                                             f"RT: {current_values[1]:.2f}<br> "
                                             f"RT-max: {current_values[2]:.2f}",
                                     type="success",
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True
                                     ),
                False,
                buttons_style
            )
        raise PreventUpdate

    @app.callback(
        Output('rt-range-slider', 'value', allow_duplicate=True),
        Output('has-unsaved-changes', 'data', allow_duplicate=True),
        Output('action-buttons-container', 'style', allow_duplicate=True),

        Input("pko-image-clicked", "children"),
        Input('reset-btn', 'nClicks'),
        State('wdir', 'children'),
        prevent_initial_call=True
    )
    def reset_changes(image_clicked, reset_clicks, wdir):
        """
        Reset the slider values to the saved RT values
        :param image_clicked:
        :param reset_clicks:
        :param wdir:
        :return:
        """
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'reset-btn' and reset_clicks:
            targets = T.get_targets(wdir)
            rt_min, rt, rt_max = targets.loc[targets['peak_label'] == image_clicked, ["rt_min", "rt", "rt_max"]].iloc[0]

            buttons_style = {
                'visibility': 'hidden',
                'opacity': '0',
                'transition': 'opacity 0.3s ease-in-out'
            }
            return [int(rt_min), int(rt), int(rt_max)], False, buttons_style
        return no_update, no_update, no_update

    @app.callback(
        Output('confirm-modal', 'visible', allow_duplicate=True),
        Input('stay-btn', 'nClicks'),
        Input('close-without-save-btn', 'nClicks'),
        prevent_initial_call=True
    )
    def handle_confirm_modal(stay_clicks, close_clicks):
        return False

    # callback to compute or read the chromatograms
    @app.callback(
        Output('chromatograms', 'data'),
        # Output('chromatograms-progress-container', 'style'),
        Input('tab', 'value'),
        Input('chromatograms', 'data'),
        State('wdir', 'children'),
        background=True,
        running=[
            (
                    Output("chromatograms-progress-container", "style"),
                    {"display": "flex",
                     "justifyContent": "center",
                     "alignItems": "center",
                     "flexDirection": "column",
                     "minWidth": "200px",
                     "maxWidth": "500px",
                     "margin": "auto",},
                    {"display": "none"},
            ),
        ],
        progress=[Output("chromatograms-progress", "percent"),
                  ],
        prevent_initial_call=True
    )
    def compute_chromatograms(set_progress, tab, chromatograms, wdir):

        ctx = dash.callback_context
        prop_id = ctx.triggered[0]['prop_id']
        print(f"{prop_id = }")

        if tab != 'Optimization':
            raise PreventUpdate

        with DDB.duckdb_connection(wdir) as con:
            if con is None:
                return "Could not connect to database."
            start = time.perf_counter()
            DDB.compute_and_insert_chromatograms_from_ms_data(con)
            print(f"Chromatograms computed in {time.perf_counter() - start:.2f} seconds")
            # DDB.compute_and_insert_chromatograms_iteratively(con, set_progress=set_progress)
            df = con.execute("SELECT * FROM chromatograms WHERE peak_label = 'Acetoacetic acid'").df()
            import pandas as pd
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(f"{df.head() = }")
        return True

    @app.callback(
        Output("pko-peak-preview-images", "children"),
        Output('plot-preview-pagination', 'total'),

        Input('chromatograms', 'data'),

        Input('plot-preview-pagination', 'current'),
        Input('plot-preview-pagination', 'pageSize'),

        Input('sample-type-tree', 'checkedKeys'),
        Input("pko-figure-options", "value"),
        Input('card-plot-selection', 'value'),
        Input({"index": "pko-drop-target-output", "type": "output"}, "children"),
        State("wdir", "children"),
    )
    def peak_preview(chromatograms, current_page, page_size, checkedkeys, options, selection, dropped_target, wdir):
        logging.info(f'Create peak previews {wdir}')

        if not chromatograms:
            raise PreventUpdate

        with duckdb_connection(wdir) as conn:
            ms_files_selection = conn.execute(f"SELECT ms_file_label, color, label FROM samples_metadata WHERE label IN"
                                              f" {checkedkeys}").df()

            if ms_files_selection.empty:
                return (fac.AntdNotification(
                    message="No chromatogram to preview",
                    description='No files selected for peak optimization in MS-Files tab. Please, mark some files in the '
                            'Sample Type tree. "use_for_optimization" is use as the initial selection only',
                    type="error", duration=None,
                    placement='bottom',
                    showProgress=True,
                    stack=True
                ), no_update)
            logging.info(f"Using {len(ms_files_selection)} files for peak preview. ({ms_files_selection = })")

            if selection == 'all':
                total_targets_query = "SELECT COUNT(*) FROM targets WHERE preselected_processing = TRUE"
                query = "SELECT * FROM targets WHERE preselected_processing = TRUE LIMIT ? OFFSET ?"
            elif selection == 'bookmarked':
                total_targets_query = "SELECT COUNT(*) FROM targets WHERE bookmark = TRUE"
                query = "SELECT * FROM targets WHERE bookmark = TRUE LIMIT ? OFFSET ?"
            else:
                total_targets_query = "SELECT COUNT(*) FROM targets WHERE bookmark = FALSE"
                query = "SELECT * FROM targets WHERE bookmark = FALSE LIMIT ? OFFSET ?"

            total_targets = conn.execute(total_targets_query).fetchone()[0]
            target_selection = conn.execute(query, [page_size, (current_page - 1) * page_size]).df()

            plots = []
            for _, row in target_selection.iterrows():
                peak_label, mz_mean, mz_width, rt, rt_min, rt_max, bookmark, score = row[
                    ["peak_label", "mz_mean", "mz_width", "rt", "rt_min", "rt_max", 'bookmark', 'score']
                ]

                query = f"""SELECT 
                                sel.label,
                                sel.color,
                                chr.scan_time, 
                                chr.intensity
                            FROM chromatograms AS chr
                            JOIN ms_files_selection AS sel ON sel.ms_file_label = chr.ms_file_label
                            WHERE chr.peak_label = ?"""

                all_chrom_df = conn.execute(query, [peak_label]).df()
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                print(f"{all_chrom_df.head(5) = }")

                fig = create_preview_peakshape_plotly(
                    all_chrom_df,
                    rt,
                    rt_min,
                    rt_max,
                    peak_label=peak_label,
                    log='log' in options
                )

                plots.append(
                    fac.AntdCard([
                        dcc.Graph(
                            id={'type': 'graph-card-preview', 'index': f"{peak_label}-{uuid.uuid4().hex[:6]}"},
                            figure=fig,
                            style={'height': '150px', 'width': '200px', 'margin': '0px'},
                            config={
                                'displayModeBar': False,
                                'staticPlot': True,  # Totalmente estático para máxima performance
                                'doubleClick': False,
                                'showTips': False,
                                'responsive': False  # Tamaño fijo
                            },
                        ),
                        fac.AntdTooltip(
                            fac.AntdButton(
                                icon=fac.AntdIcon(icon='antd-delete'),
                                type='text',
                                size='small',
                                id={'type': 'delete-target-btn', 'index': peak_label},
                                style={
                                    'padding': '4px',
                                    'minWidth': '24px',
                                    'height': '24px',
                                    'borderRadius': '50%',
                                    'background': 'rgba(0, 0, 0, 0.5)',
                                    'color': 'white',
                                    'position': 'absolute',
                                    'bottom': '8px',
                                    'right': '8px',
                                    'zIndex': 10,
                                    'opacity': '0',
                                    'transition': 'opacity 0.3s ease'
                                },
                                className='peak-action-button',
                            ),
                            title='Delete target',
                            color='red',
                            placement='bottom',
                        ),
                        fac.AntdRate(
                            id={'type': 'rate-target-card', 'index': peak_label},
                            count=1,
                            defaultValue=0,
                            value=int(bookmark),
                            allowHalf=False,
                            tooltips=['Bookmark this target'],
                            style={'position': 'absolute', 'top': '8px', 'right': '8px', 'zIndex': 10},
                        )],
                        id={'type': 'plot-card-preview', 'index': peak_label},
                        style={'cursor': 'pointer'},
                        styles={'header': {'display': 'none'}, 'body': {'padding': '5px'}},
                        hoverable=True,
                        className='peak-card-container',
                    )
                )
        return plots, total_targets

    @app.callback(
        Output("pko-image-clicked", "children"),
        [Input({"type": "plot-card-preview", "index": ALL}, "nClicks")],
        [Input({"type": "delete-target-btn", "index": ALL}, "nClicks")],
        [Input({"type": "rate-target-card", "index": ALL}, "value")],
        prevent_initial_call=True,
    )
    def pko_image_clicked(ndx, dt_ndx, b_ndx):

        if ndx is None or len(ndx) == 0:
            raise PreventUpdate

        ctx = dash.callback_context

        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_id = ctx_trigger['index']
        trigger_type = ctx_trigger['type']

        if len(ctx.triggered) > 1 or trigger_type != "plot-card-preview":
            raise PreventUpdate
        return trigger_id

    @app.callback(
        Output('delete-modal', 'visible'),
        Output('delete-modal', 'children'),
        Output('delete-target-clicked', 'children'),
        Input({"type": "delete-target-btn", "index": ALL}, 'nClicks'),
        prevent_initial_call=True
    )
    def show_delete_modal(delete_clicks):

        ctx = dash.callback_context
        if not delete_clicks or not ctx.triggered:
            raise PreventUpdate
        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_id = ctx_trigger['index']

        if len(dash.callback_context.triggered) > 1:
            raise PreventUpdate

        return True, fac.AntdParagraph(f"Are you sure you want to delete `{trigger_id}` target?"), trigger_id

    @app.callback(
        Output({"index": "pko-drop-target-output", "type": "output"}, "children"),
        Input('delete-modal', 'okCounts'),
        Input('delete-modal', 'cancelCounts'),
        Input('delete-modal', 'closeCounts'),
        Input('delete-target-clicked', 'children'),
        State('wdir', 'children'),
        prevent_initial_call=True
    )
    def plk_delete(okCounts, cancelCounts, closeCounts, target, wdir):
        if not okCounts or cancelCounts or closeCounts:
            raise PreventUpdate
        targets = T.get_targets(wdir)
        targets = targets[targets['peak_label'] != target]

        T.write_targets(targets, wdir)
        return fac.AntdNotification(message=f"{target} deleted",
                                    type="success",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    )

    @app.callback(
        Output('notifications-container', "children", allow_duplicate=True),
        Input({"type": "rate-target-card", "index": ALL}, "value"),
        State('wdir', 'children'),
        prevent_initial_call=True
    )
    def bookmark_target(value, wdir):
        # TODO: change bookmark to bool since the AntdRate component returns an int and the db require a bool
        ctx = dash.callback_context
        if not ctx.triggered or len(dash.callback_context.triggered) > 1:
            raise PreventUpdate

        targets = T.get_targets(wdir)

        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_id = ctx_trigger['index']

        if targets['bookmark'].values.tolist() == value:
            raise PreventUpdate

        targets['bookmark'] = value
        T.write_targets(targets, wdir)
        new_value = targets.loc[targets['peak_label'] == trigger_id, 'bookmark'].values[0]

        return fac.AntdNotification(message=f"Target {trigger_id} {'' if new_value else 'un'}bookmarked", duration=3,
                                    placement='bottom', type="success", showProgress=True, stack=True)




