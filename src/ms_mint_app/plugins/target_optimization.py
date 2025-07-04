import os
import shutil
import logging

from tqdm import tqdm

import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns

import dash
from dash import html, dcc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, ALL

import dash_bootstrap_components as dbc

import plotly.graph_objects as go

from ms_mint import Mint

import pandas as pd

from .. import tools as T
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

def create_preview_peakshape(
    ms_files, mz_mean, mz_width, rt, rt_min, rt_max, image_label, wdir, peak_label, colors
):
    """Create peak shape previews."""
    logging.info(f'Create_preview_peakshape {peak_label}')
    fig, ax = plt.subplots(figsize=(2, 1), dpi=30)
    y_max = 0
    for fn in ms_files:
        color = colors[T.filename_to_label(fn)]
        if color is None or color == "":
            color = "grey"
        fn_chro = T.get_chromatogram(fn, mz_mean, mz_width, wdir)
        fn_chro = fn_chro[
            (rt_min < fn_chro["scan_time"]) & (fn_chro["scan_time"] < rt_max)
        ]
        ax.plot(fn_chro["scan_time"], fn_chro["intensity"], lw=1, color=color)
        y_max = max(y_max, fn_chro["intensity"].max())
    if not rt:
        rt = np.mean([rt_min, rt_max])
    if (not np.isnan(rt)) and not (np.isnan(rt_max)) and not (np.isnan(rt_min)):
        x = max(min(rt, rt_max), rt_min)
        rt_mean = np.mean([rt_min, rt_max])
        color_value = np.abs(rt_mean - rt) / 10
        color = T.float_to_color(color_value, vmin=0, vmax=1, cmap="coolwarm")
        ax.vlines(x, 0, y_max, lw=3, color=color)
    title = f'{peak_label[:30]}\nm/z={mz_mean:.2f}'
    ax.set_title(title, y=1.0, pad=15)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlabel("Scan Time [s]")
    ax.set_ylabel("Intensity")
    filename = T.savefig(fig, kind="peak-preview", wdir=wdir, label=image_label)
    plt.close(fig)
    logging.info(f'Create_preview_peakshape {peak_label} done.')
    return filename


config = {
    'scrollZoom': True,             # allows scroll wheel zooming
    'displayModeBar': True,         # show toolbar
    'modeBarButtonsToAdd': [],      # no need to add zoom/pan – already present
    'displaylogo': False
}

multi_figure_layout = dbc.Row([
    
    # Figures Row
    dbc.Row([
        # Full Screen EIC Column
        dbc.Col([
            html.H4("Full EIC"),
            dcc.Graph(
                id="pko-full-figure",
                style={'width': '100%', 'height': '500px'},
                config=config
            )
        ], width=6, className="px-1"),
        
        # Zoom Screen Column
        dbc.Col([
            html.H4("Selected Region"),
            dcc.Graph(
                id="pko-zoom-figure",
                style={'width': '100%', 'height': '500px'},
                config=config,
            )
        ], width=6, className="px-1 bg-red")
    ], className="g-0 mx-0"),
], className='w-100')

_layout = dbc.Container([
    dbc.Row([
        # Side Panel for Controls
        dbc.Col([
            # File Selection Section
            dbc.Card([
                dbc.CardHeader(html.H5("File Selection", className="mb-0")),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="pko-ms-selection",
                        options=[
                            {
                                "label": "Use selected files from metadata table (use_for_optimization)",
                                "value": "peakopt",
                            },
                            {"label": "Use all files (may take a long time)", "value": "all"},
                        ],
                        value="peakopt",
                        clearable=False,
                        className="mb-2"
                    )
                ])
            ], className="mb-3"),
            
            # Peak Previews Control Section
            dbc.Card([
                dbc.CardHeader(html.H5("Peak Previews", className="mb-0")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Button("Show/Update figures", id="pko-peak-preview", className="w-100 mb-1"), width=6),
                        dbc.Col(dbc.Button("Regenerate figures", id="pko-peak-preview-from-scratch", className="w-100 mb-1"), width=6),
                    ]),
                ])
            ], className="mb-3"),
            
                  # Peak Previews Control Section
            dbc.Card([
                dbc.CardHeader(html.H5("Batch Processing", className="mb-0")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Button("Detect RT", id="pko-detect-rt-for-all", color='danger', className="w-100"), width=6),
                        dbc.Col(dbc.Button("Detect RT span", id="pko-detect-rtspan-for-all", color='danger', className="w-100"), width=6),
                    ])
                ])
            ], className="mb-3"),
            
            # Interactive RT Optimization Section
            dbc.Card([
                dbc.CardHeader(html.H5("RT Optimization", className="mb-0")),
                dbc.CardBody([


                    # Navigation Buttons
                    dbc.Row([
                        dbc.Col(dbc.Button("<< Prev", id="pko-prev", className="w-100 mb-1"), width=6),
                        dbc.Col(dbc.Button("Next >>", id="pko-next", className="w-100 mb-1"), width=6),
                    ]),
                    
                    
                    # Dropdown for target selection
                    dcc.Dropdown(
                        id="pko-dropdown", 
                        options=[], 
                        value=None,
                        className="mb-2"
                    ),
                    
                    # Progress Bar
                    dbc.Progress(
                        id="pko-progress-bar",
                        value=0,
                        className="mb-2"
                    ),
                    
                    # RT Control Buttons
                    dbc.Row([
                        dbc.Col(dbc.Button("Detect RT", id="pko-detect-rt", className="w-100 mb-1"), width=6),
                        dbc.Col(dbc.Button("Detect RT span", id="pko-detect-rtspan", className="w-100 mb-1"), width=6),
                    ]),
                    
                    dbc.Row([
                        dbc.Col(dbc.Button("Drop Target", id="pko-drop-target", color='danger', className="w-100 mb-1"), width=6),
                    ]),
                    

                ])
            ], className="mb-3"),
            
            # Figure Options
            dbc.Card([
                dbc.CardBody([
                    dcc.Checklist(
                        id="pko-figure-options",
                        options=[{"value": "log", "label": "Logarithmic y-scale"}],
                        value=[],
                        className="mb-2"
                    )
                ])
            ])
        ], width=3),  # Reduced width for side panel
        
        # Main Content Column
        dbc.Col([
            # Peak Preview Images (Now directly above the main figure)
            html.Div(
                id="pko-peak-preview-images",
                className="overflow-auto mb-2",
                style={"maxHeight": "150px"}
            ),

            multi_figure_layout,
            
            # Main Figure
            html.H4("Manual selection tool"),

            dcc.Loading(
                dcc.Graph(
                    id="pko-figure", 
                    style={'width': '100%', 'height': '80vh'}
                )
            ),
            
            dbc.Row([
                dbc.Col(dbc.Button("Set RT manually", id="pko-confirm-rt", className="w-100 mb-1"), width=6),
                dbc.Col(dbc.Button("Set RT span manually", id="pko-set-rt", className="w-100 mb-1"), width=6),
            ]),
            
            # Hidden div for image click tracking
            html.Div(id="pko-image-clicked", style={'display': 'none'})
        ], width=9),  # Expanded width for main content
        
    ])
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
    ],
)

def layout():
    return _layout


def callbacks(app, fsc, cache, cpu=None):
    @app.callback(
        Output("pko-dropdown", "options"),
        Input("tab", "value"),
        Input({"index": "pko-drop-target-output", "type": "output"}, "children"),
        State("wdir", "children"),
        State("pko-dropdown", "options"),
    )
    def pko_controls(tab, peak_deleted, wdir, old_options):
        if tab != _label:
            raise PreventUpdate
        targets = T.get_targets(wdir)
        if targets is None:
            logging.warning("Target-list is empty")
            raise PreventUpdate
        options = [
            {"label": label, "value": i} for i, label in enumerate(targets.index)
        ]
        if options == old_options:
            raise PreventUpdate
        return options


    @app.callback(
        Output("pko-figure", "figure"),
        Input("pko-dropdown", "value"),
        Input("pko-figure-options", "value"),
        Input({"index": "pko-set-rt-output", "type": "output"}, "children"),
        Input("pko-dropdown", "options"),
        Input({"index": "pko-detect-rtspan-for-all-output", "type": "output"}, "children"),
        Input({"index": "pko-detect-rtspan-output", "type": "output"}, "children"),
        Input({"index": "pko-confirm-rt-output", "type": "output"}, "children"),
        Input({"index": "pko-detect-rt-output", "type": "output"}, "children"),
        Input({"index": "pko-detect-rt-for-all-output", "type": "output"}, "children"),
        State("pko-ms-selection", "value"),
        State("wdir", "children"),
    )
    def pko_figure(
        peak_label_ndx,
        options,
        n_clicks,
        options_changed,
        find_largest_peak,
        find_largest_peak_single,
        rt_set,
        rt_detected_single,
        rt_detected_all,
        ms_selection,
        wdir,
    ):
        fig = None
        if peak_label_ndx is None:
            raise PreventUpdate
        targets = T.get_targets(wdir).reset_index()
        if ms_selection == "peakopt":
            ms_files = T.get_ms_fns_for_peakopt(wdir)
        elif ms_selection == "all":
            ms_files = T.get_ms_fns(wdir)

        cols = ["mz_mean", "mz_width", "rt", "rt_min", "rt_max", "peak_label"]

        peak_label_ndx = peak_label_ndx % len(targets)
        mz_mean, mz_width, rt, rt_min, rt_max, label = targets.loc[peak_label_ndx, cols]
        margin = 30
                
        #if rt is None:
        #    if rt_min and rt_max:
        #        rt = np.mean([rt_min, rt_max])
        #else:
        #    if not rt_min:
        #        rt_min = max(0, rt - margin)
        #    if not rt_max:
        #        rt_max = rt + margin

        if True or fig is None:
            fig = go.Figure()
            fig.layout.hovermode = "closest"
            fig.layout.xaxis.range = [rt_min, rt_max]

            fig.update_layout(
                yaxis_title="Intensity",
                xaxis_title="Scan Time [s]",
                xaxis=dict(rangeslider=dict(visible=True, thickness=0.33)),
            )
            fig.update_layout(title=label)
            if "log" in options:
                fig.update_yaxes(type="log")

        if rt:
            fig.add_vline(rt)

        if rt_min and rt_max:
            fig.add_vrect(
                x0=rt_min, x1=rt_max, line_width=0, fillcolor="green", opacity=0.1
            )

        n_files = len(ms_files)
        for i, fn in tqdm(enumerate(ms_files), total=n_files, desc="PKO-figure"):
            fsc.set("progress", int(100 * (i + 1) / n_files))

            name = os.path.basename(fn)
            name, _ = os.path.splitext(name)
            chrom = T.get_chromatogram(fn, mz_mean, mz_width, wdir)
            fig.add_trace(
                go.Scatter(x=chrom["scan_time"], y=chrom["intensity"], name=name)
            )
            fig.update_layout(showlegend=False)
            fig.update_layout(hoverlabel=dict(namelength=-1))
        return fig

    @app.callback(
        Output("pko-progress-bar", "value"),
        Input("pko-dropdown", "value"),
        State("pko-dropdown", "options"),
    )
    def set_progress(value, options):
        if (value is None) or (options is None):
            raise PreventUpdate
        progress = int(100 * (value + 1) / len(options))
        return progress

    @app.callback(
        Output({"index": "pko-set-rt-output", "type": "output"}, "children"),
        Input("pko-set-rt", "n_clicks"),
        State("pko-dropdown", "value"),
        State("pko-figure", "figure"),
        State("wdir", "children"),
    )
    def pko_set_rt_span(n_clicks, peak_label, fig, wdir):
        if n_clicks is None:
            raise PreventUpdate
        rt_min, rt_max = fig["layout"]["xaxis"]["range"]
        rt_min, rt_max = np.round(rt_min, 4), np.round(rt_max, 4)
        rt = np.mean([rt_min, rt_max])
        T.update_targets(wdir, peak_label, rt_min, rt_max, rt)
        return dbc.Alert(f"Set RT span to ({rt_min},{rt_max})", color="info")

    @app.callback(
        Output({"index": "pko-confirm-rt-output", "type": "output"}, "children"),
        Input("pko-confirm-rt", "n_clicks"),
        State("pko-dropdown", "value"),
        State("pko-figure", "figure"),
        State("wdir", "children"),
    )
    def pko_confirm_rt(n_clicks, peak_label, fig, wdir):
        if n_clicks is None:
            raise PreventUpdate

        rt_min, rt_max = fig["layout"]["xaxis"]["range"]
        rt_min, rt_max = np.round(rt_min, 4), np.round(rt_max, 4)

        image_label = f"{peak_label}_{rt_min}_{rt_max}"

        _, fn = T.get_figure_fn(
            kind="peak-preview", wdir=wdir, label=image_label, format="png"
        )

        rt = np.mean([rt_min, rt_max])

        T.update_targets(wdir, peak_label, rt=rt)

        if os.path.isfile(fn):
            os.remove(fn)

        return dbc.Alert(f"Set RT span to ({rt_min},{rt_max})", color="info")

    @app.callback(
        Output("pko-dropdown", "value"),
        Input("pko-prev", "n_clicks"),
        Input("pko-next", "n_clicks"),
        Input("pko-image-clicked", "children"),
        State("pko-dropdown", "value"),
        State("pko-dropdown", "options"),
        State("wdir", "children"),
    )
    def pko_prev_next(
        n_prev, n_next, image_clicked, value, options, wdir
    ):
        if (
            n_prev is None
            and n_next is None
            and image_clicked is None
        ):
            raise PreventUpdate

        prop_id = dash.callback_context.triggered[0]["prop_id"]

        if prop_id.startswith("pko-image-clicked"):
            for entry in options:
                if entry["label"] == image_clicked:
                    return entry["value"]
        elif value is None:
            return 0
        elif prop_id.startswith("pko-prev"):
            return (value - 1) % len(options)
        elif prop_id.startswith("pko-next"):
            return (value + 1) % len(options)

    @app.callback(
        Output("pko-peak-preview-images", "children"),
        Input("pko-peak-preview", "n_clicks"),
        Input("pko-peak-preview-from-scratch", "n_clicks"),
        State("pko-ms-selection", "value"),
        State("wdir", "children"),
    )
    def peak_preview(n_clicks, from_scratch, ms_selection, wdir):  # peak_opt, #set_rt,
        logging.info(f'Create peak previews {wdir}')
        if n_clicks is None:
            raise PreventUpdate
        # reset updating after 5 attempts
        n_attempts = fsc.get(f"{wdir}-update-attempt")
        if n_attempts is None:
            n_attempts = 1
        elif n_attempts % 5:
            fsc.set(f"{wdir}-updating", False)

        # increment counter of attempts
        fsc.set(f"{wdir}-update-attempt", n_attempts + 1)

        if fsc.get(f"{wdir}-updating") is True:
            raise PreventUpdate

        fsc.set(f"{wdir}-updating", True)

        prop_id = dash.callback_context.triggered[0]["prop_id"]
        regenerate = prop_id.startswith("pko-peak-preview-from-scratch")
        if regenerate:
            image_path = os.path.join(wdir, "figures", "peak-preview")
            if os.path.isdir(image_path):
                shutil.rmtree(image_path)

        if ms_selection == "peakopt":
            ms_files = T.get_ms_fns_for_peakopt(wdir)
        elif ms_selection == "all":
            ms_files = T.get_ms_fns(wdir)
        else:
            assert False, ms_selection

        if len(ms_files) == 0:
            return dbc.Alert(
                'No files selected for peak optimization in MS-Files tab. '
                'Please, select some files in column "use_for_optimization".',
                color="warning",
            )
        else:
            logging.info(
                f"Using {len(ms_files)} files for peak preview. ({ms_selection})"
            )

        targets = T.get_targets(wdir)
        
        file_colors = T.file_colors(wdir)

        n_total = len(targets)

        sns.set_context("paper")
        images = []
        for i, (peak_label, row) in tqdm(enumerate(targets.iterrows()), total=n_total):
            fsc.set("progress", int(100 * (i + 1) / n_total))
            mz_mean, mz_width, rt, rt_min, rt_max = row[
                ["mz_mean", "mz_width", "rt", "rt_min", "rt_max"]
            ]

            if not rt_min:
                rt_min = 0
            if  not rt_max:
                rt_max = 1000

            image_label = f"{peak_label}_{rt_min}_{rt_max}"

            _, fn = T.get_figure_fn(
                kind="peak-preview", wdir=wdir, label=image_label, format="png"
            )

            if not os.path.isfile(fn) or regenerate:
                logging.info(f"Regenerating figure for {peak_label}")
                create_preview_peakshape(
                    ms_files,
                    mz_mean,
                    mz_width,
                    rt,
                    rt_min,
                    rt_max,
                    image_label,
                    wdir,
                    peak_label=peak_label,
                    colors=file_colors,
                )

            if os.path.isfile(fn):
                src = T.png_fn_to_src(fn)
            else:
                src = None

            _id = {"index": peak_label, "type": "image"}
            image_id = f"image-{i}"
            images.append(
                html.A(
                    id=_id,
                    children=html.Img(
                        src=src, id=image_id, style={"margin": "0px", "height": "150px"}
                    ),
                )
            )
            images.append(
                dbc.Tooltip(peak_label, target=image_id, style={"font-size": "50"})
            )
        fsc.set(f"{wdir}-updating", False)
        return images

    @app.callback(
        Output("pko-image-clicked", "children"),
        # Input needs brakets to make prevent_initital_call work
        [Input({"type": "image", "index": ALL}, "n_clicks")],
        prevent_initial_call=True,
    )
    def pko_image_clicked(ndx):
        if ndx is None or len(ndx) == 0:
            raise PreventUpdate
        ctx = dash.callback_context
        clicked = ctx.triggered[0]["prop_id"]
        clicked = clicked.replace('{"index":"', "")
        clicked = clicked.split('","type":')[0].replace("\\", "")
        if len(dash.callback_context.triggered) > 1:
            raise PreventUpdate
        return clicked

    @app.callback(
        Output({"index": "pko-drop-target-output", "type": "output"}, "children"),
        Input("pko-drop-target", "n_clicks"),
        State("pko-dropdown", "value"),
        State("wdir", "children"),
    )
    def plk_delete(n_clicks, peak_ndx, wdir):
        if n_clicks is None:
            raise PreventUpdate
        targets = T.get_targets(wdir).reset_index()
        peak_label = targets.loc[peak_ndx, "peak_label"]
        targets = targets.drop(peak_ndx, axis=0)
        T.write_targets(targets, wdir)
        return dbc.Alert(f"{peak_label} removed from targets.", color="info")


    # Callback for detecting RT for a single target
    @app.callback(
        Output({"index": "pko-detect-rt-output", "type": "output"}, "children"),
        Input("pko-detect-rt", "n_clicks"),
        State("pko-dropdown", "value"),
        State("pko-ms-selection", "value"),
        State("wdir", "children"),
    )
    def detect_rt_single(n_clicks, peak_label_ndx, ms_selection, wdir):
        # Check if button was clicked
        if n_clicks is None:
            raise PreventUpdate
        
        # Check if a target is selected
        if peak_label_ndx is None:
            return dbc.Alert("No target selected in the dropdown", color="warning")
        
        # Get targets and MS files
        targets = T.get_targets(wdir).reset_index()
        
        if ms_selection == "peakopt":
            ms_files = T.get_ms_fns_for_peakopt(wdir)
        elif ms_selection == "all":
            ms_files = T.get_ms_fns(wdir)
        
        # Get the actual peak label string from the index
        peak_label = targets.at[peak_label_ndx, 'peak_label']
        
        # Initialize Mint
        mint = Mint()
        mint.targets = targets
        mint.ms_files = ms_files
        
        # Only detect RT for this specific peak
        mint.opt.detect_largest_peak_rt(peak_labels=[peak_label])
        T.write_targets(mint.targets, wdir)
        
        return dbc.Alert(f"Detected RT for {peak_label}", color="info")

    # Callback for detecting RT for all targets
    @app.callback(
        Output({"index": "pko-detect-rt-for-all-output", "type": "output"}, "children"),
        Input("pko-detect-rt-for-all", "n_clicks"),
        State("pko-ms-selection", "value"),
        State("wdir", "children"),
    )
    def detect_rt_all(n_clicks, ms_selection, wdir):
        if n_clicks is None:
            raise PreventUpdate
        
        logging.warning(f'Running RT detection for all targets in {wdir}')
        
        # Get targets and MS files
        targets = T.get_targets(wdir)
        
        if ms_selection == "peakopt":
            ms_files = T.get_ms_fns_for_peakopt(wdir)
        elif ms_selection == "all":
            ms_files = T.get_ms_fns(wdir)
        
        # Initialize Mint
        mint = Mint()
        mint.targets = targets
        mint.ms_files = ms_files
        
        # Detect RT for all targets
        mint.opt.detect_largest_peak_rt()
        T.write_targets(mint.targets, wdir)
        
        return dbc.Alert("Detected RT for all targets", color="success")

    # Callback for processing a single target
    @app.callback(
        Output({"index": "pko-detect-rtspan-output", "type": "output"}, "children"),
        Input("pko-detect-rtspan", "n_clicks"),
        State("pko-dropdown", "value"),
        State("pko-ms-selection", "value"),
        State("wdir", "children"),
    )
    def detect_rt_span_single(n_clicks, peak_label_ndx, ms_selection, wdir):
        # Check if button was clicked
        if n_clicks is None:
            raise PreventUpdate
        
        # Check if a target is selected
        if peak_label_ndx is None:
            return dbc.Alert("No target selected in the dropdown", color="warning")
        
        # Get targets and MS files
        targets = T.get_targets(wdir).reset_index()
        
        if ms_selection == "peakopt":
            ms_files = T.get_ms_fns_for_peakopt(wdir)
        elif ms_selection == "all":
            ms_files = T.get_ms_fns(wdir)
        
        # Get the actual peak label string from the index
        peak_label = targets.at[peak_label_ndx, 'peak_label']
        
        # Initialize Mint
        mint = Mint()
        mint.targets = targets
        mint.ms_files = ms_files
        
        # Only optimize this specific peak
        mint.opt.rt_min_max(peak_labels=[peak_label], rel_height=0.8)
        T.write_targets(mint.targets, wdir)
        
        return dbc.Alert(f"Optimized RT span for {peak_label}", color="info")

    # Callback for processing all targets
    @app.callback(
        Output({"index": "pko-detect-rtspan-for-all-output", "type": "output"}, "children"),
        Input("pko-detect-rtspan-for-all", "n_clicks"),
        State("pko-ms-selection", "value"),
        State("wdir", "children"),
    )
    def detect_rt_span_all(n_clicks, ms_selection, wdir):
        if n_clicks is None:
            raise PreventUpdate
        
        logging.warning(f'Running RT span detection for all targets in {wdir}')
        
        # Get targets and MS files
        targets = T.get_targets(wdir)
        
        if ms_selection == "peakopt":
            ms_files = T.get_ms_fns_for_peakopt(wdir)
        elif ms_selection == "all":
            ms_files = T.get_ms_fns(wdir)
        
        # Initialize Mint
        mint = Mint()
        mint.targets = targets
        mint.ms_files = ms_files
        
        # Optimize all targets (don't specify peak_labels)
        mint.opt.rt_min_max(rel_height=0.8)
        T.write_targets(mint.targets, wdir)
        
        return dbc.Alert("Optimized RT span for all targets", color="success")

    @app.callback(
        Output("pko-full-figure", "figure"),
        Output("pko-zoom-figure", "figure"),
        Input("pko-dropdown", "value"),
        Input("pko-set-rt", "n_clicks"),
        Input("pko-figure", "figure"),
        Input({"index": "pko-detect-rt-output", "type": "output"}, "children"),
        Input({"index": "pko-detect-rt-for-all-output", "type": "output"}, "children"),
        Input({"index": "pko-detect-rtspan-output", "type": "output"}, "children"),
        Input({"index": "pko-detect-rtspan-for-all-output", "type": "output"}, "children"),
        State("pko-ms-selection", "value"),
        State("wdir", "children"),
    )
    def update_figures(
        peak_label_ndx, 
        set_rt_clicks, 
        pko_figure,
        # Add parameters for the new inputs
        rt_detected_single,
        rt_detected_all,
        rtspan_detected_single,
        rtspan_detected_all,
        ms_selection, 
        wdir
    ):
        # Basic figure generation logic
        if peak_label_ndx is None:
            raise PreventUpdate

        # Retrieve targets
        targets = T.get_targets(wdir).reset_index()
        
        # Determine MS files
        if ms_selection == "peakopt":
            ms_files = T.get_ms_fns_for_peakopt(wdir)
        elif ms_selection == "all":
            ms_files = T.get_ms_fns(wdir)

        # Get current target details
        peak_label_ndx = peak_label_ndx % len(targets)
        mz_mean, mz_width, rt, rt_min, rt_max, label = targets.loc[
            peak_label_ndx, 
            ["mz_mean", "mz_width", "rt", "rt_min", "rt_max", "peak_label"]
        ]
        
        # Determine RT range
        margin = 30
        #if not rt:
        #    rt = np.mean([rt_min, rt_max]) if rt_min and rt_max else None
        #rt_min = max(0, rt - margin) if rt is not None else 0
        #rt_max = rt + margin if rt is not None else 1000
        
        # Create figures
        full_fig = go.Figure()
        zoom_fig = go.Figure()

        # Add chromatogram traces
        for fn in ms_files:
            chrom = T.get_chromatogram(fn, mz_mean, mz_width, wdir)
                        
            # Full figure trace
            full_fig.add_trace(go.Scattergl(
                x=chrom['scan_time'], 
                y=chrom['intensity'], 
                mode='markers', 
                fill='tozeroy',
                marker=dict(size=3),
                name=os.path.basename(fn)
            ))
            
            # Zoom figure trace (filtered to RT range)
            zoom_chrom = chrom[
                (chrom['scan_time'] >= rt_min) & 
                (chrom['scan_time'] <= rt_max)
            ]
            zoom_fig.add_trace(go.Scattergl(
                x=zoom_chrom['scan_time'], 
                y=zoom_chrom['intensity'], 
                mode='markers',
                fill='tozeroy',
                marker=dict(size=3),
                name=os.path.basename(fn)
            ))

        # Update layouts
        for fig in [full_fig, zoom_fig]:
            fig.update_layout(
                title=f"{label} (m/z={mz_mean:.2f})",
                xaxis_title="Scan Time [s]",
                yaxis_title="Intensity"
            )

        # Add RT markers        
        if rt:
            rt_line = dict(type='line', x0=rt, x1=rt, y0=0, y1=1, yref='paper')
            full_fig.add_shape(rt_line, line=dict(color='red', dash='dash'))
            zoom_fig.add_shape(rt_line, line=dict(color='red', dash='dash'))

        # Update layouts with reduced margins and full width
        for fig in [full_fig, zoom_fig]:
            fig.update_layout(
                title=f"{label} (m/z={mz_mean:.2f})",
                xaxis_title="Scan Time [s]",
                yaxis_title="Intensity",
                width=None,
                autosize=True,
                showlegend=False,
            )
        
        return full_fig, zoom_fig