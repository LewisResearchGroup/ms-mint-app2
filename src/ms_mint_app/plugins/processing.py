import os

from pathlib import Path as P

import dash
from dash import html

import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dcc import send_file, send_bytes

from dash.dependencies import Input, Output, State
from dash_tabulator import DashTabulator

from ms_mint.Mint import Mint

from .. import tools as T
from ..plugin_interface import PluginInterface
from dash import dcc

from ms_mint.standards import RESULTS_COLUMNS

_label = "Processing"

class ProcessingPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 7
        print(f'Initiated {_label} plugin')

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)
    
    def outputs(self):
        return _outputs

_property_options = T.list_to_options(RESULTS_COLUMNS)


tabulator_options = {
    "groupBy": "ms_type",
    "selectable": True,
    "headerFilterLiveFilterDelay": 3000,
    "layout": "fitDataFill",
    # "height": "900px",
}

downloadButtonType = {
    "css": "btn btn-primary",
    "text": "Export",
    "type": "csv",
    "filename": "Targets",
}

clearFilterButtonType = {"css": "btn btn-outline-dark", "text": "Clear Filters"}


results_table = html.Div(
    id="results-table-container",
    style={"minHeight": 100, "margin": "50px 50px 0px 0px"},
    children=[
        DashTabulator(
            id="results-table",
            columns=[
                {"formatter": "rowSelection", "titleFormatter": "rowSelection",
                 "titleFormatterParams": { "rowRange": "active"},  # only toggle the values of the active filtered rows,
                "hozAlign": "center", "headerSort": False, "width": "1px", "frozen": True,},
                {"title":"ms_file_label", "field":"ms_file_label", "headerTooltip": "This is a tooltip", },
                {"title":"peak_label", "field":"peak_label", "headerTooltip": "This is a tooltip", },
                {"title":"mz_mean", "field":"mz_mean", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"mz_width", "field":"mz_width", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"mz", "field":"mz", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"rt", "field":"rt", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"rt_min", "field":"rt_min", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"rt_max", "field":"rt_max", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"rt_unit", "field":"rt_unit", "hozAlign": "center", "headerTooltip": "This is a tooltip"},
                {"title":"intensity_threshold", "field":"intensity_threshold", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"polarity", "field":"polarity", "hozAlign": "center", "headerTooltip": "This is a tooltip"},
                {"title":"filterLine", "field":"filterLine", "hozAlign": "center", "headerTooltip": "This is a tooltip"},
                {"title":"ms_type", "field":"ms_type", "hozAlign": "center", "headerTooltip": "This is a tooltip"},
                {"title":"category", "field":"category", "headerTooltip": "This is a tooltip"},
                {"title":"target_filename", "field":"target_filename", "headerTooltip": "This is a tooltip"},
                {"title":"peak_area", "field":"peak_area", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"peak_area_top3", "field":"peak_area_top3", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"peak_n_datapoints", "field":"peak_n_datapoints", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"peak_max", "field":"peak_max", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"peak_rt_of_max", "field":"peak_rt_of_max", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"peak_min", "field":"peak_min", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"peak_median", "field":"peak_median", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"peak_mean", "field":"peak_mean", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"peak_delta_int", "field":"peak_delta_int", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"peak_shape_rt", "field":"peak_shape_rt", "headerTooltip": "This is a tooltip"},
                {"title":"peak_shape_int", "field":"peak_shape_int", "headerTooltip": "This is a tooltip"},
                {"title":"peak_mass_diff_25pc", "field":"peak_mass_diff_25pc", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"peak_mass_diff_50pc", "field":"peak_mass_diff_50pc", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"peak_mass_diff_75pc", "field":"peak_mass_diff_75pc", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"peak_score", "field":"peak_score", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"total_intensity", "field":"total_intensity", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
                {"title":"ms_file_size_MB", "field":"ms_file_size_MB", "hozAlign": "right", "headerTooltip": "This is a tooltip"},
            ],
            options=tabulator_options,
            downloadButtonType=downloadButtonType,
            clearFilterButtonType=clearFilterButtonType,
        )
    ],
)


_layout = html.Div(
    [
        html.H3("Run MINT"),
        dbc.Row([
            dbc.Col(dbc.Button("Run MINT", id="run-mint", style={"width": "100%"})),
            dbc.Col(dbc.Button("Download all results", id="res-download", style={"width": "100%"}, color='secondary')),
            dbc.Col(html.Div([
                        dbc.Button("Download dense matrix", id="res-download-peakmax", style={"width": "100%"}, color='secondary'),
                        dcc.Dropdown(id='proc-download-property', options=_property_options, value='peak_area_top3'),
                        dcc.Checklist(id='proc-download-options', options=['Transposed']),
                    ])
            ),
            dbc.Col(dbc.Button("Delete results", id="res-delete", style={"width": "100%"}, color='danger')),
        ]),
        html.Div(
                id="processing-progress-container",
                style={"display": "none"},
                children=[
                    html.P("Processing files..."),
                    dbc.Progress(id="processing-progress-bar", animated=True, striped=True,
                                 max=100, label="Processing...")
                ]
            ),
        dcc.Interval(id="processing-poll-interval", interval=500, n_intervals=0, disabled=False),
        dcc.Loading(results_table)
    ]
)

_outputs = html.Div(
    id="run-outputs",
    children=[
        html.Div(
            id={"index": "run-mint-output", "type": "output"},
            style={"visibility": "hidden"},
        ),
    ],
)


def layout():
    return _layout


def callbacks(app, fsc, cache):
    @app.callback(
        Output("results-table", "data"),
        Input({"index": "run-mint-output", "type": "output"}, "children"),
        Input("res-delete", "n_clicks"),
        Input("wdir", "children"),
    )
    def get_results(value, n_clicks, wdir):
        ctx = dash.callback_context
        prop_id = ctx.triggered[0]["prop_id"]

        if prop_id == "res-delete.n_clicks":
            if n_clicks is None:
                raise PreventUpdate
            os.remove(T.get_results_fn(wdir))
            return []
        else:
            try:
                df = T.get_results(wdir)
                return df.to_dict("records")
            except Exception:
                raise PreventUpdate


    @app.callback(
        Output("processing-progress-bar", "value"),
        Output("processing-progress-bar", "label"),
        Output("processing-progress-container", "style", allow_duplicate=True),
        Input("processing-poll-interval", "n_intervals"),
        prevent_initial_call=True
    )
    def update_progress(n):
        return fsc.get("progress"), "Processing...", {"display": "none"}

    @app.callback(
        [
            Output("res-download-data", "data"),
            Input("res-download", "n_clicks"),
            Input("res-download-peakmax", "n_clicks"),
            State("proc-download-property", "value"),
            State('proc-download-options', 'value'),
            State("wdir", "children"),
        ]
    )
    def download_results(n_clicks, n_clicks_peakmax, property, options, wdir):
        if (n_clicks is None) and (n_clicks_peakmax is None):
            raise PreventUpdate
        ctx = dash.callback_context

        prop_id = ctx.triggered[0]["prop_id"]

        if prop_id == "res-download.n_clicks":
            fn = T.get_results_fn(wdir)
            workspace = os.path.basename(wdir)
            return [
                send_file(fn, filename=f"{T.today()}-MINT__{workspace}-long.csv")
            ]

        elif prop_id == "res-download-peakmax.n_clicks":
            workspace = os.path.basename(wdir)
            results = T.get_results(wdir)
            df = results.pivot_table(property, "ms_file_label", "peak_label")
            if options is not None and 'Transposed' in options:
                df = df.T
            buffer = T.df_to_in_memory_excel_file(df)
            return [
                send_bytes(
                    buffer, filename=f"{T.today()}-MINT__{workspace}__results_{property}.xlsx"
                )
            ]

    @app.callback(
        Output({"index": "run-mint-output", "type": "output"}, "children"),
        Output("processing-progress-container", "style", allow_duplicate=True),
        Input("run-mint", "n_clicks"),
        State("wdir", "children"),
        prevent_initial_call=True
    )
    def run_mint(n_clicks, wdir):
        if n_clicks is None:
            raise PreventUpdate

        def set_progress(x):
            fsc.set("progress", x)

        mint = Mint(verbose=False, progress_callback=set_progress)
        targets_fn = T.get_targets_fn(wdir)
        output_fn = T.get_results_fn(wdir)
        try:
            mint.load_targets(targets_fn)
            mint.targets = mint.targets[mint.targets.rt_min.notna() & mint.targets.rt_max.notna()]
            mint.ms_files = T.get_ms_fns(wdir)
            mint.run(fn=output_fn)
        except Exception as e:
            return dbc.Alert(str(e), color="danger")
        return dbc.Alert("Finished running MINT", color="success"), {"display": "block"}
