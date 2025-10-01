import logging
from pathlib import Path as P

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_tabulator import DashTabulator
from scipy.signal import savgol_filter, find_peaks

from .. import tools as T

# Replace with your actual test file
test_file = '/home/mario/Documents/MINT_test/SRMDataForSoren/2024_10_16_Chr_STD0_Rep1.mzXML'


class MS2BrowserPlugin:
    label = "MS2 Browser"
    order = 999
    df_static = None

    @staticmethod
    def layout(**kwargs):
        return _layout

    @staticmethod
    def create_channel_timeline_plot(df, channel):
        if "ESI SRM ms2" in channel:
            mask = df["filterLine"] == channel
        else:
            mask = df["filterLine_to_ELMAVEN"] == channel

        filtered = df[mask & (df["intensity"] != 0)].copy()
        if filtered.empty:
            return go.Figure().update_layout(title=f"No MS2 spectra for channel {channel}")

        expanded = []
        for _, row in filtered.iterrows():
            scan_time = row["scan_time"]
            if "ESI SRM ms2" in channel:
                channel = row["filterLine"]
            else:
                channel = row["filterLine_to_ELMAVEN"]

            expanded.append({
                "scan_time": scan_time,
                "channel": channel,
                "mz": row["mz"],
                "intensity": row["intensity"],
            })

        df_expanded = pd.DataFrame(expanded)

        fig = go.Figure()
        for _, row in df_expanded.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["scan_time"], row["scan_time"]],
                y=[0, row["intensity"]],
                mode="lines",
                line=dict(width=1, color="black"),
                hoverinfo="text",
                hovertext=f"Scan time: {row['scan_time']}<br>"
                          f"m/z: {row['mz']:.4f}<br>"
                          f"Intensity: {row['intensity']:.1f}<br>"
                          f"Channel: {row['channel']}"
            ))

        fig.update_layout(
            title=f"Fragmentation Pattern Over Time for Channel {channel}",
            xaxis_title="Scan Time",
            yaxis_title="Fragment Intensity",
            yaxis_title_standoff=20,
            showlegend=False,
            height=350,
            margin=dict(l=60, r=20, t=30, b=40),
        )
        return fig

    @staticmethod
    def detect_ms2_peak(self, smooth=True, debug=False):
        """
        Detects the main chromatographic peak in MS2 fragment time series.

        Returns:
            - peak_time (float)
            - peak_intensity (float)
            - auc (float)
            - total (float)
            - smoothed (array)
            - peaks (array)
            - props (dict)
        """
        x = self["scan_time"].values
        y = self["intensity"].values

        # Smooth
        y_smooth = savgol_filter(y, window_length=11, polyorder=2) if smooth and len(y) >= 11 else y

        # Peak detection
        peaks, props = find_peaks(y_smooth, height=1000, prominence=500, width=3)

        if len(peaks) == 0:
            return None, None, None, None, y_smooth, [], {}

        main_peak_idx = peaks[np.argmax(props["peak_heights"])]
        peak_time = x[main_peak_idx]
        peak_intensity = props["peak_heights"].max()

        # AUC around ±5 scans
        start = max(0, main_peak_idx - 5)
        end = min(len(y), main_peak_idx + 6)
        auc = np.trapz(y[start:end], x[start:end])
        total = y[start:end].sum()

        return peak_time, peak_intensity, auc, total, y_smooth, peaks, props

    # @staticmethod
    # def create_debuggable_ms2_plot(df, show_debug=False):
    #     peak_time, peak_intensity, auc, total, y_smooth, peaks, props = (
    #         MS2BrowserPlugin.detect_ms2_peak(df)
    #     )
    #
    #     x = df["scan_time"].values
    #     y = df["intensity"].values
    #
    #     fig = go.Figure()
    #
    #     # Base stick plot
    #     fig.add_trace(go.Scatter(
    #         x=x,
    #         y=y,
    #         mode="lines",
    #         line=dict(width=1, color="black"),
    #         name="Raw"
    #     ))
    #
    #     if show_debug:
    #         fig.add_trace(go.Scatter(
    #             x=x,
    #             y=y_smooth,
    #             mode="lines",
    #             line=dict(width=1, color="blue"),
    #             name="Smoothed"
    #         ))
    #
    #         fig.add_trace(go.Scatter(
    #             x=x[peaks],
    #             y=props["peak_heights"],
    #             mode="markers",
    #             marker=dict(size=8, color="red"),
    #             name="Detected Peaks"
    #         ))
    #
    #     fig.update_layout(
    #         title=f"MS2 Fragmentation Pattern — Peak Intensity: {peak_intensity:,.0f}, AUC: {auc:,.0f} @ "
    #               f"t={peak_time:.2f}" if peak_intensity else "No Peak Detected",
    #         xaxis_title="Scan Time",
    #         yaxis_title="Fragment Intensity",
    #         showlegend=show_debug,
    #         height=350,
    #     )
    #
    #     return fig

    @staticmethod
    def outputs():
        return None

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)


options = {
    "layout": "fitDataStretch",
    "selectable": 1,
    "pagination": "local",
    "paginationSize": 10,
    "movableColumns": True,
    "resizableColumns": True,
}

clearFilterButtonType = {"css": "btn btn-outline-dark", "text": "Clear Filters"}

ms2_table = html.Div(
    id="ms2-table-container",
    style={"Height": 0, "marginTop": "30px"},
    children=[
        DashTabulator(
            id="ms2-table",
            columns=[
                {"title": "File Name", "field": "file_name", "headerFilter": True}, ],
            options=options,
        )
    ],
)

columns = [
    {"title": 'scan_id', "field": 'scan_id', "headerFilter": True},
    {"title": 'ms_level', "field": 'ms_level', "headerFilter": True},
    {"title": 'polarity', "field": 'polarity', "headerFilter": True},
    {"title": 'scan_time', "field": 'scan_time', "headerFilter": True},
    {"title": 'mz', "field": 'mz', "headerFilter": True},
    {"title": 'intensity', "field": 'intensity', "headerFilter": True},
    {"title": 'mz_precursor', "field": 'mz_precursor', "headerFilter": True},
    {"title": 'filterLine', "field": 'filterLine', "headerFilter": True},
    {"title": 'filterLine_to_ELMAVEN', "field": 'filterLine_to_ELMAVEN', "headerFilter": True},
]

content_table = html.Div(
    id="content-table-container",
    style={"Height": 0, "marginTop": "30px"},
    children=[
        DashTabulator(
            id="content-table",
            columns=columns,
            options={
                "layout": "fitDataStretch",
                "pagination": "local",
                "paginationSize": 10,
                "movableColumns": True,
                "resizableColumns": True,
            },
        )
    ],
)

dropdown = dcc.Dropdown(
    id="channel-selector",
    placeholder="Select a channel...",
    style={"width": "400px", "marginBottom": "10px"},
)

debug_toggle = dcc.Checklist(
    options=[{"label": "Debug Peak Detection", "value": "debug"}],
    value=[],
    id="ms2-debug-toggle",
    style={"marginBottom": "10px"}
)

_layout = html.Div(
    [
        dbc.Row([
            dbc.Col([dcc.Loading(ms2_table)], width=3),
            dbc.Col([dcc.Loading(content_table)], width=9),
        ]),
        html.Div([
            html.H4("Select Channel"),
            dropdown,
            debug_toggle,
        ], style={"marginTop": "20px"}),
        dcc.Loading(
            html.Div(id="channel-fragment-plot-container", style={"marginTop": "30px"})
        )
    ]
)


def layout():
    return _layout


def callbacks(app, fsc, cache):
    @app.callback(
        Output("ms2-table", "data"),
        Input("wdir", "data"),
        State("active-workspace", "children"),
    )
    def ms2_table(wdir, workspace):
        ms_files = T.get_ms_fns(wdir)
        logging.info(f"# MS2 Files in {wdir} {workspace} {len(ms_files)}")

        data = pd.DataFrame(
            {"file_name": [P(fn).stem for fn in ms_files if T.get_ms_level_from_filename(fn) == "ms2"]}
        )
        return data.to_dict("records")

    @app.callback(
        Output("content-table", "data"),
        Output("channel-selector", "options"),
        Output("channel-selector", "value"),
        Input("wdir", "data"),
        Input("ms2-table", "rowClicked"),
    )
    def content_table(wdir, row):
        ff = {fo.stem: fo for fo in P(wdir).joinpath("ms_files").glob("*.feather")}
        if row is None or row["file_name"] not in ff:
            raise PreventUpdate
        df = pd.read_feather(ff[row["file_name"]])
        MS2BrowserPlugin.df_static = df
        # channels
        cols = [c for c in ["filterLine", "filterLine_to_ELMAVEN"] if c in df.columns]
        values = df[cols].values.flatten().tolist()
        clean_values = [v for v in values if isinstance(v, str) and pd.notna(v)]
        channel_options = sorted(list(np.unique(clean_values)))

        return df.to_dict("records"), channel_options, channel_options[0]

    @app.callback(
        Output("channel-fragment-plot-container", "children"),
        Input("channel-selector", "value"),
        Input("ms2-debug-toggle", "value"),
        prevent_initial_call=True,
    )
    def update_channel_plot(channel, debug_flags):
        if channel is None:
            return dcc.Markdown("⚠️ Select a channel to view fragmentation over time.")
        df = MS2BrowserPlugin.df_static.copy()
        fig = MS2BrowserPlugin.create_channel_timeline_plot(df, channel)
        return dcc.Graph(figure=fig)
