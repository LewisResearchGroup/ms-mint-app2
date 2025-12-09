import numpy as np
import plotly.graph_objects as go
from plotly_resampler import FigureResampler
from dash import Dash, dcc, html


# --------------------------
# Generate mock chromatograms
# --------------------------
def generate_mock_chromatograms(n_chrom=1000, n_points=1200, seed=0):
    rng = np.random.default_rng(seed)
    chroms = []

    for _ in range(n_chrom):
        x = np.linspace(0, 60, n_points)

        # 1) UGLY, DRIFTING BASELINE
        # random walk + low-frequency sine + noise
        drift = rng.normal(0, 0.02, n_points).cumsum()
        sine = 0.3 * np.sin(2 * np.pi * x / rng.uniform(30, 80))
        baseline = 1.0 + drift + sine + rng.normal(0, 0.05, n_points)

        # 2) RANDOM NUMBER OF PEAKS (some narrow, some wide, some tiny)
        n_peaks = rng.integers(3, 15)   # many overlapping peaks
        peaks = np.zeros_like(x)
        for _p in range(n_peaks):
            center = rng.uniform(0, 60)
            # some narrow, some super broad
            width = rng.uniform(0.2, 10.0)
            height = rng.uniform(50, 8000)
            peaks += height * np.exp(-0.5 * ((x - center) / width) ** 2)

        # 3) RANDOM SPIKES (salt-and-pepper artefacts)
        spikes = np.zeros_like(x)
        n_spikes = rng.integers(5, 30)
        spike_idx = rng.integers(0, n_points, size=n_spikes)
        spikes[spike_idx] = rng.uniform(100, 20000, size=n_spikes)

        # 4) OCCASIONAL ZEROED-OUT REGIONS (e.g. detector dropout)
        y = baseline + peaks + spikes
        y[y < 0] = 0.0

        n_zero_blocks = rng.integers(0, 4)
        for _zb in range(n_zero_blocks):
            start = rng.integers(0, n_points - 10)
            length = rng.integers(5, 80)
            end = min(n_points, start + length)
            y[start:end] = 0.0

        # 5) Some random jitter on top
        y += rng.normal(0, 0.1 * np.maximum(1, y.max()), size=n_points)
        y[y < 0] = 0.0

        chroms.append((x, y))

    return chroms


CHROMS = generate_mock_chromatograms()


# --------------------------
# Build resampled figure
# --------------------------
def build_fig_resampled(chroms):

    # A list of colors to cycle through
    color_cycle = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # yellow-green
        "#17becf",  # cyan
    ]
    n_colors = len(color_cycle)

    # 1. Build a normal figure first
    fig = go.Figure()

    for i, (x, y) in enumerate(chroms):
        color = color_cycle[i % n_colors]   # cycle colors
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="lines",
                line=dict(width=1, color=color),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title="Retention Time (s)",
        yaxis_title="Intensity",
    )

    # 2. Convert into a resampled figure
    fr_fig = FigureResampler(
        fig,
        default_n_shown_samples=500,
        convert_existing_traces=True,  # interpret x/y as full-resolution hf_x/hf_y
    )

    return fr_fig


RESAMPLED_FIG = build_fig_resampled(CHROMS)


# --------------------------
# Dash app (1 big plot)
# --------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H2("plotly-resampler — 1000 Chromatograms × 1200 Points (Colored Traces)"),

    html.Div([
        dcc.Graph(id="g1", figure=RESAMPLED_FIG,
                  style={"height": "850px", "width": "1500px"}),
    ], style={"display": "flex", "justifyContent": "center", "alignItems": "center"})
])

if __name__ == "__main__":
    app.run(debug=True, port=8050)
