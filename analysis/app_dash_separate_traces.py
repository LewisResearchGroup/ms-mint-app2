import numpy as np
from dash import Dash, dcc, html
import plotly.graph_objects as go

# --------------------------
# Generate mock chromatograms
# --------------------------
def generate_mock_chromatograms(n_chrom=1000, n_points=1200, seed=0):
    rng = np.random.default_rng(seed)
    chroms = []

    for _ in range(n_chrom):
        x = np.linspace(0, 60, n_points)
        baseline = rng.normal(0, 0.2, n_points)

        center = rng.uniform(10, 50)
        width = rng.uniform(2, 8)
        peak = np.exp(-0.5 * ((x - center) / width)**2) * rng.uniform(100, 3000)

        y = baseline + peak
        y[y < 0] = 0
        chroms.append((x, y))

    return chroms

CHROMS = generate_mock_chromatograms()

# --------------------------
# Build figure (many traces)
# --------------------------
def build_fig_separate(chroms):
    fig = go.Figure()

    for x, y in chroms:
        fig.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode="lines",
            line=dict(width=1, color="rgba(0,0,0,0.25)"),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title="Retention Time (s)",
        yaxis_title="Intensity"
    )
    return fig

SEPARATE_FIG = build_fig_separate(CHROMS)

# --------------------------
# Dash app (4 cards)
# --------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Separate Traces — 1000 Chromatograms × 1200 Points"),

    html.Div([
        dcc.Graph(id="g1", figure=SEPARATE_FIG, style={"height": "350px"}),
        dcc.Graph(id="g2", figure=SEPARATE_FIG, style={"height": "350px"}),
        dcc.Graph(id="g3", figure=SEPARATE_FIG, style={"height": "350px"}),
        dcc.Graph(id="g4", figure=SEPARATE_FIG, style={"height": "350px"}),
        dcc.Graph(id="g5", figure=SEPARATE_FIG, style={"height": "350px"}),
        dcc.Graph(id="g6", figure=SEPARATE_FIG, style={"height": "350px"}),
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr"})
])

if __name__ == "__main__":
    app.run(debug=True)
