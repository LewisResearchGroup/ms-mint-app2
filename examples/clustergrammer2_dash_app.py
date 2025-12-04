"""
Minimal Dash + Clustergrammer2 (clustergrammer-gl) example.

Renders the same mock compound x sample matrix used by the Clustergrammer
example, but with the WebGL Clustergrammer2 UI embedded in Dash.

Run:
1) pip install dash pandas numpy clustergrammer2
2) python examples/clustergrammer2_dash_app.py [--compounds 400 --samples 1000 --seed 7 --port 8051]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from clustergrammer2 import Network
from dash import Dash, html


DATA_PATH = Path(__file__).with_name("mock_compound_sample_matrix.csv")
CG2_CDN_VERSION = "0.25.0"  # version of clustergrammer-gl to load from the CDN
VIZ_WIDTH = 900
VIZ_HEIGHT = 900
# add a small buffer to avoid scrollbars while keeping a 1:1 pixel ratio (prevents canvas text distortion)
FRAME_HEIGHT = VIZ_HEIGHT + 40


def build_mock_matrix(num_compounds: int = 80, num_samples: int = 100, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    compounds = [f"Compound {i:03d}" for i in range(1, num_compounds + 1)]
    samples = [f"Sample {i:03d}" for i in range(1, num_samples + 1)]
    data = rng.normal(loc=0.2, scale=1.1, size=(num_compounds, num_samples))
    return pd.DataFrame(data, index=compounds, columns=samples).round(4)


def load_or_create_mock_data(num_compounds: int, num_samples: int, seed: int) -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, index_col=0)
        if df.shape == (num_compounds, num_samples):
            return df
    df = build_mock_matrix(num_compounds, num_samples, seed)
    DATA_PATH.write_text(df.to_csv())
    return df


def build_clustergrammer2_srcdoc(matrix: pd.DataFrame) -> str:
    net = Network()
    net.load_df(matrix)
    net.cluster()

    net_json_js = json.dumps(json.loads(net.export_net_json()))
    cdn_root = f"https://unpkg.com/clustergrammer-gl@{CG2_CDN_VERSION}"

    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{cdn_root}/lib/css/bootstrap.css">
    <link rel="stylesheet" href="{cdn_root}/lib/css/font-awesome.min.css">
    <link rel="stylesheet" href="{cdn_root}/lib/css/custom_scrolling.css">
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
      }}
      #cg2-root {{
        box-sizing: border-box;
        width: {VIZ_WIDTH}px;
        height: {FRAME_HEIGHT}px;
        overflow: hidden;
      }}
      #cg2-container {{
        width: {VIZ_WIDTH}px;
        height: {VIZ_HEIGHT}px;
      }}
    </style>
    <script src="{cdn_root}/clustergrammer-gl.js"></script>
  </head>
  <body>
    <div id="cg2-root">
      <div id="cg2-container"></div>
    </div>
    <script>
      var network_data = {net_json_js};
      var args = {{
        container: document.getElementById('cg2-container'),
        network: network_data,
        viz_width: {VIZ_WIDTH},
        viz_height: {VIZ_HEIGHT}
      }};
      CGM(args);
    </script>
  </body>
</html>
"""


def make_app(matrix: pd.DataFrame) -> Dash:
    iframe_src = build_clustergrammer2_srcdoc(matrix)
    sample_preview = matrix.iloc[:8, : min(6, matrix.shape[1])]

    app = Dash(__name__)
    app.title = "Clustergrammer2 mock matrix"

    app.layout = html.Div(
        [
            html.H3(f"Clustergrammer2 mock matrix ({matrix.shape[0]} compounds x {matrix.shape[1]} samples)"),
            html.Iframe(
                title="Clustergrammer2 heatmap",
                srcDoc=iframe_src,
                style={"border": "1px solid #ccc", "width": f"{VIZ_WIDTH}px", "height": f"{FRAME_HEIGHT}px"},
            ),
            html.H4("Data preview (first 8 rows x up to 6 cols)"),
            html.Pre(sample_preview.to_csv()),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal Clustergrammer2 Dash app.")
    parser.add_argument("--compounds", type=int, default=80, help="Number of compounds (rows) to generate.")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples (columns) to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--port", type=int, default=8051, help="Port for the Dash server.")
    args = parser.parse_args()

    matrix = load_or_create_mock_data(args.compounds, args.samples, args.seed)
    app = make_app(matrix)
    app.run(host="0.0.0.0", port=args.port, debug=True)


if __name__ == "__main__":
    main()
