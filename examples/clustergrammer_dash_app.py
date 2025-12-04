"""
Minimal Dash + Clustergrammer example.

Renders an 80 (compounds) x 100 (samples) mock matrix (or a larger one via CLI
flags) with the stock Clustergrammer UI embedded in Dash. No custom overrides.

Run:
1) pip install dash pandas numpy clustergrammer
2) python examples/clustergrammer_dash_app.py [--compounds 400 --samples 1000 --seed 7 --port 8050]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, html
from clustergrammer import Network

# clustergrammer-py targets older pandas and still uses .ix; provide a small shim.
class _IxShim:
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        return self.obj.loc[key]

    def __setitem__(self, key, value):
        self.obj.loc[key] = value


if not hasattr(pd.DataFrame, "ix"):
    pd.DataFrame.ix = property(lambda self: _IxShim(self))  # type: ignore[attr-defined]


DATA_PATH = Path(__file__).with_name("mock_compound_sample_matrix.csv")


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


def build_clustergrammer_srcdoc(matrix: pd.DataFrame) -> str:
    net = Network()
    net.load_df(matrix)
    net.cluster()

    net_json = json.loads(net.export_net_json())
    net_json_js = json.dumps(net_json)

    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- Use official Clustergrammer assets -->
    <link rel="stylesheet" href="https://maayanlab.github.io/clustergrammer/lib/css/bootstrap.css">
    <link rel="stylesheet" href="https://maayanlab.github.io/clustergrammer/lib/css/font-awesome.min.css">
    <link rel="stylesheet" href="https://maayanlab.github.io/clustergrammer/css/custom.css">
    <style>
      html, body, #cg-container {{
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
      }}
    </style>
    <script src="https://maayanlab.github.io/clustergrammer/lib/js/d3.js"></script>
    <script src="https://maayanlab.github.io/clustergrammer/lib/js/jquery-1.11.2.min.js"></script>
    <script src="https://maayanlab.github.io/clustergrammer/lib/js/underscore-min.js"></script>
    <script src="https://maayanlab.github.io/clustergrammer/lib/js/bootstrap.min.js"></script>
    <script src="https://maayanlab.github.io/clustergrammer/clustergrammer.js"></script>
  </head>
  <body>
    <div id="cg-container"></div>
    <script>
      var network_data = {net_json_js};
      var args = {{
        root: '#cg-container',
        network_data: network_data
      }};
      Clustergrammer(args);
    </script>
  </body>
</html>
"""


def make_app(matrix: pd.DataFrame) -> Dash:
    iframe_src = build_clustergrammer_srcdoc(matrix)
    sample_preview = matrix.iloc[:8, : min(6, matrix.shape[1])]

    app = Dash(__name__)
    app.title = "Clustergrammer mock matrix"

    app.layout = html.Div(
        [
            html.H3(f"Clustergrammer mock matrix ({matrix.shape[0]} compounds x {matrix.shape[1]} samples)"),
            html.Iframe(
                title="Clustergrammer heatmap",
                srcDoc=iframe_src,
                style={"border": "1px solid #ccc", "width": "100%", "height": "820px"},
            ),
            html.H4("Data preview (first 8 rows x up to 6 cols)"),
            html.Pre(sample_preview.to_csv()),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal Clustergrammer Dash app.")
    parser.add_argument("--compounds", type=int, default=80, help="Number of compounds (rows) to generate.")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples (columns) to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--port", type=int, default=8050, help="Port for the Dash server.")
    args = parser.parse_args()

    matrix = load_or_create_mock_data(args.compounds, args.samples, args.seed)
    app = make_app(matrix)
    app.run(host="0.0.0.0", port=args.port, debug=True)


if __name__ == "__main__":
    main()
