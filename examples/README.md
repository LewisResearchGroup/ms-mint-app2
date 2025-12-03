# Clustergrammer Dash example

Minimal Dash app that embeds the stock Clustergrammer UI to visualize a mock matrix (compounds as rows, samples as columns). No UI overrides—just the official Clustergrammer assets.

## Files
- `clustergrammer_dash_app.py`: Dash server that renders the clustergrammer heatmap.
- `mock_compound_sample_matrix.csv`: Mock data (80 compounds x 100 samples) regenerated if shapes differ.

## Run
```bash
pip install dash pandas numpy clustergrammer
python examples/clustergrammer_dash_app.py [--compounds 400 --samples 1000 --seed 7 --port 8050]
```

Then open http://localhost:8050.

## Using your own data
Replace `mock_compound_sample_matrix.csv` with your matrix (rows=compounds, cols=samples) and restart the app.
