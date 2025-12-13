import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

np.random.seed(0)

n_compounds = 50

df_results = pd.DataFrame({
    "compound_id": [f"C{i:03d}" for i in range(n_compounds)],
    "log2FC": np.random.normal(0, 1, n_compounds),
    "p_value": np.random.uniform(1e-4, 1, n_compounds)
})

df_results["-log10_p"] = -np.log10(df_results["p_value"])

n_samples_per_group = 10

records = []

for compound in df_results["compound_id"]:
    for group in ["Group_A", "Group_B"]:
        for i in range(n_samples_per_group):
            records.append({
                "compound_id": compound,
                "group": group,
                "abundance": np.random.lognormal(mean=2, sigma=0.4)
            })

df_long = pd.DataFrame(records)

fig_volcano = px.scatter(
    df_results,
    x="log2FC",
    y="-log10_p",
    custom_data=["compound_id"],
    title="Volcano plot"
)

fig_volcano.update_traces(
    mode="markers",
    marker=dict(size=8)
)

app = Dash(__name__)

app.layout = html.Div(
    style={"width": "80%", "margin": "auto"},
    children=[
        dcc.Graph(
            id="volcano-plot",
            figure=fig_volcano
        ),
        dcc.Graph(
            id="distribution-plot"
        )
    ]
)

@app.callback(
    Output("distribution-plot", "figure"),
    Input("volcano-plot", "clickData")
)
def update_distribution(clickData):
    if clickData is None:
        return {}

    compound_id = clickData["points"][0]["customdata"][0]

    df_compound = df_long[df_long["compound_id"] == compound_id]

    fig = px.violin(
        df_compound,
        x="group",
        y="abundance",
        box=True,
        points="all",
        title=f"Distribution of {compound_id}"
    )

    return fig

if __name__ == "__main__":
    app.run(debug=True)
