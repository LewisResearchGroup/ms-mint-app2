import dash
import feffery_antd_components as fac
import feffery_utils_components as fuc
import numpy as np
import pandas as pd
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from .analysis_tools import pca
from ..duckdb_manager import duckdb_connection, create_pivot
from ..plugin_interface import PluginInterface
import plotly.express as px

_label = "Analysis"


class AnalysisPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 9
        print(f'Initiated {_label} plugin')

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)

    def outputs(self):
        return _outputs


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    vals = df.to_numpy(dtype=float)
    mean = np.nanmean(vals, axis=1, keepdims=True)
    std = np.nanstd(vals, axis=1, ddof=0, keepdims=True)
    std = np.where(std == 0, np.nan, std)  # if row is constant, return NaN
    return (vals - mean) / std


def rocke_durbin(df: pd.DataFrame, c: float) -> pd.DataFrame:
    # df: samples x features (metabolites)
    z = zscore_rows(df)
    ef = np.log((z + np.sqrt(z ** 2 + c ** 2)) / 2.0)
    return pd.DataFrame(ef, index=df.index, columns=df.columns)


def run_pca_samples_in_cols(df: pd.DataFrame, n_components=None, random_state=0):
    """Run PCA with samples in rows and targets in columns."""

    X = df.to_numpy(dtype=float)  # samples x targets

    pca = PCA(n_components=n_components, random_state=random_state)
    scores = pca.fit_transform(X)  # samples x components

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    pc_labels = [f"PC{i + 1}" for i in range(pca.n_components_)]

    loadings = pd.DataFrame(
        pca.components_,
        columns=df.columns,
        index=pc_labels,
    )
    scores_df = pd.DataFrame(
        scores,
        index=df.index,
        columns=pc_labels,
    )

    return {
        "pca": pca,
        "scores": scores_df,
        "loadings": loadings,
        "explained_variance_ratio": pd.Series(explained, index=pc_labels),
        "cumulative_variance_ratio": pd.Series(cumulative, index=pc_labels),
    }


clustergram_tab = html.Div(
    fuc.FefferyResizable(
        fac.AntdSpin(
            fac.AntdCenter(
                html.Img(id='bar-graph-matplotlib', style={
                    'width': '100%',
                    'height': '100%',
                    'object-fit': 'cover',  # Usa 'cover' o 'contain' según tu necesidad,
                    'border': '1px solid #dee2e6'
                }),
                style={
                    'height': '100%',
                    'background': '#dee2e6',
                },
            ),
            text='Loading clustergram...',
        ),
        minWidth=100,
        minHeight=100,
    ),
    style={'overflow': 'auto', 'height': 'calc(100vh - 156px)'}
)
pca_tab = html.Div(
    [
        fac.AntdSpin(
            fac.AntdRow(
                [
                    fac.AntdCol(
                        dcc.Graph(id='pca-graph'),
                        flex=1
                    ),
                    fac.AntdCol(
                        dcc.Graph(id='pca-variance-graph'),
                        flex=1
                    ),
                ]
            ),
            text='Loading PCA...',
        )
    ]
)

_layout = html.Div(
    [
        fac.AntdFlex(
            [
                fac.AntdFlex(
                    [
                        fac.AntdTitle(
                            'Analysis', level=4, style={'margin': '0'}
                        ),
                        fac.AntdIcon(
                            id='analysis-tour-icon',
                            icon='pi-info',
                            style={"cursor": "pointer", 'paddingLeft': '10px'},
                        ),
                    ],
                    align='center',
                ),
                fac.AntdDropdown(
                    id='processing-options',
                    title='Options',
                    buttonMode=True,
                    arrow=True,
                    menuItems=[
                        {'title': fac.AntdText('Download', strong=True), 'key': 'processing-download'},
                        {'isDivider': True},
                        {'title': fac.AntdText('Delete selected', strong=True, type='warning'),
                         'key': 'processing-delete-selected'},
                        {'title': fac.AntdText('Clear table', strong=True, type='danger'),
                         'key': 'processing-delete-all'},
                    ],
                    buttonProps={'style': {'textTransform': 'uppercase'}},
                ),
            ],
            justify="space-between",
            align="center",
            gap="middle",
        ),
        fac.AntdTabs(
            id='analysis-tabs',
            items=[
                {'key': 'clustergram', 'label': 'Clustergram', 'children': clustergram_tab},
                {'key': 'pca', 'label': 'PCA', 'children': pca_tab},
            ],
            centered=True,
            defaultActiveKey='clustergram',

        )
    ]
)

_outputs = None


def layout():
    return _layout


def callbacks(app, fsc, cache):
    @app.callback(
        Output('bar-graph-matplotlib', 'src'),
        Output('pca-graph', 'figure'),
        Output('pca-variance-graph', 'figure'),

        Input('section-context', 'data'),
        Input('analysis-tabs', 'activeKey'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def show_tab_content(section_context, tab_key, wdir):

        if section_context['page'] != 'Analysis':
            raise PreventUpdate
        with duckdb_connection(wdir) as conn:
            df = create_pivot(conn)
            df.set_index('ms_file_label', inplace=True)
            ndf_sample_type = df['sample_type']
            df = df.drop(columns=['ms_type', 'sample_type'], axis=1)
            ndf = rocke_durbin(df, c=10)
        if tab_key == 'clustergram':
            import seaborn as sns
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use('Agg')
            plt.rcParams["figure.dpi"] = 150
            sns.set_theme(font_scale=0.5)

            fig = sns.clustermap(ndf.T, method='ward', metric='euclidean', cmap='vlag', standard_scale=None,
                                 col_cluster=False, figsize=(10, 10))
            from io import BytesIO
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300)
            fig.savefig('test.png', format="png")
            import base64
            fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
            fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
            return fig_bar_matplotlib, dash.no_update, dash.no_update

        elif tab_key == 'pca':
            results = run_pca_samples_in_cols(ndf, n_components=5)
            results['scores']['color_group'] = ndf_sample_type
            pca_fig = px.scatter(
                results['scores'],
                    x='PC1',
                    y='PC2',
                    color='color_group',
                title='PCA'
            )
            variance_fig = go.Figure()
            variance_fig.add_trace(
                go.Bar(
                    x=results['explained_variance_ratio'].index,
                    y=results['explained_variance_ratio'].values,
                    showlegend=False,
                    marker=dict(
                        color='#bbbbbb',
                    )
                )
            )
            variance_fig.add_trace(
                go.Scatter(
                    x=results['cumulative_variance_ratio'].index,
                    y=results['cumulative_variance_ratio'].values,
                    showlegend=False,
                    marker=dict(
                        color='#bbbbbb',
                    )
                ),
            )
            variance_fig.update_layout({'title': {'text': 'PCA Variance' }})
            return dash.no_update, pca_fig, variance_fig

        return dash.no_update, dash.no_update, dash.no_update