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
from plotly.subplots import make_subplots
from .analysis_tools import pca
from ..duckdb_manager import duckdb_connection, create_pivot
from ..plugin_interface import PluginInterface
import plotly.express as px

_label = "Analysis"
PCA_COMPONENT_OPTIONS = [
    {'label': f'PC{i}', 'value': f'PC{i}'}
    for i in range(1, 6)
]


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


def rocke_durbin(df: pd.DataFrame, c: float) -> pd.DataFrame:
    # df: samples x features (metabolites)
    z = df.to_numpy(dtype=float)
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


clustermap_tab = html.Div(
    fuc.FefferyResizable(
        fac.AntdSpin(
            fac.AntdCenter(
                html.Img(id='bar-graph-matplotlib', style={
                    'width': '100%',
                    'height': '100%',
                    'object-fit': 'cover',
                    'border': '1px solid #dee2e6'
                }),
                style={
                    'height': '100%',
                    'background': '#dee2e6',
                },
            ),
            id='clustermap-spinner',
            spinning=True,
            text='Loading clustermap...',
        ),
        minWidth=100,
        minHeight=100,
    ),
    style={'overflow': 'auto', 'height': 'calc(100vh - 156px)'}
)
pca_tab = html.Div(
    [
        fac.AntdSpace(
            [
                html.Span("X axis:"),
                fac.AntdSelect(
                    id='pca-x-comp',
                    options=PCA_COMPONENT_OPTIONS,
                    value='PC1',
                    allowClear=False,
                    style={'width': 140},
                ),
                html.Span("Y axis:"),
                fac.AntdSelect(
                    id='pca-y-comp',
                    options=PCA_COMPONENT_OPTIONS,
                    value='PC2',
                    allowClear=False,
                    style={'width': 140},
                ),
            ],
            style={'marginBottom': 10},
        ),
        fac.AntdSpin(
            dcc.Graph(id='pca-graph'),
            text='Loading PCA...',
        ),
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
                # fac.AntdDropdown(
                #     id='processing-options',
                #     title='Options',
                #     buttonMode=True,
                #     arrow=True,
                #     menuItems=[
                #         {'title': fac.AntdText('Download', strong=True), 'key': 'processing-download'},
                #         {'isDivider': True},
                #         {'title': fac.AntdText('Delete selected', strong=True, type='warning'),
                #          'key': 'processing-delete-selected'},
                #         {'title': fac.AntdText('Clear table', strong=True, type='danger'),
                #          'key': 'processing-delete-all'},
                #     ],
                #     buttonProps={'style': {'textTransform': 'uppercase'}},
                # ),
            ],
            justify="space-between",
            align="center",
            gap="middle",
        ),
        fac.AntdTabs(
            id='analysis-tabs',
            items=[
                {'key': 'clustermap', 'label': 'Clustermap', 'children': clustermap_tab},
                {'key': 'pca', 'label': 'PCA', 'children': pca_tab},
                {
                    'key': 'raincloud',
                    'label': 'Violin',
                    'children': html.Div(
                        [
                            fac.AntdSelect(
                                id='raincloud-comp',
                                placeholder='Select compound',
                                allowClear=False,
                                style={'width': 260, 'marginBottom': 12},
                            ),
                            dcc.Graph(id='raincloud-graph'),
                        ]
                    ),
                },
            ],
            centered=True,
            defaultActiveKey='clustermap',
            style={'margin': '12px 0 0 0'},
            tabBarLeftExtraContent=fac.AntdSelect(
                id='analysis-metric-select',
                placeholder='Metric',
                options=[
                    {'label': 'Peak Area', 'value': 'peak_area'},
                    {'label': 'Peak Area (Top 3)', 'value': 'peak_area_top3'},
                    {'label': 'Peak Max', 'value': 'peak_max'},
                    {'label': 'Peak Mean', 'value': 'peak_mean'},
                    {'label': 'Peak Median', 'value': 'peak_median'},
                ],
                value='peak_area',
                optionFilterProp='label',
                optionFilterMode='case-insensitive',
                allowClear=False,
                style={'width': 220},
            ),
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
        Output('raincloud-graph', 'figure'),
        Output('raincloud-comp', 'options'),
        Output('raincloud-comp', 'value'),

        Input('section-context', 'data'),
        Input('analysis-tabs', 'activeKey'),
        Input('pca-x-comp', 'value'),
        Input('pca-y-comp', 'value'),
        Input('raincloud-comp', 'value'),
        Input('analysis-metric-select', 'value'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def show_tab_content(section_context, tab_key, x_comp, y_comp, rain_comp, metric_value, wdir):

        if section_context['page'] != 'Analysis':
            raise PreventUpdate

        if not wdir:
            raise PreventUpdate

        with duckdb_connection(wdir) as conn:
            metric = metric_value or 'peak_area'
            df = create_pivot(conn, value=metric)
            df.set_index('ms_file_label', inplace=True)
            ndf_sample_type = df['sample_type']
            order_df = conn.execute(
                "SELECT ms_file_label FROM samples ORDER BY run_order NULLS LAST, ms_file_label"
            ).df()["ms_file_label"].tolist()
            ordered_labels = [lbl for lbl in order_df if lbl in df.index]
            leftover_labels = [lbl for lbl in df.index if lbl not in ordered_labels]
            df = df.loc[ordered_labels + leftover_labels]
            colors_df = conn.execute(
                "SELECT ms_file_label, sample_type, color FROM samples"
            ).df()
            colors_df["color_key"] = colors_df["sample_type"].fillna(colors_df["ms_file_label"])
            color_map = (
                colors_df.dropna(subset=["color"])
                .drop_duplicates(subset="color_key")
                .set_index("color_key")["color"]
                .to_dict()
            )
            raw_df = df.copy()
            df = df.drop(columns=['ms_type', 'sample_type'], axis=1)
            compound_options = [
                {'label': c, 'value': c}
                for c in raw_df.columns
                if c not in ('ms_type', 'sample_type')
            ]

            # Guard against NaN/inf and empty matrices before downstream plots
            for _df in (df, raw_df):
                _df.replace([np.inf, -np.inf], np.nan, inplace=True)
                _df.dropna(axis=0, how='all', inplace=True)
                _df.dropna(axis=1, how='all', inplace=True)
                if _df.isna().any().any():
                    _df.fillna(0, inplace=True)
            if df.empty or raw_df.empty:
                raise PreventUpdate

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            zdf = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

            ndf = rocke_durbin(df, c=10)
            ndf.replace([np.inf, -np.inf], np.nan, inplace=True)
            ndf.dropna(axis=0, how='all', inplace=True)
            ndf.dropna(axis=1, how='all', inplace=True)
            if ndf.isna().any().any():
                ndf.fillna(0, inplace=True)
            if ndf.empty:
                raise PreventUpdate

        if tab_key == 'clustermap':
            import seaborn as sns
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use('Agg')
            sns.set_theme(font_scale=0.25)
            
            fig = sns.clustermap(
                                 zdf.T,
                                 method='ward', metric='euclidean', 
                                 cmap='vlag', center=0, vmin=-3, vmax=3,
                                 standard_scale=None,
                                 col_cluster=False, 
                                 dendrogram_ratio=0.1,
                                 figsize=(8, 8),
                                 cbar_kws={"orientation": "horizontal"},
                                 cbar_pos=(0.01, 0.95, 0.075, 0.01),
                                )
            fig.ax_heatmap.tick_params(which='both', axis='both', length=0)
            fig.ax_cbar.tick_params(which='both', axis='both', width=0.3, length=2, labelsize=4)
            fig.ax_cbar.set_title("Z-score", fontsize=6, pad=4)


            from io import BytesIO
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300)
            fig.savefig('test.png', format="png")
            import base64
            fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
            fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
            return fig_bar_matplotlib, dash.no_update, dash.no_update, compound_options, dash.no_update

        elif tab_key == 'pca':
            results = run_pca_samples_in_cols(ndf, n_components=5)
            color_labels = ndf_sample_type.fillna(
                pd.Series(ndf_sample_type.index, index=ndf_sample_type.index)
            )
            results['scores']['color_group'] = color_labels
            x_axis = x_comp or 'PC1'
            y_axis = y_comp or 'PC2'
            loadings = results['loadings']
            component_id = x_axis
            if component_id in loadings.index:
                top_features = loadings.loc[component_id].abs().sort_values(ascending=False).head(15).index
                loading_bar = go.Bar(
                    x=top_features,
                    y=loadings.loc[component_id, top_features],
                    name=component_id,
                    width=0.6,
                    marker=dict(color='#bbbbbb'),
                    showlegend=False,
                )
            else:
                loading_bar = None

            variance_bar = go.Bar(
                x=results['explained_variance_ratio'].index,
                y=results['explained_variance_ratio'].values,
                width=0.5,
                showlegend=False,
                marker=dict(color='#bbbbbb'),
            )
            variance_line = go.Scatter(
                x=results['cumulative_variance_ratio'].index,
                y=results['cumulative_variance_ratio'].values,
                mode='lines+markers',
                marker=dict(color='#999999'),
                line=dict(color='#999999'),
                showlegend=False,
            )

            base_scatter = px.scatter(
                results['scores'],
                x=x_axis,
                y=y_axis,
                color='color_group',
                color_discrete_map=color_map if color_map else None,
                title=f'PCA ({x_axis} vs {y_axis})'
            )

            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[[{'rowspan': 2}, {}],
                       [None, {}]],
                column_widths=[0.55, 0.45],
                vertical_spacing=0.12,
                horizontal_spacing=0.1,
                subplot_titles=(
                    f"PCA ({x_axis} vs {y_axis})",
                    "PCA Variance",
                    f"Loadings ({component_id})" if component_id in loadings.index else "Loadings",
                ),
            )

            for t in base_scatter.data:
                fig.add_trace(t, row=1, col=1)

            fig.add_trace(variance_bar, row=1, col=2)
            fig.add_trace(variance_line, row=1, col=2)

            # Label PCA scatter axes
            fig.update_xaxes(title_text=x_axis, row=1, col=1)
            fig.update_yaxes(title_text=y_axis, row=1, col=1)

            if loading_bar:
                fig.add_trace(loading_bar, row=2, col=2)

            fig.update_layout(
                height=700,
                margin=dict(l=140, r=30, t=60, b=50),
                legend_title_text="Sample Type",
                legend=dict(
                    x=-0.05,
                    y=1.04,
                    xanchor="right",
                    yanchor="top",
                    orientation="v",
                    title=dict(text="Sample Type<br>", font=dict(size=14)),
                    font=dict(size=12),
                ),
                xaxis_title_font=dict(size=16),
                yaxis_title_font=dict(size=16),
                xaxis_tickfont=dict(size=12),
                yaxis_tickfont=dict(size=12),
            )

            return dash.no_update, fig, dash.no_update, compound_options, dash.no_update

        elif tab_key == 'raincloud':
            # Build options list
            selected = rain_comp or (compound_options[0]['value'] if compound_options else None)
            rain_fig = go.Figure()
            if selected:
                melt_df = raw_df[[selected]].join(ndf_sample_type).reset_index().rename(columns={
                    'ms_file_label': 'Sample',
                    'sample_type': 'Sample Type',
                    selected: 'Intensity',
                })
                # log2 transform with small epsilon to avoid log(0)
                melt_df['Intensity (log2)'] = np.log2(melt_df['Intensity'].clip(lower=1e-9))
                rain_fig = px.violin(
                    melt_df,
                    x='Sample Type',
                    y='Intensity (log2)',
                    color='Sample Type',
                    color_discrete_map=color_map if color_map else None,
                    box=False,
                    points='all',
                    hover_data=['Sample', 'Sample Type', 'Intensity', 'Intensity (log2)'],
                )
                rain_fig.update_traces(jitter=0.25, meanline_visible=False)
                # Clamp KDE tails similar to seaborn cut; use 1st-99th percentiles
                low, high = (
                    melt_df['Intensity (log2)'].quantile(0.01),
                    melt_df['Intensity (log2)'].quantile(0.99),
                )
                rain_fig.update_traces(spanmode='hard', span=[low, high], selector=dict(type='violin'))
                rain_fig.update_layout(
                    title=f"{selected}",
                    title_font=dict(size=16),
                    yaxis_title='log2 (Intensity)',
                    xaxis_title='Sample Type',
                    yaxis=dict(range=[0, None], fixedrange=False),
                    margin=dict(l=60, r=20, t=50, b=60),
                    legend=dict(
                            title=dict(text="Sample Type<br>", font=dict(size=14)),
                            font=dict(size=12),
                        ),
                    xaxis_title_font=dict(size=16),
                    yaxis_title_font=dict(size=16),
                    xaxis_tickfont=dict(size=12),
                    yaxis_tickfont=dict(size=12),
                )

            return dash.no_update, dash.no_update, rain_fig, compound_options, selected

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    @app.callback(
        Output('clustermap-spinner', 'spinning'),
        Input('analysis-tabs', 'activeKey'),
        Input('bar-graph-matplotlib', 'src'),
        prevent_initial_call=True,
    )
    def toggle_clustermap_spinner(active_tab, bar_src):
        from dash import callback_context

        if active_tab != 'clustermap':
            return False

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        # When user switches to clustermap tab, force spinner on even if previous image exists
        if trigger == 'analysis-tabs':
            return True

        # Otherwise, keep spinning until image src is set
        return bar_src is None
