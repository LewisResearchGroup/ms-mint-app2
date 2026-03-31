"""PCA tab for Analysis plugin."""

from ._shared import (
    fac, html, dcc, go, pd, np, logger,
    PCA_COMPONENT_OPTIONS, PLOTLY_HIGH_RES_CONFIG,
    make_subplots, px, dash, Input, Output, State, PreventUpdate
)
from sklearn.decomposition import PCA


def run_pca_samples_in_cols(df: pd.DataFrame, n_components=None, random_state=0):
    """Run PCA with samples in rows and targets in columns."""

    X = df.to_numpy(dtype=float)  # samples x targets

    pca = PCA(n_components=n_components, random_state=random_state, svd_solver="full")
    scores = pca.fit_transform(X)  # samples x components

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    pc_labels = [f"PC{i + 1}" for i in range(pca.n_components_)]

    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_), # https://stats.stackexchange.com/questions/143905/loadings-vs-eigenvectors-in-pca-when-to-use-one-or-another
        columns=pc_labels,
        index=df.columns,
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


def create_layout():
    """Return the PCA tab layout component."""
    return html.Div(
        [
            # PCA-specific controls: X axis and Y axis
            fac.AntdFlex(
                [
                    fac.AntdSpace(
                        [
                            html.Span("X axis:", style={'fontWeight': 500}),
                            fac.AntdSelect(
                                id='pca-x-comp',
                                options=PCA_COMPONENT_OPTIONS,
                                value='PC1',
                                allowClear=False,
                                style={'width': 100},
                            ),
                        ],
                        align='center',
                        size='small',
                    ),
                    fac.AntdSpace(
                        [
                            html.Span("Y axis:", style={'fontWeight': 500}),
                            fac.AntdSelect(
                                id='pca-y-comp',
                                options=PCA_COMPONENT_OPTIONS,
                                value='PC2',
                                allowClear=False,
                                style={'width': 100},
                            ),
                        ],
                        align='center',
                        size='small',
                    ),
                ],
                gap='middle',
                align='center',
                style={'marginBottom': '12px'},
            ),
            fac.AntdSpin(
                dcc.Graph(
                    id='pca-graph',
                    config=PLOTLY_HIGH_RES_CONFIG,
                    style={'height': 'calc(100vh - 220px)', 'width': '100%', 'minHeight': '400px'},
                    # Start with invisible figure to show only spinner during loading
                    figure={
                        'data': [],
                        'layout': {
                            'xaxis': {'visible': False},
                            'yaxis': {'visible': False},
                            'paper_bgcolor': 'white',
                            'plot_bgcolor': 'white',
                            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                            'autosize': True,
                        }
                    },
                ),
                text='Loading PCA...',
                style={'minHeight': '20vh', 'width': '100%'},
            ),
        ],
        style={'height': '100%'},
    )


def _ordered_unique(values):
    return list(dict.fromkeys(str(value) for value in values if pd.notna(value)))


def _resolve_component_pair(results, x_comp, y_comp):
    available_components = list(results['scores'].columns)
    if not available_components:
        return 'PC1', 'PC1'
    x_axis = x_comp if x_comp in available_components else available_components[0]
    if y_comp in available_components:
        y_axis = y_comp
    elif len(available_components) > 1:
        y_axis = available_components[1]
    else:
        y_axis = available_components[0]
    return x_axis, y_axis


def _add_missing_group_legend_entries(fig, all_groups, plotted_groups, color_map, visible_groups):
    visible_source = all_groups if visible_groups is None else visible_groups
    visible_group_set = set(str(group) for group in (visible_source or []))
    for group in all_groups or []:
        group_name = str(group)
        if group_name in plotted_groups:
            continue
        fig.add_trace(
            go.Scatter(
                # Plotly can drop completely empty traces from the legend.
                # Keep a placeholder point and park it in legendonly mode.
                x=[None],
                y=[None],
                mode='markers',
                name=group_name,
                legendgroup=group_name,
                showlegend=True,
                visible=True if group_name in visible_group_set else 'legendonly',
                marker=dict(
                    color=(color_map or {}).get(group_name, '#bbbbbb'),
                    symbol='circle',
                    size=9,
                ),
                hoverinfo='skip',
            ),
            row=1,
            col=1,
        )


def _apply_pca_layout(fig, group_label):
    fig.update_layout(
        autosize=True,
        margin=dict(l=140, r=80, t=60, b=50),
        legend_title_text=group_label,
        legend=dict(
            x=-0.06,
            y=1.05,
            xanchor='right',
            orientation='v',
            title=dict(text=f'{group_label}<br>', font=dict(size=12)),
            font=dict(size=11),
            itemsizing='constant',
            tracegroupgap=7.5,
        ),
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        xaxis_tickfont=dict(size=12),
        yaxis_tickfont=dict(size=12),
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
    )


def generate_pca_placeholder_figure(color_map, group_label, x_comp, y_comp, all_groups=None, visible_groups=None, message=None):
    """Generate a placeholder PCA figure while keeping legend entries interactive."""
    x_axis = x_comp or 'PC1'
    y_axis = y_comp or 'PC2'
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{'rowspan': 2}, {}],
               [None, {}]],
        column_widths=[0.6, 0.4],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        subplot_titles=(
            f"PCA ({x_axis} vs {y_axis})",
            "Cummulative Variance",
            f"|Loadings| ({x_axis})",
        ),
    )
    _add_missing_group_legend_entries(
        fig,
        _ordered_unique(all_groups or []),
        plotted_groups=set(),
        color_map=color_map,
        visible_groups=visible_groups,
    )
    fig.update_xaxes(title_text=x_axis, row=1, col=1)
    fig.update_yaxes(title_text=y_axis, row=1, col=1)
    fig.add_annotation(
        text=message or "Select at least 2 samples to compute PCA.",
        x=0.28,
        y=0.52,
        xref='paper',
        yref='paper',
        showarrow=False,
        font=dict(size=14, color='#666666'),
        align='center',
    )
    _apply_pca_layout(fig, group_label)
    return fig


def generate_pca_figure(
    ndf, color_labels, color_map, group_label, x_comp, y_comp,
    pca_results=None, all_groups=None, visible_groups=None
):
    """Generate the PCA figure."""
    logger.info(f"Generating PCA ({x_comp} vs {y_comp})...")
    # n_components should be <= min(n_samples, n_features)
    if pca_results is None:
        results = run_pca_samples_in_cols(ndf, n_components=min(ndf.shape[0], ndf.shape[1], 5))
    else:
        results = pca_results
    results['scores']['color_group'] = color_labels
    results['scores']['sample_label'] = results['scores'].index
    x_axis, y_axis = _resolve_component_pair(results, x_comp, y_comp)
    component_id = x_axis
    loading_bar = None
    loading_category_order = None
    loadings_df = results.get('loadings')
    has_component = isinstance(loadings_df, pd.DataFrame) and component_id in loadings_df.columns
    if has_component:
        ordered = loadings_df[component_id].abs().sort_values(ascending=False).head(10)
        loading_category_order = list(reversed(ordered.index.tolist()))
        loading_bar = go.Bar(
            x=ordered.loc[loading_category_order],
            y=loading_category_order,
            orientation='h',
            name=component_id,
            marker=dict(color='#bbbbbb'),
            showlegend=False,
            text=loading_category_order,
            textposition='outside',
            cliponaxis=False,
            hovertemplate='<b>%{y}</b><br>|Loading|=%{x:.4f}<extra></extra>',
        )
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
        symbol='color_group',
        category_orders={'color_group': _ordered_unique(all_groups or color_labels)},
        color_discrete_map=color_map if color_map else None,
        hover_data={'sample_label': True},
        title=f'PCA ({x_axis} vs {y_axis})'
    )
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{'rowspan': 2}, {}],
               [None, {}]],
        column_widths=[0.6, 0.4],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        subplot_titles=(
            f"PCA ({x_axis} vs {y_axis})",
            "Cummulative Variance",
            f"|Loadings| ({component_id})" if has_component else "Loadings",
        ),
    )
    for t in base_scatter.data:
        fig.add_trace(t, row=1, col=1)
    plotted_groups = {str(trace.name) for trace in base_scatter.data}
    _add_missing_group_legend_entries(
        fig,
        _ordered_unique(all_groups or color_labels),
        plotted_groups,
        color_map,
        visible_groups,
    )
    fig.add_trace(variance_bar, row=1, col=2)
    fig.add_trace(variance_line, row=1, col=2)
    # Label PCA scatter axes
    fig.update_xaxes(title_text=x_axis, row=1, col=1)
    fig.update_yaxes(title_text=y_axis, row=1, col=1)
    if loading_bar:
        fig.add_trace(loading_bar, row=2, col=2)
        fig.update_yaxes(
            categoryorder='array',
            categoryarray=loading_category_order,
            showticklabels=False,
            row=2,
            col=2,
        )
        # Nudge the subplot title away from the bars for a bit more breathing room
        for ann in fig['layout']['annotations']:
            if isinstance(ann.text, str) and ann.text.startswith("Loadings"):
                ann.y = ann.y + 0.02
    _apply_pca_layout(fig, group_label)
    
    return fig


def _extract_visible_legend_groups(fig_dict):
    if hasattr(fig_dict, 'to_plotly_json'):
        fig_dict = fig_dict.to_plotly_json()
    data = (fig_dict or {}).get('data') or []
    visible_groups = []
    for trace in data:
        if not trace.get('showlegend', True):
            continue
        trace_name = trace.get('name')
        if not trace_name:
            continue
        trace_visible = trace.get('visible', True)
        if trace_visible in (False, 'legendonly'):
            continue
        visible_groups.append(str(trace_name))
    return _ordered_unique(visible_groups)


def register_callbacks(app):
    """Register PCA callbacks."""
    @app.callback(
        Output('analysis-pca-visible-groups', 'data'),
        Input('pca-graph', 'restyleData'),
        State('pca-graph', 'figure'),
        State('analysis-grouping-select', 'value'),
        prevent_initial_call=True,
    )
    def persist_visible_groups(_restyle_data, figure, group_by_value):
        if not figure or not group_by_value:
            raise PreventUpdate
        visible_groups = _extract_visible_legend_groups(figure)
        return {
            'group_by': group_by_value,
            'visible_groups': visible_groups,
        }
