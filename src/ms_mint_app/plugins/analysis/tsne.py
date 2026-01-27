"""t-SNE tab for Analysis plugin."""

from ._shared import (
    fac, html, dcc, go, px, pd, np, logger, os,
    TSNE_COMPONENT_OPTIONS, PLOTLY_HIGH_RES_CONFIG,
    get_physical_cores, create_invisible_figure, dash
)
from sklearn.manifold import TSNE


def create_layout():
    """Return the t-SNE tab layout component."""
    return html.Div(
        [
            # t-SNE-specific controls
            fac.AntdFlex(
                [
                    fac.AntdSpace(
                        [
                            html.Span("X axis:", style={'fontWeight': 500}),
                            fac.AntdSelect(
                                id='tsne-x-comp',
                                options=TSNE_COMPONENT_OPTIONS,
                                value='t-SNE-1',
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
                                id='tsne-y-comp',
                                options=TSNE_COMPONENT_OPTIONS,
                                value='t-SNE-2',
                                allowClear=False,
                                style={'width': 100},
                            ),
                        ],
                        align='center',
                        size='small',
                    ),
                    fac.AntdDivider(direction='vertical', style={'height': '24px', 'margin': '0 12px'}),
                    fac.AntdSpace(
                        [
                            fac.AntdText("Perplexity:", style={'fontWeight': 500}),
                            fac.AntdSlider(
                                id='tsne-perplexity-slider',
                                min=1,
                                max=100,
                                step=1,
                                value=30,
                                style={'width': '200px'},
                                tooltipPrefix='Perplexity: '
                            ),
                        ],
                        align='center',
                        size='small',
                    ),
                    fac.AntdButton(
                        "Generate",
                        id="tsne-regenerate-btn",
                        icon=fac.AntdIcon(icon="antd-sync"),
                        style={'textTransform': 'uppercase'},
                    ),
                ],
                gap='middle',
                align='center',
                style={'marginBottom': '12px'},
            ),
            fac.AntdSpin(
                dcc.Graph(
                    id='tsne-graph',
                    config=PLOTLY_HIGH_RES_CONFIG,
                    style={'height': 'calc(100vh - 220px)', 'width': '80%', 'minHeight': '400px', 'margin': '0 auto'},
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
                text='Loading t-SNE...',
                style={'minHeight': '20vh', 'width': '100%'},
            ),
        ],
        style={'height': '100%'},
    )


def generate_tsne_figure(ndf, color_labels, color_map, group_label, x_comp, y_comp, perplexity):
    """Generate the t-SNE figure."""
    logger.info("Generating t-SNE...")
    
    # CPU limit logic
    n_jobs = max(1, min((os.cpu_count() or 4) // 2, get_physical_cores()))

    # t-SNE components (usually 2, sometimes 3)
    n_components = 3
    # Use random_state for reproducibility
    perplexity = perplexity if perplexity else 30
    
    # Ensure perplexity is valid for sample size
    n_samples = ndf.shape[0]
    if n_samples > 0:
            perplexity = min(perplexity, max(1, n_samples - 1))
            
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_jobs=n_jobs, random_state=42, init='pca')
    
    # Handle sparse/empty
    if ndf.empty or ndf.shape[0] < 2:
        return create_invisible_figure()

    # Fit t-SNE
    # For t-SNE, use the normalized data
    data_for_tsne = ndf.to_numpy()
    embedded = tsne.fit_transform(data_for_tsne)
    
    # Create results DataFrame
    tsne_cols = [f"t-SNE-{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(embedded, index=ndf.index, columns=tsne_cols)
    
    # Add metadata for plotting
    scores_df['color_group'] = color_labels
    scores_df['sample_label'] = scores_df.index
    
    x_axis = x_comp or 't-SNE-1'
    y_axis = y_comp or 't-SNE-2'
    
    # Determine actual columns to use (fallback to available if selected not present)
    if x_axis not in scores_df.columns and len(scores_df.columns) > 0:
        x_axis = scores_df.columns[0]
    if y_axis not in scores_df.columns and len(scores_df.columns) > 1:
        y_axis = scores_df.columns[1]
        
    fig = px.scatter(
        scores_df,
        x=x_axis,
        y=y_axis,
        color='color_group',
        symbol='color_group',
        color_discrete_map=color_map if color_map else None,
        hover_data={'sample_label': True},
    )
    
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
    return fig


def register_callbacks(app):
    """Register t-SNE callbacks."""
    pass
