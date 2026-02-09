"""Bar Chart tab for Analysis plugin."""

from ._shared import (
    fac, html, dcc, go, px, pd, np, logger,
    Input, Output, State, ALL, MATCH, PreventUpdate,
    METRIC_OPTIONS, PLOTLY_HIGH_RES_CONFIG,
    dash, get_download_config
)
from scipy.stats import ttest_ind, f_oneway
from .pca import run_pca_samples_in_cols
from .chromatogram_shared import (
    resolve_compound_options_and_selection,
    update_chromatogram_from_click,
)


def create_layout():
    """Return the Bar Chart tab layout component."""
    return html.Div(
        [
            fac.AntdFlex(
                [
                    fac.AntdText('Target to display', strong=True),
                    fac.AntdSelect(
                        id='bar-comp-checks',
                        options=[],
                        value=None,
                        allowClear=False,
                        optionFilterProp='label',
                        optionFilterMode='case-insensitive',
                        style={'width': '320px'},
                    ),
                ],
                align='center',
                gap='small',
                wrap=True,
                style={'paddingBottom': '0.75rem'},
            ),

            # Main content: bar plot on left, chromatogram on right
            fac.AntdFlex(
                [
                    # Bar plot container (left side)
                    html.Div(
                        fac.AntdSpin(
                            html.Div(
                                id='bar-graphs',
                                style={
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'gap': '24px',
                                },
                            ),
                            id='bar-spinner',
                            spinning=True,
                            text='Loading Bar Plot...',
                            style={'minHeight': '300px', 'width': '100%'},
                        ),
                        style={'width': 'calc(60% - 6px)', 'height': '450px', 'overflowY': 'auto'},
                    ),
                    # Chromatogram container (right side)
                    html.Div(
                        [
                            fac.AntdSpin(
                                dcc.Graph(
                                    id='bar-chromatogram',
                                    config={'displayModeBar': True, 'responsive': True},
                                    style={'height': '410px', 'width': '100%'},
                                ),
                                text='Loading Chromatogram...',
                            ),
                            fac.AntdFlex(
                                [
                                    fac.AntdText("Log2 Scale", style={'marginRight': '8px', 'fontSize': '12px'}),
                                    fac.AntdSwitch(
                                        id='bar-log-scale-switch',
                                        checked=False,
                                        checkedChildren='On',
                                        unCheckedChildren='Off',
                                        size='small',
                                    ),
                                ],
                                justify='end',
                                align='center',
                                style={'marginTop': '4px', 'width': '100%'}
                            )
                        ],
                        id='bar-chromatogram-container',
                        style={'display': 'block', 'width': 'calc(38% - 6px)', 'height': 'auto'} # Allow height to grow
                    ),
                ],
                gap='middle',
                wrap=False,
                justify='center',
                align='center',
                style={'width': '100%', 'height': 'calc(100vh - 200px)'},
            ),
        ],
        id='analysis-bar-content',
    ), bar_selected_sample_store


# Store to track the currently selected sample for bar highlighting (defined outside for callback access)
bar_selected_sample_store = dcc.Store(id='bar-selected-sample', data=None)


def generate_bar_plots(bar_matrix, group_series, color_map, group_label, metric, norm_value, bar_comp_checks, compound_options, filename='bar_plot'):
    """Generate the Bar plots."""
    config = get_download_config(filename=filename, image_format='svg')
    from dash import callback_context
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    bar_options, selected_compound = resolve_compound_options_and_selection(
        matrix=bar_matrix,
        compound_options=compound_options,
        current_value=bar_comp_checks,
        trigger_id=triggered,
        selector_trigger_id='bar-comp-checks',
        run_pca_fn=run_pca_samples_in_cols,
        allow_legacy_list=False,
    )
    
    graphs = []
    if selected_compound and selected_compound in bar_matrix.columns:
        selected = selected_compound
        melt_df = bar_matrix[[selected]].join(group_series).reset_index().rename(columns={
            'ms_file_label': 'Sample',
            group_series.name: group_label,
            selected: 'Intensity',
        })
        
        metric_label = next((opt['label'] for opt in METRIC_OPTIONS if opt['value'] == metric), metric)
        melt_df['PlotValue'] = melt_df['Intensity']
        y_label = metric_label

        # Calculate stats for Bar plot
        stats_df = melt_df.groupby(group_label)['PlotValue'].agg(['mean', 'sem', 'count']).reset_index()
        
        # Use numeric indexing for X-axis to allow custom jitter and width control
        unique_groups = stats_df[group_label].tolist()
        group_map = {g: i for i, g in enumerate(unique_groups)}
        
        # Dynamic bar width: narrower if few groups, standard otherwise
        n_groups = len(unique_groups)
        bar_width = 0.5 if n_groups <= 3 else None
        
        # Calculate jittered X for scatter
        x_jittered = []
        jitter_amount = 0.15 # Max offset from center
        for g in melt_df[group_label]:
            idx = group_map.get(g)
            if idx is not None:
                    # Deterministic pseudo-random jitter based on value for stability or just random?
                    # Random is standard for jitter
                    noise = np.random.uniform(-jitter_amount, jitter_amount)
                    x_jittered.append(idx + noise)
            else:
                    x_jittered.append(None)

        fig = go.Figure()

        # Add Bars with Error Bars
        fig.add_trace(go.Bar(
            x=[group_map[g] for g in stats_df[group_label]],
            y=stats_df['mean'],
            error_y=dict(type='data', array=stats_df['sem'], visible=True),
            marker_color=[color_map.get(g, '#333') for g in stats_df[group_label]] if color_map else None,
            name='Mean ± SE',
            opacity=0.7,
            showlegend=False,
            width=bar_width
        ))

        # Add individual points on top with jitter
        fig.add_trace(go.Scatter(
            x=x_jittered,
            y=melt_df['PlotValue'],
            mode='markers',
            marker=dict(
                color='rgba(50, 50, 50, 0.7)',
                line=dict(width=1, color='white'),
                size=8
            ),
            customdata=melt_df[['Sample']].values,
            hovertemplate="<b>%{customdata[0]}</b><br>Val: %{y:.2e}<extra></extra>",
            name='Samples',
            showlegend=False
        ))

        # Legend entries per group (single bar trace doesn't create categorical legend)
        if color_map:
            for g in unique_groups:
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(color=color_map.get(g, '#333'), size=8),
                    name=str(g),
                    showlegend=True,
                ))

        # Significance tests (reuse logic)
        groups = [
            g['PlotValue'].dropna().to_numpy()
            for _, g in melt_df.groupby(group_label)
        ]
        groups = [g for g in groups if len(g) >= 2]
        method = None
        p_val = None
        if len(groups) == 2:
            method = "t-test"
            _, p_val = ttest_ind(groups[0], groups[1], equal_var=False, nan_policy='omit')
        elif len(groups) > 2:
            method = "ANOVA"
            _, p_val = f_oneway(*groups)
        
        title_text = f"{selected}"
        if method and p_val is not None and np.isfinite(p_val):
            display_p = f"{p_val:.3e}"
            title_text += f" <span style='font-size: 14px; font-weight: normal; color: #555;'>({method}, p={display_p})</span>"

        fig.update_layout(
            title=dict(text=title_text),
            title_font=dict(size=16),
            yaxis_title=y_label,
            xaxis_title=group_label,
            yaxis=dict(rangemode='tozero' if norm_value in ['none', 'durbin'] else 'normal', fixedrange=False),
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(unique_groups))),
                ticktext=unique_groups,
                range=[-0.5, len(unique_groups) - 0.5], # Ensure nicely centered view
                showticklabels=False,
            ),
            margin=dict(l=140, r=20, t=80, b=80),
            height=410,
            legend=dict(
                title=dict(text=f"{group_label}: ", font=dict(size=13)),
                font=dict(size=12),
                orientation='v',
                yanchor='top',
                y=1.0,
                xanchor='right',
                x=-0.1,
            ),
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
            clickmode='event',
            showlegend=True,
        )
        graphs.append(dcc.Graph(
            id={'type': 'bar-plot', 'index': 'main'},
            figure=fig, 
            style={'height': '410px', 'width': '100%'},
            config=config
        ))
    return graphs, bar_options, selected_compound

def register_callbacks(app):
    """Register Bar Chart callbacks."""

    @app.callback(
        Output('bar-chromatogram', 'figure'),
        Output('bar-chromatogram-container', 'style'),
        Output('bar-selected-sample', 'data'),
        Input({'type': 'bar-plot', 'index': ALL}, 'clickData'),
        Input('bar-comp-checks', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('bar-log-scale-switch', 'checked'),
        State('bar-selected-sample', 'data'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def update_bar_chromatogram_on_click(clickData_list, peak_label, group_by_col, metric, normalization, log_scale, current_selection, wdir):
        return update_chromatogram_from_click(
            clickData_list=clickData_list,
            peak_label=peak_label,
            group_by_col=group_by_col,
            log_scale=log_scale,
            current_selection=current_selection,
            wdir=wdir,
            click_trigger_key='bar-plot',
            log_switch_trigger_id='bar-log-scale-switch',
        )
