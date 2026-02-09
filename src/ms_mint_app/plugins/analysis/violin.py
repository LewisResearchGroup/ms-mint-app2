"""Violin tab for Analysis plugin."""

from ._shared import (
    fac, html, dcc, go, px, pd, np, logger,
    Input, Output, State, ALL, MATCH, PreventUpdate,
    METRIC_OPTIONS, PLOTLY_HIGH_RES_CONFIG,
    dash, get_download_config
)
from scipy.stats import ttest_ind, f_oneway
from .pca import run_pca_samples_in_cols
from .chromatogram_shared import (
    patch_chromatogram_zoom,
    resolve_compound_options_and_selection,
    update_chromatogram_from_click,
)


def create_layout():
    """Return the Violin tab layout layout component."""
    return html.Div(
        [
            fac.AntdFlex(
                [
                    fac.AntdText('Target to display', strong=True),
                    fac.AntdSelect(
                        id='violin-comp-checks',
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

            # Main content: violin plot on left, chromatogram on right
            fac.AntdFlex(
                [
                    # Violin plot container (left side)
                    html.Div(
                        fac.AntdSpin(
                            html.Div(
                                id='violin-graphs',
                                style={
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'gap': '24px',
                                },
                            ),
                            id='violin-spinner',
                            spinning=True,
                            text='Loading Violin...',
                            style={'minHeight': '300px', 'width': '100%'},
                        ),
                        style={'width': 'calc(60% - 6px)', 'height': '450px', 'overflowY': 'auto'},
                    ),
                    # Chromatogram container (right side)
                    html.Div(
                        [
                            fac.AntdSpin(
                                dcc.Graph(
                                    id='violin-chromatogram',
                                    config={'displayModeBar': True, 'responsive': True},
                                    style={'height': '410px', 'width': '100%'},
                                ),
                                text='Loading Chromatogram...',
                            ),
                            fac.AntdFlex(
                                [
                                    fac.AntdText("Log2 Scale", style={'marginRight': '8px', 'fontSize': '12px'}),
                                    fac.AntdSwitch(
                                        id='violin-log-scale-switch',
                                        checked=False,
                                        checkedChildren='On',
                                        unCheckedChildren='Off',
                                        size='small',
                                    ),
                                ],
                                justify='end',
                                align='center',
                                style={'marginTop': '12px', 'width': '100%'}
                            )
                        ],
                        id='violin-chromatogram-container',
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
        id='analysis-violin-content',
    ), violin_selected_sample_store


# Store to track the currently selected sample for violin highlighting (defined outside for callback access)
violin_selected_sample_store = dcc.Store(id='violin-selected-sample', data=None)


def _build_violin_graphs(violin_matrix, group_series, color_map, group_label, metric, norm_value, selected_compound, filename='violin_plot'):
    config = get_download_config(filename=filename, image_format='svg')
    graphs = []
    if selected_compound and selected_compound in violin_matrix.columns:
        selected = selected_compound
        melt_df = violin_matrix[[selected]].join(group_series).reset_index().rename(columns={
            'ms_file_label': 'Sample',
            group_series.name: group_label,
            selected: 'Intensity',
        })
        
        # Use label from METRIC_OPTIONS instead of hardcoded strings
        metric_label = next((opt['label'] for opt in METRIC_OPTIONS if opt['value'] == metric), metric)

        melt_df['PlotValue'] = melt_df['Intensity']
        y_label = metric_label

        fig = px.violin(
            melt_df,
            x=group_label,
            y='PlotValue',
            color=group_label,
            color_discrete_map=color_map if color_map else None,
            box=False,
            points='all',
            hover_data={
                'Sample': True,
                group_label: False,  # redundant with x-axis
                'PlotValue': False,  # redundant with Intensity
                'Intensity': ':.2e'  # formatted intensity
            },
        )
        # Custom hover template to be very concise and clean
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>Int: %{customdata[1]}<extra></extra>",
            selector=dict(type='violin')
        )
        fig.update_traces(jitter=0.25, meanline_visible=False, pointpos=-0.5, selector=dict(type='violin'))
        # Clamp KDE tails with spanmode='hard', similar to seaborn cut; use 1st-99th percentiles
        low, high = (
            melt_df['PlotValue'].quantile(0.01),
            melt_df['PlotValue'].quantile(0.99),
        )
        fig.update_traces(spanmode='hard', span=[low, high], side='positive', scalemode='width',
                            selector=dict(type='violin'))
        # Simple significance test: t-test for 2 groups, ANOVA for >2
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

            margin=dict(l=0, r=10, t=110, b=80),
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
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            xaxis_tickfont=dict(size=12),
            xaxis_showticklabels=False,
            yaxis_tickfont=dict(size=12),
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
            clickmode='event'
        )
        graphs.append(dcc.Graph(
            id={'type': 'violin-plot', 'index': 'main'},
            figure=fig, 
            style={'height': '410px', 'width': '100%'},
            config=config
        ))
    return graphs


def generate_violin_plots(violin_matrix, group_series, color_map, group_label, metric, norm_value, violin_comp_checks, compound_options, filename='violin_plot'):
    """Generate the Violin/Raincloud plots."""
    logger.info("Generating Violin/Raincloud plots...")
    from dash import callback_context
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    violin_options, selected_compound = resolve_compound_options_and_selection(
        matrix=violin_matrix,
        compound_options=compound_options,
        current_value=violin_comp_checks,
        trigger_id=triggered,
        selector_trigger_id='violin-comp-checks',
        run_pca_fn=run_pca_samples_in_cols,
        allow_legacy_list=True,
    )

    graphs = _build_violin_graphs(
        violin_matrix, group_series, color_map, group_label, metric, norm_value, selected_compound, filename=filename
    )
    return graphs, violin_options, selected_compound


def register_callbacks(app):
    """Register Violin callbacks."""
    
    @app.callback(
        Output('violin-spinner', 'spinning'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('violin-graphs', 'children'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('analysis-grouping-select', 'value'),
        prevent_initial_call=True,
    )
    def toggle_violin_spinner(active_tab, violin_children, metric_value, norm_value, group_value):
        from dash import callback_context

        if active_tab != 'raincloud':
            return False

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        # When user switches to raincloud or changes parameters, force spinner on
        if trigger in ('analysis-sidebar-menu', 'analysis-metric-select', 'analysis-normalization-select', 'analysis-grouping-select'):
            return True

        # Otherwise, stop spinning once content is loaded
        return not violin_children

    @app.callback(
        Output('violin-chromatogram', 'figure'),
        Output('violin-chromatogram-container', 'style'),
        Output('violin-selected-sample', 'data'),
        Input({'type': 'violin-plot', 'index': ALL}, 'clickData'),
        Input('violin-comp-checks', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('violin-log-scale-switch', 'checked'),
        State('violin-selected-sample', 'data'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def update_chromatogram_on_click(clickData_list, peak_label, group_by_col, metric, normalization, log_scale, current_selection, wdir):
        return update_chromatogram_from_click(
            clickData_list=clickData_list,
            peak_label=peak_label,
            group_by_col=group_by_col,
            log_scale=log_scale,
            current_selection=current_selection,
            wdir=wdir,
            click_trigger_key='violin-plot',
            log_switch_trigger_id='violin-log-scale-switch',
        )

    @app.callback(
        Output('violin-chromatogram', 'figure', allow_duplicate=True),
        Input('violin-chromatogram', 'relayoutData'),
        State('violin-chromatogram', 'figure'),
        prevent_initial_call=True
    )
    def update_violin_chromatogram_zoom(relayout, figure_state):
        return patch_chromatogram_zoom(relayout, figure_state)

    @app.callback(
        Output({'type': 'violin-plot', 'index': MATCH}, 'figure'),
        Input({'type': 'violin-plot', 'index': MATCH}, 'clickData'),
        State({'type': 'violin-plot', 'index': MATCH}, 'figure'),
        prevent_initial_call=True
    )
    def highlight_selected_point(clickData, fig_dict):
        """Draw a red circle around the selected sample point."""
        if not clickData:
            raise PreventUpdate
        
        # Parse clickData
        point = clickData['points'][0]
        curve_number = point['curveNumber']
        point_index = point['pointNumber']
        
        # Reconstruct figure
        fig = go.Figure(fig_dict)
        
        # Clear any previous selection styling
        for i, trace in enumerate(fig.data):
            # Reset selectedpoints for all traces
            if hasattr(trace, 'selectedpoints'):
                trace.selectedpoints = None
        
        # Apply selection to the clicked trace/point
        if curve_number < len(fig.data):
            fig.data[curve_number].selectedpoints = [point_index]
            
            # Style selected point with red color and larger size, keep unselected fully visible
            fig.data[curve_number].selected = dict(
                marker=dict(
                    color='red',
                    size=14,
                    opacity=1.0
                )
            )
            fig.data[curve_number].unselected = dict(
                marker=dict(opacity=1.0)  # Keep unselected points fully visible
            )
        
        return fig
