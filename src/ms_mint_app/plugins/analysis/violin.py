"""Violin tab for Analysis plugin."""

from ._shared import (
    fac, html, dcc, go, px, pd, np, logger,
    Input, Output, State, ALL, MATCH, PreventUpdate,
    duckdb_connection, GROUP_LABELS, METRIC_OPTIONS, PLOTLY_HIGH_RES_CONFIG,
    _calc_y_range_numpy, dash, get_download_config
)
from scipy.stats import ttest_ind, f_oneway
from .pca import run_pca_samples_in_cols


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
                        style={'width': 'calc(55% - 6px)', 'height': '450px', 'overflowY': 'auto'},
                    ),
                    # Chromatogram container (right side)
                    html.Div(
                        [
                            fac.AntdSpin(
                                dcc.Graph(
                                    id='violin-chromatogram',
                                    config={'displayModeBar': True, 'responsive': True},
                                    style={'height': '450px', 'width': '100%'},
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
                                style={'marginTop': '4px', 'width': '100%'}
                            )
                        ],
                        id='violin-chromatogram-container',
                        style={'display': 'block', 'width': 'calc(43% - 6px)', 'height': 'auto'} # Allow height to grow
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


def generate_violin_plots(violin_matrix, group_series, color_map, group_label, metric, norm_value, violin_comp_checks, compound_options, filename='violin_plot'):
    """Generate the Violin/Raincloud plots."""
    config = get_download_config(filename=filename, image_format='svg')
    logger.info("Generating Violin/Raincloud plots...")
    # Build options list; sort by absolute PC1 loading if available so the most
    # influential metabolites surface first.
    loadings_for_sort = None
    violin_options = compound_options
    # Default selection: highest absolute loading on PC1 (per current metric/norm).
    default_violin = None
    if violin_options:
        try:
            pca_results = run_pca_samples_in_cols(
                violin_matrix,
                n_components=min(violin_matrix.shape[0], violin_matrix.shape[1], 5)
            )
            loadings = pca_results.get('loadings')
            if loadings is not None and 'PC1' in loadings.columns:
                loadings_for_sort = loadings
                default_violin = loadings['PC1'].abs().idxmax()
        except Exception:
            if violin_options:
                default_violin = violin_options[0]['value']
    
    if not default_violin and violin_options:
            default_violin = violin_options[0]['value']

    if loadings_for_sort is not None and 'PC1' in loadings_for_sort.columns:
        pc1_sorted = loadings_for_sort['PC1'].abs().sort_values(ascending=False)
        option_map = {opt['value']: opt for opt in compound_options}
        violin_options = [option_map[val] for val in pc1_sorted.index if val in option_map]
    
    from dash import callback_context
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    user_selected = triggered == 'violin-comp-checks'
    
    selected_compound = None
    if user_selected and violin_comp_checks:
            # Handle potential legacy list value or single value
            if isinstance(violin_comp_checks, list):
                selected_compound = violin_comp_checks[0] if violin_comp_checks else default_violin
            else:
                selected_compound = violin_comp_checks
    else:
            # Use current selection if valid and not triggering a reset, otherwise default
            if violin_comp_checks and not isinstance(violin_comp_checks, list) and violin_comp_checks in violin_matrix.columns:
                selected_compound = violin_comp_checks
            else:
                selected_compound = default_violin
    
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
            height=450,
            legend=dict(
                    title=dict(text=f"{group_label}: ", font=dict(size=13)),
                    font=dict(size=12),
                    orientation='h',
                    yanchor='top',
                    y=-0.3,
                    xanchor='left',
                    x=0,
                ),
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            xaxis_tickfont=dict(size=12),
            yaxis_tickfont=dict(size=12),
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
            clickmode='event'
        )
        graphs.append(dcc.Graph(
            id={'type': 'violin-plot', 'index': 'main'},
            figure=fig, 
            style={'height': '450px', 'width': '100%'},
            config=config
        ))
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
        import random
        from dash import ALL
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update

        # Extract clickData safely
        clickData = clickData_list[0] if clickData_list else None

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Reset triggers - when these change, auto-select a random sample
        reset_triggers = [
            'violin-comp-checks', 
            'analysis-grouping-select', 
            'analysis-metric-select', 
            'analysis-normalization-select'
        ]

        ms_file_label = None
        
        # If triggered by a click, extract the sample from clickData
        if 'violin-plot' in ctx.triggered[0]['prop_id']:
            try:
                ms_file_label = clickData['points'][0]['customdata'][0]
            except (KeyError, IndexError, TypeError):
                pass
        
        # If triggered by log scale toggle, keep current selection
        elif trigger_id == 'violin-log-scale-switch' and current_selection:
            ms_file_label = current_selection

        # If no sample (or reset triggered), auto-select a random sample with valid signal
        if not ms_file_label and wdir and peak_label:
            with duckdb_connection(wdir) as conn:
                if conn:
                    # Get top 5 samples with highest intensity contrast (max - min)
                    # This ensures we pick a sample with actual peaks, not flat lines
                    top_samples = conn.execute("""
                        SELECT ms_file_label 
                        FROM chromatograms
                        WHERE peak_label = ?
                        ORDER BY (list_max(intensity) - list_min(intensity)) DESC
                        LIMIT 5
                    """, [peak_label]).fetchall()
                    
                    if top_samples:
                        import random
                        ms_file_label = random.choice(top_samples)[0]
                    else:
                        # Fallback: if all samples are flat, just pick any
                        random_sample = conn.execute("""
                            SELECT DISTINCT c.ms_file_label 
                            FROM chromatograms c
                            WHERE c.peak_label = ?
                            ORDER BY random()
                            LIMIT 1
                        """, [peak_label]).fetchone()
                        if random_sample:
                            ms_file_label = random_sample[0]

        if not ms_file_label or not wdir or not peak_label:
            # Return empty placeholder figure
            fig = go.Figure()
            # fig.add_annotation(
            #     text="Select a sample from the violin plot",
            #     xref="paper", yref="paper",
            #     x=0.5, y=0.5,
            #     showarrow=False,
            #     font=dict(size=14, color="gray")
            # )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                template="plotly_white",
                margin=dict(l=0, r=0, t=0, b=0),
            )
            return fig, {'display': 'block', 'width': 'calc(43% - 6px)', 'height': '450px'}, None

        # Fetch data
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update, dash.no_update, dash.no_update
            
            # 1. Get RT span info for the target
            rt_info = conn.execute("SELECT rt_min, rt_max FROM targets WHERE peak_label = ?", [peak_label]).fetchone()
            rt_min, rt_max = rt_info if rt_info else (None, None)

            # 2. Identify neighbors if a grouping column is selected
            neighbor_files = []
            group_val = None
            
            # Determine group label
            group_label = GROUP_LABELS.get(group_by_col, group_by_col or 'Sample')

            if group_by_col:
                # Get the group value for the clicked sample
                try:
                    group_val_query = f'SELECT "{group_by_col}" FROM samples WHERE ms_file_label = ?'
                    row = conn.execute(group_val_query, [ms_file_label]).fetchone()
                    
                    if row:
                        group_val = row[0]
                        # Fetch up to 10 random other samples from the same group
                        if group_val is None:
                             neighbors_query = f"""
                                SELECT ms_file_label, color 
                                FROM samples 
                                WHERE "{group_by_col}" IS NULL AND ms_file_label != ? 
                                ORDER BY random() 
                                LIMIT 10
                            """
                             neighbor_files = conn.execute(neighbors_query, [ms_file_label]).fetchall()
                        else:
                            neighbors_query = f"""
                                SELECT ms_file_label, color 
                                FROM samples 
                                WHERE "{group_by_col}" = ? AND ms_file_label != ? 
                                ORDER BY random() 
                                LIMIT 10
                            """
                            neighbor_files = conn.execute(neighbors_query, [group_val, ms_file_label]).fetchall()
                except Exception as e:
                    logger.warning(f"Failed to fetch neighbors: {e}")
                    pass

            # Determine display value for legend
            display_val = group_val
            if group_by_col and not group_val:
                 display_val = f"{group_label} (unset)"

            # 3. Fetch chromatograms
            # We need the clicked sample + neighbors
            files_to_fetch = [ms_file_label] + [n[0] for n in neighbor_files]
            placeholders = ','.join(['?'] * len(files_to_fetch))
            
            chrom_query = f"""
                SELECT c.ms_file_label, c.scan_time, c.intensity, s.color
                FROM chromatograms c
                JOIN samples s ON c.ms_file_label = s.ms_file_label
                WHERE c.peak_label = ? AND c.ms_file_label IN ({placeholders})
            """
            
            chrom_data = conn.execute(chrom_query, [peak_label] + files_to_fetch).fetchall()
            
            if not chrom_data:
                fig = go.Figure()
                fig.add_annotation(text="No chromatogram data found", showarrow=False)
                return fig, {'display': 'block', 'width': 'calc(43% - 6px)', 'height': '450px'}, ms_file_label

            # Organize data
            # chrom_data: [(ms_file_label, scan_time, intensity, color), ...]
            data_map = {row[0]: row for row in chrom_data}
            
            fig = go.Figure()

            # Plot neighbors first (background)
            # Plot neighbors first (background)
            for n_label, n_color in neighbor_files:
                if n_label in data_map:
                    _, scan_times, intensities, _ = data_map[n_label]
                    
                    # Check for valid signal (skip flat lines)
                    if len(intensities) > 0 and min(intensities) == max(intensities):
                        continue

                    # If grouping is active but value is missing, use the "unset" color (gray)
                    if group_by_col and group_val is None:
                        n_color = '#bbbbbb'
                    
                    if log_scale:
                         intensities = np.log2(np.array(intensities) + 1)

                    fig.add_trace(go.Scatter(
                        x=scan_times,
                        y=intensities,
                        mode='lines',
                        name=str(display_val),
                        legendgroup=str(display_val),
                        showlegend=False,
                        line=dict(width=1, color=n_color),
                        opacity=0.4,
                        hovertemplate=f"<b>{n_label}</b><br>Scan Time: %{{x:.2f}}<br>Intensity: %{{y:.2e}}<extra>{display_val}</extra>"
                    ))

            # Plot clicked sample (foreground)
            if ms_file_label in data_map:
                _, scan_times, intensities, main_color = data_map[ms_file_label]
                
                # Check for valid signal
                is_flat = len(intensities) > 0 and min(intensities) == max(intensities)
                
                if is_flat:
                     fig.add_annotation(
                        text="Selected sample has no valid signal (flat line)",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=14, color="gray")
                    )
                else:
                    # If grouping is active but value is missing, use the "unset" color (gray)
                    if group_by_col and not group_val:
                        main_color = '#bbbbbb'

                    legend_name = str(display_val) if group_by_col else ms_file_label
                    
                    if log_scale:
                         intensities = np.log2(np.array(intensities) + 1)

                    fig.add_trace(go.Scatter(
                        x=scan_times,
                        y=intensities,
                        mode='lines',
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=True,
                        hovertemplate=f"<b>{ms_file_label}</b><br>Scan Time: %{{x:.2f}}<br>Intensity: %{{y:.2e}}<extra>{legend_name}</extra>",
                        line=dict(width=2, color=main_color),
                        fill='tozeroy',
                        opacity=1.0
                    ))
            
            # Add RT span
            if rt_min is not None and rt_max is not None:
                fig.add_vrect(
                    x0=rt_min, x1=rt_max,
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0,
                )

            # Set initial X-axis range to show RT span with padding (like optimization modal)
            # Show one span width on each side of the RT span
            x_range_min = None
            x_range_max = None
            if rt_min is not None and rt_max is not None:
                span_width = rt_max - rt_min
                # Use 5 seconds as padding since at this point this is optimized
                padding = 5
                x_range_min = rt_min - padding
                x_range_max = rt_max + padding
                
                # Clamp to actual data range
                all_x = []
                for trace in fig.data:
                    if hasattr(trace, 'x') and trace.x is not None:
                        all_x.extend(trace.x)
                if all_x:
                    data_x_min = min(all_x)
                    data_x_max = max(all_x)
                    x_range_min = max(x_range_min, data_x_min)
                    x_range_max = min(x_range_max, data_x_max)

            # Calculate Y-axis range for the visible X range
            y_range = None
            if x_range_min is not None and x_range_max is not None:
                traces_data = [{'x': list(t.x), 'y': list(t.y)} for t in fig.data if hasattr(t, 'x') and hasattr(t, 'y')]
                y_range = _calc_y_range_numpy(traces_data, x_range_min, x_range_max, is_log=False)

            # Truncate title if too long
            title_label = ms_file_label
            if len(title_label) > 50:
                 title_label = title_label[:20] + "..." + title_label[-20:]

            # Update layout
            y_title = "Intensity (Log2)" if log_scale else "Intensity"
            fig.update_layout(
                title=dict(text=f"{peak_label} | {title_label}", font=dict(size=14)),
                xaxis_title="Scan Time (s)",
                yaxis_title=y_title,
                xaxis_title_font=dict(size=16),
                yaxis_title_font=dict(size=16),
                xaxis_tickfont=dict(size=12),
                yaxis_tickfont=dict(size=12),
                template="plotly_white",
                margin=dict(l=50, r=20, t=110, b=80),
                height=450,
                showlegend=True,
                legend=dict(
                    title=dict(text=f"{group_label}: ", font=dict(size=13)),
                    font=dict(size=12),
                    orientation='h',
                    yanchor='top',
                    y=-0.3,
                    xanchor='left',
                    x=0,
                ),
                xaxis=dict(
                    range=[x_range_min, x_range_max] if x_range_min is not None else None,
                    autorange=x_range_min is None,
                ),
                yaxis=dict(
                    range=y_range if y_range else None,
                    autorange=y_range is None,
                ),
            )
            return fig, {'display': 'block', 'width': 'calc(43% - 6px)', 'height': '450px'}, ms_file_label

    @app.callback(
        Output('violin-chromatogram', 'figure', allow_duplicate=True),
        Input('violin-chromatogram', 'relayoutData'),
        State('violin-chromatogram', 'figure'),
        prevent_initial_call=True
    )
    def update_violin_chromatogram_zoom(relayout, figure_state):
        """
        When user zooms on X-axis, auto-fit Y-axis to visible data range.
        This matches the behavior in the optimization modal chromatogram.
        """
        if not relayout or not figure_state:
            raise PreventUpdate

        # Check for X-axis zoom event
        x_range = (relayout.get('xaxis.range[0]'), relayout.get('xaxis.range[1]'))
        y_range = (relayout.get('yaxis.range[0]'), relayout.get('yaxis.range[1]'))
        
        # If user explicitly set Y range, respect it
        if y_range[0] is not None and y_range[1] is not None:
            raise PreventUpdate
        
        # Only act when X-axis zoom happens
        if x_range[0] is None or x_range[1] is None:
            raise PreventUpdate
        
        # Auto-fit Y-axis to visible data in the X range
        from dash import Patch
        fig_patch = Patch()
        
        traces = figure_state.get('data', [])
        y_calc = _calc_y_range_numpy(traces, x_range[0], x_range[1], is_log=False)
        
        if y_calc:
            fig_patch['layout']['xaxis']['range'] = [x_range[0], x_range[1]]
            fig_patch['layout']['xaxis']['autorange'] = False
            fig_patch['layout']['yaxis']['range'] = y_calc
            fig_patch['layout']['yaxis']['autorange'] = False
            return fig_patch
        
        raise PreventUpdate

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
