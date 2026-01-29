"""Bar Chart tab for Analysis plugin."""

from ._shared import (
    fac, html, dcc, go, pd, np, logger,
    Input, Output, State, ALL, MATCH, PreventUpdate,
    duckdb_connection, GROUP_LABELS, METRIC_OPTIONS, PLOTLY_HIGH_RES_CONFIG,
    _calc_y_range_numpy, dash, get_download_config
)
from scipy.stats import ttest_ind, f_oneway
from .pca import run_pca_samples_in_cols


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
                        style={'width': 'calc(55% - 6px)', 'height': '450px', 'overflowY': 'auto'},
                    ),
                    # Chromatogram container (right side)
                    html.Div(
                        [
                            fac.AntdSpin(
                                dcc.Graph(
                                    id='bar-chromatogram',
                                    config={'displayModeBar': True, 'responsive': True},
                                    style={'height': '450px', 'width': '100%'},
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
        id='analysis-bar-content',
    ), bar_selected_sample_store


# Store to track the currently selected sample for bar highlighting (defined outside for callback access)
bar_selected_sample_store = dcc.Store(id='bar-selected-sample', data=None)


def generate_bar_plots(bar_matrix, group_series, color_map, group_label, metric, norm_value, bar_comp_checks, compound_options, filename='bar_plot'):
    """Generate the Bar plots."""
    config = get_download_config(filename=filename, image_format='svg')
    default_bar = None
    bar_options = compound_options
    loadings_for_sort = None
    
    # Similar Logic for default selection as Violin
    if bar_options:
        try:
            pca_results = run_pca_samples_in_cols(
                bar_matrix,
                n_components=min(bar_matrix.shape[0], bar_matrix.shape[1], 5)
            )
            loadings = pca_results.get('loadings')
            if loadings is not None and 'PC1' in loadings.columns:
                loadings_for_sort = loadings
                default_bar = loadings['PC1'].abs().idxmax()
        except Exception:
                if bar_options:
                    default_bar = bar_options[0]['value']
    
    if not default_bar and bar_options:
            default_bar = bar_options[0]['value']

    if loadings_for_sort is not None and 'PC1' in loadings_for_sort.columns:
        pc1_sorted = loadings_for_sort['PC1'].abs().sort_values(ascending=False)
        option_map = {opt['value']: opt for opt in compound_options}
        bar_options = [option_map[val] for val in pc1_sorted.index if val in option_map]
    
    from dash import callback_context
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    user_selected = triggered == 'bar-comp-checks'
    
    selected_compound = None
    if user_selected and bar_comp_checks:
            selected_compound = bar_comp_checks
    else:
            if bar_comp_checks and bar_comp_checks in bar_matrix.columns:
                selected_compound = bar_comp_checks
            else:
                selected_compound = default_bar
    
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
                range=[-0.5, len(unique_groups) - 0.5] # Ensure nicely centered view
            ),
            margin=dict(l=60, r=20, t=80, b=80),
            height=450,
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
            clickmode='event'
        )
        graphs.append(dcc.Graph(
            id={'type': 'bar-plot', 'index': 'main'},
            figure=fig, 
            style={'height': '450px', 'width': '100%'},
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
            'bar-comp-checks', 
            'analysis-grouping-select', 
            'analysis-metric-select', 
            'analysis-normalization-select'
        ]

        ms_file_label = None
        
        # If triggered by a click, extract the sample from clickData
        if 'bar-plot' in ctx.triggered[0]['prop_id']:
            try:
                ms_file_label = clickData['points'][0]['customdata'][0]
            except (KeyError, IndexError, TypeError):
                pass
        
        # If triggered by log scale toggle, keep current selection
        elif trigger_id == 'bar-log-scale-switch' and current_selection:
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
            #     text="Select a sample from the bar plot",
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
