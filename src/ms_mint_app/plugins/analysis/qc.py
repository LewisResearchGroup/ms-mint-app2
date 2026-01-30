"""QC (Quality Control) tab for Analysis plugin."""

from pathlib import Path

from ._shared import (
    fac, html, dcc, go, px, pd, np, logger,
    duckdb_connection, PLOTLY_HIGH_RES_CONFIG,
    Input, Output, State, PreventUpdate, dash,
    GROUP_COLUMNS, GROUP_LABELS, METRIC_OPTIONS, allowed_metrics,
    rocke_durbin, _calc_y_range_numpy
)


def create_layout():
    """Return the QC tab layout layout component."""
    return html.Div(
        [
            fac.AntdFlex(
                [
                    # Left side: Stacked QC Scatter Plots
                    html.Div(
                        html.Div(
                            fac.AntdSpin(
                                fac.AntdFlex(
                                    [
                                        fac.AntdDivider(
                                            children="RT",
                                            lineColor="#ccc",
                                            fontColor="#444",
                                            fontSize="14px",
                                            style={'margin': '12px 0'}
                                        ),

                                        html.Div([
                                            dcc.Graph(
                                                id='qc-rt-graph',
                                                config=PLOTLY_HIGH_RES_CONFIG,
                                                style={'height': '280px', 'width': '100%'},
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
                                            )
                                        ], style={'height': '280px', 'width': '100%', 'display': 'block'}),
                                        fac.AntdDivider(
                                            children="Peak Area",
                                            lineColor="#ccc",
                                            fontColor="#444",
                                            fontSize="14px",
                                            style={'margin': '12px 0'}
                                        ),

                                        html.Div([
                                            dcc.Graph(
                                                id='qc-mz-graph',
                                                config=PLOTLY_HIGH_RES_CONFIG,
                                                style={'height': '280px', 'width': '100%'},
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
                                            )
                                        ], style={'height': '280px', 'width': '100%', 'display': 'block'}),
                                    ],
                                    vertical=True,
                                    style={'width': '100%'}
                                ),
                                id='qc-spinner',
                                spinning=True,
                                text='Loading QC plots...',
                                style={'minHeight': '300px', 'width': '100%'},
                            ),
                            style={'minHeight': '100%', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center'},
                        ),
                        style={'width': 'calc(55% - 6px)', 'height': '100%', 'overflowY': 'auto'},
                    ),
                    
                    # Right side: Chromatogram
                    html.Div(
                        [
                            fac.AntdSpin(
                                dcc.Graph(
                                    id='qc-chromatogram',
                                    config={'displayModeBar': True, 'responsive': True},
                                    style={'height': '100%', 'width': '100%'},
                                ),
                                text='Loading Chromatogram...',
                            ),
                            fac.AntdFlex(
                                [
                                    fac.AntdText("Log2 Scale", style={'marginRight': '8px', 'fontSize': '12px'}),
                                    fac.AntdSwitch(
                                        id='qc-log-scale-switch',
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
                        id='qc-chromatogram-container',
                        style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'justifyContent': 'center',
                            'alignItems': 'center',
                            'width': 'calc(43% - 6px)',
                            'height': '100%'
                        }
                    ),
                ],
                gap='large',
                wrap=False,
                justify='center',
                align='center',
                style={'width': '100%', 'height': 'calc(100vh - 160px)'},
            ),
            
            # QC Controls (Hidden by default, shown in header via portal/callback or standard layout)
            # Actually, the target selector is in the header, defined in main layout.
            # But we might need a store here?
            dcc.Store(id='qc-selected-sample', data=None),
        ],
        id='analysis-qc-content',
    )


def register_callbacks(app):
    """Register all QC-specific callbacks."""
    
    @app.callback(
        Output('qc-target-select', 'options'),
        Output('qc-target-select', 'value'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('wdir', 'data'),
        State('qc-target-select', 'value'),
        prevent_initial_call=False,
    )
    def update_qc_target_options(current_key, wdir, current_value):
        """Populate QC target dropdown when QC tab is selected."""
        if current_key != 'qc' or not wdir:
            raise PreventUpdate
        
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return [], None
            
            # Get targets that have results
            targets = conn.execute("""
                SELECT DISTINCT t.peak_label 
                FROM targets t
                JOIN results r ON t.peak_label = r.peak_label
                ORDER BY t.peak_label COLLATE NOCASE
            """).fetchall()
            
            if not targets:
                return [], None
            
            options = [{'label': t[0], 'value': t[0]} for t in targets]
            value = current_value if current_value in [t[0] for t in targets] else targets[0][0]
            
            return options, value
            
    @app.callback(
        Output('qc-target-wrapper', 'style'),
        Input('analysis-sidebar-menu', 'currentKey'),
    )
    def toggle_header_visibility(current_key):
        """Toggle visibility of the QC target selector in the header."""
        if current_key == 'qc':
            return {'display': 'flex'}
        return {'display': 'none'}

    @app.callback(
        Output('qc-rt-graph', 'figure'),
        Output('qc-mz-graph', 'figure'),
        Output('qc-spinner', 'spinning'),
        Input('qc-target-select', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('wdir', 'data'),
        prevent_initial_call=False,
    )
    def generate_qc_plots(peak_label, group_by, metric_value, norm_value, wdir):
        """Generate QC plots: RT and m/z in separate figures."""
        if not peak_label or not wdir:
            raise PreventUpdate
        
        from plotly.subplots import make_subplots
        
        with duckdb_connection(wdir) as conn:
            if conn is None:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No data available", paper_bgcolor='white', plot_bgcolor='white')
                return empty_fig, empty_fig, False
            
            group_col = group_by if group_by else 'sample_type'

            metric = 'peak_area'
            if metric_value == 'scalir_conc' or metric_value in allowed_metrics:
                metric = metric_value
            norm_value = norm_value or 'none'

            target_table = 'results'
            if metric == 'scalir_conc':
                scalir_path = Path(wdir) / "results" / "scalir" / "concentrations.csv"
                if not scalir_path.exists():
                    empty_fig = go.Figure()
                    empty_fig.update_layout(title="SCALiR concentrations not found", paper_bgcolor='white', plot_bgcolor='white')
                    return empty_fig, empty_fig, False
                try:
                    conn.execute(f"CREATE OR REPLACE TEMP VIEW scalir_temp_conc AS SELECT * FROM read_csv_auto('{scalir_path}')")
                    conn.execute("""
                        CREATE OR REPLACE TEMP VIEW scalir_results_view AS 
                        SELECT 
                            r.ms_file_label, 
                            r.peak_label,
                            r.peak_rt_of_max,
                            r.peak_mz_of_max,
                            r.peak_area,
                            s.pred_conc AS scalir_conc 
                        FROM results r 
                        LEFT JOIN scalir_temp_conc s 
                        ON r.ms_file_label = CAST(s.ms_file AS VARCHAR) AND r.peak_label = s.peak_label
                    """)
                    target_table = 'scalir_results_view'
                except Exception as e:
                    logger.error(f"Error preparing SCALiR data: {e}")
                    empty_fig = go.Figure()
                    empty_fig.update_layout(title="Failed to prepare SCALiR data", paper_bgcolor='white', plot_bgcolor='white')
                    return empty_fig, empty_fig, False
            
            # Check if peak_mz_of_max exists
            has_peak_mz = False
            try:
                res_cols = [c[0] for c in conn.execute("DESCRIBE results").fetchall()]
                if 'peak_mz_of_max' in res_cols:
                    has_peak_mz = True
            except:
                pass

            mz_col_sql = "r.peak_mz_of_max," if has_peak_mz else "NULL as peak_mz_of_max,"

            query = f"""
                SELECT 
                    r.ms_file_label,
                    r.peak_rt_of_max,
                    {mz_col_sql}
                    r.{metric},
                    t.rt_min,
                    t.rt_max,
                    t.mz_mean,
                    t.mz_width,
                    COALESCE(s.{group_col}, 'unset') as group_val,
                    s.color,
                    s.acquisition_datetime,
                    ROW_NUMBER() OVER (ORDER BY s.acquisition_datetime NULLS LAST, s.ms_file_label) as sample_order
                FROM {target_table} r
                JOIN targets t ON r.peak_label = t.peak_label
                JOIN samples s ON r.ms_file_label = s.ms_file_label
                WHERE r.peak_label = ?
                ORDER BY s.acquisition_datetime NULLS LAST, s.ms_file_label
            """
            
            try:
                df = conn.execute(query, [peak_label]).df()
            except Exception as e:
                logger.error(f"QC query error: {e}")
                err_fig = go.Figure()
                err_fig.update_layout(title=f"Error: {e}", paper_bgcolor='white', plot_bgcolor='white')
                return err_fig, err_fig, False
            
            if df.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No results for selected target", paper_bgcolor='white', plot_bgcolor='white')
                return empty_fig, empty_fig, False
            
            # Use colors from database (samples table)
            unique_groups = df['group_val'].unique()
            
            color_discrete_map = {}
            default_colors = px.colors.qualitative.Plotly
            
            for i, group in enumerate(unique_groups):
                # meaningful color from database?
                group_color = df[df['group_val'] == group]['color'].iloc[0]
                
                if group == 'unset':
                     color_discrete_map[group] = '#bbbbbb'
                elif group_color and group_color != '#BBBBBB':
                     color_discrete_map[group] = group_color
                else:
                     color_discrete_map[group] = default_colors[i % len(default_colors)]

            # X-axis configuration
            x_col = 'sample_order'
            x_title = 'Sample Order (by Acquisition Time)'
            
            # Identify valid datetime data
            if 'acquisition_datetime' in df.columns:
                 # Ensure datetime type
                 try:
                     # Attempt to parse. Coerce errors to NaT
                     dt_series = pd.to_datetime(df['acquisition_datetime'], errors='coerce')
                     if dt_series.notna().any():
                         # We have valid dates. Create a formatted string column for display
                         df['acquisition_time_str'] = dt_series.dt.strftime('%Y-%m-%d %H:%M')
                         x_col = 'acquisition_time_str'
                         x_title = 'Acquisition Time'
                 except Exception:
                     pass
            
            # Get bounds
            mz_mean = df['mz_mean'].iloc[0] if pd.notna(df['mz_mean'].iloc[0]) else None
            
            # === RT FIGURE ===
            fig_rt = go.Figure()
            
            # === METRIC FIGURE ===
            fig_mz = go.Figure()

            metric_label = next((opt['label'] for opt in METRIC_OPTIONS if opt['value'] == metric), metric)
            metric_values = df[metric].astype(float)
            if norm_value in ('zscore', 'zscore_durbin'):
                std = metric_values.std(ddof=0)
                metric_values = (metric_values - metric_values.mean()) / std if std and np.isfinite(std) else metric_values * 0
            if norm_value in ('durbin', 'zscore_durbin'):
                metric_values = rocke_durbin(pd.DataFrame({metric: metric_values}), c=10)[metric]
            df[metric] = metric_values
            
            # Add traces for each group
            for group in unique_groups:
                group_df = df[df['group_val'] == group]
                base_color = color_discrete_map.get(group, '#888888')
                
                # Use per-sample color for scatter points ONLY if grouping by sample_type
                if (not group_by or group_by == 'sample_type') and 'color' in group_df.columns and group_df['color'].notna().all():
                    scatter_colors = group_df['color']
                else:
                    scatter_colors = base_color

                # === RT SCATTER ===
                fig_rt.add_trace(
                    go.Scatter(
                        x=group_df[x_col],
                        y=group_df['peak_rt_of_max'],
                        mode='markers',
                        name=str(group),
                        marker=dict(color=scatter_colors, size=6),
                        text=group_df['ms_file_label'],
                        hovertemplate='%{text}<br>RT: %{y:.2f}s<extra></extra>',
                        legendgroup=str(group),
                        showlegend=True,
                    )
                )
                
                # === PEAK AREA SCATTER ===
                fig_mz.add_trace(
                    go.Scatter(
                        x=group_df[x_col],
                        y=group_df[metric],
                        mode='markers',
                        name=str(group),
                        marker=dict(color=scatter_colors, size=6),
                        text=group_df['ms_file_label'],
                        hovertemplate='%{text}<br>Int: %{y:.2e}<extra></extra>',
                        legendgroup=str(group),
                        showlegend=True, 
                    )
                )

            layout_common = dict(
                legend=dict(
                    orientation='v',
                    yanchor='top',
                    y=0.95,
                    xanchor='right',
                    x=-0.15,
                    font=dict(size=10),
                ),
                hovermode='closest',
                margin=dict(l=100, r=10, t=50, b=40),
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=280,
            )

            # Update RT Figure Layout
            fig_rt.update_layout(
                **layout_common,
                yaxis_title='RT (sec)',
                xaxis_title="", # Separation of axes, might prefer empty here
                xaxis=dict(showticklabels=True)
            )

            # Update Peak Area Figure Layout
            fig_mz.update_layout(
                **layout_common,
                yaxis_title=metric_label,
                xaxis_title=x_title,
                xaxis=dict(showticklabels=True)
            )
            
            return fig_rt, fig_mz, False

    @app.callback(
        Output('qc-chromatogram', 'figure', allow_duplicate=True),
        Input('qc-chromatogram', 'relayoutData'),
        State('qc-chromatogram', 'figure'),
        prevent_initial_call=True
    )
    def update_qc_chromatogram_zoom(relayout, figure_state):
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
             # handle autosize button which resets axes
             if 'xaxis.autorange' in relayout and relayout['xaxis.autorange']:
                  # Ideally we'd recalculate full range, but preventing update is safer than erroring
                  # or we can let the full redraw happen via other means.
                  raise PreventUpdate
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
        Output('qc-selected-sample', 'data'),
        Input('qc-rt-graph', 'clickData'),
        Input('qc-mz-graph', 'clickData'),
        Input('qc-target-select', 'value'),
        State('wdir', 'data'),
        prevent_initial_call=False,
    )
    def update_qc_selected_sample_store(rt_click, mz_click, peak_label, wdir):
        """
        Coordinator callback: Updates the selected sample store.
        Triggered by graph clicks or target change.
        """
        from dash import callback_context, no_update
        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
        
        ms_file_label = None
        
        # 1. Handle Graph Clicks
        if triggered_id in ['qc-rt-graph', 'qc-mz-graph']:
            clickData = rt_click if triggered_id == 'qc-rt-graph' else mz_click
            if clickData:
                try:
                    point = clickData['points'][0]
                    if 'text' in point:
                        ms_file_label = point['text']
                    elif 'customdata' in point:
                        ms_file_label = point['customdata'][0] if isinstance(point['customdata'], list) else point['customdata']
                except Exception:
                    pass
            # If a click happened but we failed to parse, or if we successfully parsed, return that label
            # (even if None, though usually it won't be if clickData exists)
            return ms_file_label
            
        # 2. Handle Target Change / Initial Load (Default selection logic)
        # Only run this if we didn't get a click (i.e. triggered by dropdown or initial load)
        if wdir and peak_label:
             with duckdb_connection(wdir) as conn:
                if conn:
                    try:
                        # Pick a sample with high contrast (likely to have a peak)
                        top_sample = conn.execute("""
                            SELECT ms_file_label 
                            FROM chromatograms
                            WHERE peak_label = ?
                            ORDER BY (list_max(intensity) - list_min(intensity)) DESC
                            LIMIT 1
                        """, [peak_label]).fetchone()
                        if top_sample:
                            ms_file_label = top_sample[0]
                    except:
                        pass
        
        return ms_file_label

    @app.callback(
        Output('qc-chromatogram', 'figure'),
        Output('qc-chromatogram-container', 'style'),
        Input('qc-selected-sample', 'data'),
        Input('qc-log-scale-switch', 'checked'),
        State('qc-target-select', 'value'),
        State('analysis-grouping-select', 'value'),
        State('wdir', 'data'),
        State('qc-rt-graph', 'figure'),
        prevent_initial_call=False,
    )
    def update_qc_chromatogram(ms_file_label, log_scale, peak_label, group_by_col, wdir, rt_fig):
        """Update the chromatogram plot based on the selected sample store."""
        
        if not ms_file_label or not wdir or not peak_label:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                xaxis=dict(visible=False), 
                yaxis=dict(visible=False), 
                paper_bgcolor='white', 
                plot_bgcolor='white'
            )
            return empty_fig, {'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center', 'width': 'calc(43% - 6px)', 'height': '100%'}

         # Fetch data
        with duckdb_connection(wdir) as conn:
            if conn is None:
                 return dash.no_update, dash.no_update
            
            # 1. Get RT span info for the target
            rt_info = conn.execute("SELECT rt_min, rt_max FROM targets WHERE peak_label = ?", [peak_label]).fetchone()
            rt_min, rt_max = rt_info if rt_info else (None, None)

            # 2. Identify neighbors if a grouping column is selected
            neighbor_files = []
            group_val = None
            group_label = GROUP_LABELS.get(group_by_col, group_by_col or 'Sample')

            if group_by_col:
                try:
                    group_val_query = f'SELECT "{group_by_col}" FROM samples WHERE ms_file_label = ?'
                    row = conn.execute(group_val_query, [ms_file_label]).fetchone()
                    if row:
                        group_val = row[0]
                        if group_val is None:
                             neighbors_query = f"""
                                SELECT ms_file_label, color 
                                FROM samples 
                                WHERE "{group_by_col}" IS NULL AND ms_file_label != ? 
                                ORDER BY random() LIMIT 10
                            """
                             neighbor_files = conn.execute(neighbors_query, [ms_file_label]).fetchall()
                        else:
                            neighbors_query = f"""
                                SELECT ms_file_label, color 
                                FROM samples 
                                WHERE "{group_by_col}" = ? AND ms_file_label != ? 
                                ORDER BY random() LIMIT 10
                            """
                            neighbor_files = conn.execute(neighbors_query, [group_val, ms_file_label]).fetchall()
                except Exception:
                    pass

            display_val = group_val
            if group_by_col and not group_val:
                 display_val = f"{group_label} (unset)"

            # Resolve color from QC RT graph if available (to match scatter plot colors)
            group_color_override = None
            if rt_fig:
                # Determine the group name as it appears in the scatter plot legend
                # In generate_qc_plots, name=str(group_val) where None becomes 'unset'
                lookup_name = str(group_val) if group_val is not None else 'unset'
                
                for trace in rt_fig.get('data', []):
                    if trace.get('name') == lookup_name:
                         # Found the group trace, grab its color
                         marker_color = trace.get('marker', {}).get('color')
                         if isinstance(marker_color, str):
                             group_color_override = marker_color
                         break

            # 3. Fetch chromatograms
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
                return fig, {'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center', 'width': 'calc(43% - 6px)', 'height': '100%'}

            data_map = {row[0]: row for row in chrom_data}
            
            fig = go.Figure()

            # Plot neighbors (background)
            for n_label, n_color in neighbor_files:
                if n_label in data_map:
                    _, scan_times, intensities, _ = data_map[n_label]
                    if len(intensities) > 0 and min(intensities) == max(intensities): continue
                    
                    if group_color_override:
                        n_color = group_color_override
                    elif group_by_col and group_val is None: 
                        n_color = '#bbbbbb'
                    
                    if log_scale: intensities = np.log2(np.array(intensities) + 1)

                    fig.add_trace(go.Scatter(
                        x=scan_times, y=intensities, mode='lines',
                        name=str(display_val), legendgroup=str(display_val), showlegend=False,
                        line=dict(width=1, color=n_color), opacity=0.4,
                        hovertemplate=f"<b>{n_label}</b><br>Scan Time: %{{x:.2f}}<br>Intensity: %{{y:.2e}}<extra>{display_val}</extra>"
                    ))

            # Plot main sample
            if ms_file_label in data_map:
                _, scan_times, intensities, main_color = data_map[ms_file_label]
                
                # Check for flat line only within RT window of the target
                check_intensities = intensities
                if rt_min is not None and rt_max is not None:
                     # Filter checking logic to just the RT span
                     check_intensities = [i for t, i in zip(scan_times, intensities) if rt_min <= t <= rt_max]
                     # If the RT span is very narrow or empty, check_intensities might be empty.
                     # In that case, we might fallback to checking everything or assume it's flat.
                     # Let's assume if we have no data in the window, it's effectively "no signal".
                
                is_flat = False
                if check_intensities:
                    is_flat = min(check_intensities) == max(check_intensities)
                elif rt_min is not None:
                    # No data points in the window?
                    is_flat = True
                
                if is_flat:
                     fig.add_annotation(text="Selected sample has no valid signal (flat line)", showarrow=False)
                else:
                    if group_color_override:
                        main_color = group_color_override
                    elif group_by_col and not group_val: 
                        main_color = '#bbbbbb'
                    legend_name = str(display_val) if group_by_col else ms_file_label
                    if log_scale: intensities = np.log2(np.array(intensities) + 1)

                    fig.add_trace(go.Scatter(
                        x=scan_times, y=intensities, mode='lines',
                        name=legend_name, legendgroup=legend_name, showlegend=True,
                        hovertemplate=f"<b>{ms_file_label}</b><br>Scan Time: %{{x:.2f}}<br>Intensity: %{{y:.2e}}<extra>{legend_name}</extra>",
                        line=dict(width=2, color=main_color), fill='tozeroy', opacity=1.0
                    ))
            
            # Add RT span
            if rt_min is not None and rt_max is not None:
                fig.add_vrect(x0=rt_min, x1=rt_max, fillcolor="green", opacity=0.1, layer="below", line_width=0)

            # Auto-range logic
            x_range_min, x_range_max = None, None
            if rt_min is not None and rt_max is not None:
                padding = 5
                x_range_min, x_range_max = rt_min - padding, rt_max + padding
                # Clamp
                all_x = []
                for t in fig.data:
                    if hasattr(t, 'x') and t.x: all_x.extend(t.x)
                if all_x:
                    x_range_min = max(x_range_min, min(all_x))
                    x_range_max = min(x_range_max, max(all_x))

            # Y-range calculation
            y_range = None
            if x_range_min is not None and x_range_max is not None:
                traces_data = [{'x': list(t.x), 'y': list(t.y)} for t in fig.data if hasattr(t, 'x') and hasattr(t, 'y')]
                y_range = _calc_y_range_numpy(traces_data, x_range_min, x_range_max, is_log=False)

            title_label = ms_file_label
            if len(title_label) > 50: title_label = title_label[:20] + "..." + title_label[-20:]

            y_title = "Intensity (Log2)" if log_scale else "Intensity"
            fig.update_layout(
                title=dict(text=f"{title_label}", font=dict(size=12)),
                xaxis_title="Scan Time (s)", yaxis_title=y_title,
                xaxis_title_font=dict(size=16), yaxis_title_font=dict(size=16),
                xaxis_tickfont=dict(size=12), yaxis_tickfont=dict(size=12),
                template="plotly_white", margin=dict(l=50, r=20, t=90, b=80),
                height=450, showlegend=True,
                legend=dict(title=dict(text=f"{group_label}: ", font=dict(size=13)), font=dict(size=12), orientation='h', y=-0.3, x=0),
                xaxis=dict(range=[x_range_min, x_range_max] if x_range_min else None, autorange=x_range_min is None),
                yaxis=dict(range=y_range if y_range else None, autorange=y_range is None),
            )
            return fig, {'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center', 'width': 'calc(43% - 6px)', 'height': '100%'}

    @app.callback(
        Output('qc-rt-graph', 'figure', allow_duplicate=True),
        Output('qc-mz-graph', 'figure', allow_duplicate=True),
        Input('qc-selected-sample', 'data'),
        State('qc-rt-graph', 'figure'),
        State('qc-mz-graph', 'figure'),
        prevent_initial_call=True
    )
    def highlight_qc_sample(ms_file_label, fig_rt_dict, fig_mz_dict):
        """Draw a red circle around the selected sample point in both graphs."""
        from dash import callback_context, no_update
        
        if not ms_file_label:
            return no_update, no_update
        
        figs = []
        for fig_data in [fig_rt_dict, fig_mz_dict]:
             if not fig_data:
                  figs.append(no_update)
                  continue

             fig = go.Figure(fig_data)
             
             found = False
             # Clear previous
             for i, trace in enumerate(fig.data):
                  if hasattr(trace, 'selectedpoints'):
                       trace.selectedpoints = None
                  
                  # Search for point
                  if not found and hasattr(trace, 'text') and trace.text:
                      # If text is tuple/list
                      try:
                          # trace.text might be a tuple of strings if there are many points
                          if ms_file_label in trace.text:
                               # Find index
                               # Assuming trace.text is an iterable of strings
                               p_index = list(trace.text).index(ms_file_label)
                               
                               trace.selectedpoints = [p_index]
                               trace.selected = dict(marker=dict(color='red', size=10, opacity=1.0))
                               trace.unselected = dict(marker=dict(opacity=0.6))
                               found = True
                      except:
                          pass

             figs.append(fig)
             
        return figs[0], figs[1]
