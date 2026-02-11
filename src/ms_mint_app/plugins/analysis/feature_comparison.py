"""Feature comparison tab for Analysis plugin."""

from scipy.stats import ttest_ind
from pathlib import Path
from datetime import date
import duckdb
from ...duckdb_manager import duckdb_connection, get_workspace_name_from_wdir

from ._shared import (
    fac, html, dcc, go, pd, np, logger,
    Input, Output, State, PreventUpdate,
    GROUP_LABELS, GROUP_COLUMNS,
    prepare_metric_table, normalize_matrix, PLOTLY_HIGH_RES_CONFIG,
    _calc_y_range_numpy, _build_color_map, dash, ensure_valid_group_field,
    analysis_read_connection,
)


FDR_OPTIONS = [
    {'label': 'None', 'value': 'none'},
    {'label': 'Benjamini-Hochberg', 'value': 'bh'},
    {'label': 'Bonferroni', 'value': 'bonferroni'},
]


def create_layout():
    """Return the Feature Comparison tab layout component."""
    return html.Div(
        [
            fac.AntdFlex(
                [
                    fac.AntdText('Sample 1:', strong=False),
                    fac.AntdSelect(
                        id='comparison-sample1-select',
                        options=[],
                        value=None,
                        allowClear=False,
                        optionFilterProp='label',
                        optionFilterMode='case-insensitive',
                        style={'width': '200px'},
                    ),
                    fac.AntdText('Sample 2:', strong=False),
                    fac.AntdSelect(
                        id='comparison-sample2-select',
                        options=[],
                        value=None,
                        allowClear=False,
                        optionFilterProp='label',
                        optionFilterMode='case-insensitive',
                        style={'width': '200px'},
                    ),
                    fac.AntdText('P-value:', strong=False),
                    fac.AntdInputNumber(
                        id='comparison-pvalue-threshold',
                        value=0.05,
                        min=0,
                        max=1,
                        step=0.001,
                        style={'width': '120px'},
                    ),
                    fac.AntdText('FDR:', strong=False),
                    fac.AntdSelect(
                        id='comparison-fdr-method',
                        options=FDR_OPTIONS,
                        value='bh',
                        allowClear=False,
                        style={'width': '200px'},
                    ),
                    fac.AntdText('Plot Type:', strong=False),
                    fac.AntdSwitch(
                        id='comparison-plot-type',
                        checked=False,
                        checkedChildren='Volcano',
                        unCheckedChildren='Scatter',
                    ),
                ],
                align='center',
                gap='middle',
                wrap=True,
                style={'paddingBottom': '0.75rem'},
            ),
            fac.AntdFlex(
                [
                    # Left panel: Comparison plot (more square - narrower width)
                    html.Div(
                        fac.AntdSpin(
                            dcc.Graph(
                                id={'type': 'comparison-plot', 'index': 'main'},
                                config=PLOTLY_HIGH_RES_CONFIG,
                                style={'height': 'calc(100vh - 220px)', 'width': '100%', 'minHeight': '400px'},
                            ),
                            id='comparison-spinner',
                            spinning=False,
                            text='Loading Comparison Plot...',
                            style={'width': '100%'},
                        ),
                        style={'width': '40%'},
                    ),
                    # Right panel: Chromatogram (wider)
                    html.Div(
                        [
                            # Chromatogram Graph
                            fac.AntdSpin(
                                dcc.Graph(
                                    id='comparison-chromatogram',
                                    config=PLOTLY_HIGH_RES_CONFIG,
                                    style={'height': 'calc(100vh - 400px)', 'width': '100%', 'minHeight': '250px'},
                                ),
                                text='Loading Chromatogram...',
                            ),
                            
                            # Controls Row (Bottom)
                            fac.AntdFlex(
                                [
                                    # Left: Selection Controls
                                    fac.AntdFlex(
                                        [
                                            fac.AntdButton(
                                                'Add Selected Compound',
                                                id='comparison-add-btn',
                                                icon=fac.AntdIcon(icon='antd-plus'),
                                                title='Add current feature to selection list',
                                                style={'marginTop': '30px'}
                                            ),
                                            fac.AntdButton(
                                                'Clear Selection',
                                                id='comparison-clear-btn',
                                                icon=fac.AntdIcon(icon='antd-clear'),
                                                type='default',
                                                title='Clear selection list',
                                                style={'marginTop': '30px'}
                                            ),
                                            fac.AntdButton(
                                                'Download',
                                                id='comparison-download-btn',
                                                icon=fac.AntdIcon(icon='antd-download'),
                                                title='Download selected features',
                                                style={'marginTop': '30px'}
                                            ),
                                            fac.AntdText(
                                                id='comparison-selection-count',
                                                children='0 selected',
                                                style={'fontWeight': 'bold', 'marginLeft': '30px', 'marginTop': '30px'},
                                            ),
                                        ],
                                        gap='small',
                                        align='center',
                                    ),
                                    
                                    # Right: Log Scale & Count
                                    fac.AntdFlex(
                                        [
                                            fac.AntdText("Log2 Scale", style={'marginRight': '8px', 'fontSize': '12px'}),
                                            fac.AntdSwitch(
                                                id='comparison-log-scale-switch',
                                                checked=False,
                                                checkedChildren='On',
                                                unCheckedChildren='Off',
                                                size='small',
                                            ),
                                        ],
                                        gap='small',
                                        align='center',
                                    ),
                                ],
                                justify='space-between',
                                align='center',
                                style={'marginTop': '10px', 'width': '100%'},
                            ),

                            # Selection List
                            html.Div(
                                id='comparison-selection-list-container',
                                style={'marginTop': '10px', 'maxHeight': '100px', 'overflowY': 'auto', 'width': '100%'}
                            ),
                        ],
                        id='comparison-chromatogram-container',
                        style={'width': '50%'},
                    ),
                ],
                gap='50px',
                wrap=False,
                justify='center',
                align='center',
                style={'width': '100%'},
            ),
            dcc.Store(id='comparison-active-peak-store'),
            dcc.Store(id='comparison-selection-store', data=[]),
            dcc.Download(id='comparison-download-selection'),
        ],
        id='analysis-comparison-content',
    )




def _adjust_pvalues(pvals, method):
    if method == 'bonferroni':
        return np.minimum(pvals * len(pvals), 1.0)
    if method != 'bh':
        return pvals

    pvals = np.asarray(pvals, dtype=float)
    adjusted = np.full_like(pvals, np.nan, dtype=float)
    finite_mask = np.isfinite(pvals)
    if not finite_mask.any():
        return adjusted

    vals = pvals[finite_mask]
    order = np.argsort(vals)
    ranked = vals[order]
    n_tests = len(ranked)
    bh_vals = ranked * n_tests / (np.arange(1, n_tests + 1))
    bh_vals = np.minimum.accumulate(bh_vals[::-1])[::-1]
    bh_vals = np.minimum(bh_vals, 1.0)

    restored = np.empty_like(vals)
    restored[order] = bh_vals
    adjusted[finite_mask] = restored
    return adjusted


def _empty_plot(message, height=410):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True,
    )
    return fig


def register_callbacks(app):
    """Register Feature Comparison callbacks."""

    # Clientside callback to trigger resize when the comparison tab is shown
    # This fixes Plotly sizing issues with calc() viewport heights
    app.clientside_callback(
        """
        function(currentKey) {
            if (currentKey === 'comparison') {
                // Small delay to ensure DOM is ready
                setTimeout(function() {
                    window.dispatchEvent(new Event('resize'));
                }, 150);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('analysis-comparison-content', 'data-resize-trigger'),
        Input('analysis-sidebar-menu', 'currentKey'),
    )

    @app.callback(
        Output('comparison-sample1-select', 'options'),
        Output('comparison-sample2-select', 'options'),
        Output('comparison-sample1-select', 'value'),
        Output('comparison-sample2-select', 'value'),
        Output('comparison-sample1-select', 'disabled'),
        Output('comparison-sample2-select', 'disabled'),
        Output('analysis-notifications-container', 'children', allow_duplicate=True),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('analysis-grouping-select', 'value'),
        Input('wdir', 'data'),
        State('comparison-sample1-select', 'value'),
        State('comparison-sample2-select', 'value'),
        prevent_initial_call='initial_duplicate',
    )
    def update_sample_options(active_tab, group_by_col, wdir, current_1, current_2):
        if active_tab != 'comparison':
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        if not wdir or not group_by_col:
            return [], [], None, None, True, True, dash.no_update
        group_by_col = ensure_valid_group_field(group_by_col, allow_none=False)

        # Read-only optimization
        with analysis_read_connection(wdir, conn_factory=duckdb_connection) as conn:
            if conn is None:
                return [], [], None, None, True, True, dash.no_update

            try:
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT "{group_by_col}"
                    FROM samples
                    WHERE use_for_analysis = TRUE
                    ORDER BY "{group_by_col}"
                    """
                ).fetchall()
            except duckdb.Error as exc:
                logger.warning(
                    "Failed to load comparison group options for workspace %s using '%s': %s",
                    wdir,
                    group_by_col,
                    exc,
                )
                return [], [], None, None, True, True, dash.no_update

        group_label = GROUP_LABELS.get(group_by_col, group_by_col)
        missing_label = f"{group_label} (unset)"
        values = []
        has_null = False
        for row in rows:
            if row and row[0] is not None and row[0] != '':
                values.append(row[0])
            else:
                has_null = True

        if has_null:
            values.append(missing_label)

        options = [{'label': str(v), 'value': v} for v in values]

        has_multiple = len(values) > 1
        if not has_multiple:
            notice = fac.AntdNotification(
                message="Not enough groups to compare",
                description="Select or define at least two distinct groups in the current 'Group by' field.",
                type="info",
                duration=4,
                placement='bottom',
                showProgress=True,
                stack=True,
            )
            return options, options, None, None, True, True, notice

        default_1 = current_1 if current_1 in values else values[0]
        default_2 = current_2 if current_2 in values and current_2 != default_1 else values[1]
        return options, options, default_1, default_2, False, False, dash.no_update

    @app.callback(
        Output('comparison-spinner', 'spinning'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input({'type': 'comparison-plot', 'index': 'main'}, 'figure'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('comparison-sample1-select', 'value'),
        Input('comparison-sample2-select', 'value'),
        Input('comparison-pvalue-threshold', 'value'),
        Input('comparison-fdr-method', 'value'),
        Input('comparison-plot-type', 'checked'),
        prevent_initial_call=True,
    )
    def toggle_comparison_spinner(active_tab, figure, metric_value, norm_value, group_value, sample1, sample2, pval, fdr, plot_toggle):
        from dash import callback_context

        if active_tab != 'comparison':
            return False

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        if trigger in (
            'analysis-sidebar-menu',
            'analysis-metric-select',
            'analysis-normalization-select',
            'analysis-grouping-select',
            'comparison-sample1-select',
            'comparison-sample2-select',
            'comparison-pvalue-threshold',
            'comparison-fdr-method',
            'comparison-plot-type',
        ):
            return True

        return not figure

    @app.callback(
        Output({'type': 'comparison-plot', 'index': 'main'}, 'figure'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('comparison-sample1-select', 'value'),
        Input('comparison-sample2-select', 'value'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('comparison-pvalue-threshold', 'value'),
        Input('comparison-fdr-method', 'value'),
        Input('comparison-plot-type', 'checked'),
        Input('wdir', 'data'),
        prevent_initial_call=True,
    )
    def update_comparison_plot(active_tab, sample1, sample2, metric, norm_value, group_by_col, p_threshold, fdr_method, plot_toggle, wdir):
        if active_tab != 'comparison':
            return dash.no_update
        if not wdir:
            return _empty_plot("Select a workspace to compare features.", height=520)
        if not sample1 or not sample2:
            return _empty_plot("Select Sample 1 and Sample 2 to compare.", height=520)
        if sample1 == sample2:
            return _empty_plot("Select two different groups to compare.", height=520)

        p_threshold = 0.05 if p_threshold is None else float(p_threshold)
        p_threshold = max(min(p_threshold, 1.0), 0.0)
        fdr_method = fdr_method or 'none'

        # Read-only optimization
        with analysis_read_connection(wdir, conn_factory=duckdb_connection) as conn:
            if conn is None:
                return _empty_plot("Unable to connect to workspace database.", height=520)

            data = prepare_metric_table(conn, wdir, metric)
            if data is None or data.empty:
                return _empty_plot("No data available for comparison.", height=520)

            data.set_index('ms_file_label', inplace=True)
            group_field = group_by_col if group_by_col in data.columns else (
                'sample_type' if 'sample_type' in data.columns else None
            )
            group_label = GROUP_LABELS.get(group_field, group_field or 'Group')
            missing_group_label = f"{group_label} (unset)"
            metadata_cols = [col for col in ['ms_type', 'sample_type', *GROUP_COLUMNS] if col in data.columns]

            if group_field:
                group_series = data[group_field].replace("", pd.NA)
            else:
                group_series = pd.Series(data.index, index=data.index, name='group')
            group_series = group_series.fillna(missing_group_label)

            numeric_df = data.drop(columns=[c for c in metadata_cols if c in data.columns], axis=1)
            numeric_df = normalize_matrix(numeric_df, norm_value)
            if numeric_df.empty:
                return _empty_plot("No numeric data available for comparison.", height=520)
            group_series = group_series.reindex(numeric_df.index)

            colors_df = conn.execute(
                f"SELECT ms_file_label, color, sample_type, {', '.join(GROUP_COLUMNS)} FROM samples"
            ).df()
            color_map = _build_color_map(
                colors_df, group_field, use_sample_colors=(group_field == 'sample_type')
            )
            color_map.setdefault(missing_group_label, '#bbbbbb')

        group1_mask = group_series == sample1
        group2_mask = group_series == sample2
        if not group1_mask.any() or not group2_mask.any():
            return _empty_plot("Selected groups have no samples.", height=520)

        group1 = numeric_df.loc[group1_mask]
        group2 = numeric_df.loc[group2_mask]

        mean1 = group1.mean(axis=0)
        mean2 = group2.mean(axis=0)

        if group1.shape[0] < 2 or group2.shape[0] < 2:
            pvals = np.full(len(numeric_df.columns), np.nan, dtype=float)
        else:
            ttest = ttest_ind(
                group1.to_numpy(),
                group2.to_numpy(),
                axis=0,
                equal_var=False,
                nan_policy='omit',
            )
            pvals = ttest.pvalue

        p_adj = _adjust_pvalues(pvals, fdr_method)
        min_positive = np.nanmin(p_adj[p_adj > 0]) if np.any(p_adj > 0) else 1e-300
        min_positive = max(min_positive, 1e-300)

        use_log2fc = norm_value not in ('durbin', 'zscore_durbin')
        if use_log2fc:
            fc_values = np.log2((mean2.values + 1e-12) / (mean1.values + 1e-12))
            fc_label = f"Log2 Fold Change ({sample2}/{sample1})"
            fc_hover = "Log2FC"
        else:
            fc_values = mean2.values - mean1.values
            fc_label = f"Δ ({sample2} - {sample1})"
            fc_hover = "Δ"

        plot_df = pd.DataFrame(
            {
                'feature': numeric_df.columns,
                'mean_a': mean1.values,
                'mean_b': mean2.values,
                'log2fc': fc_values,
                'p_value': pvals,
                'p_adj': p_adj,
            }
        )
        plot_df['neg_log10_p'] = -np.log10(plot_df['p_adj'].clip(lower=min_positive))
        plot_df['significant'] = (plot_df['p_adj'] <= p_threshold) & np.isfinite(plot_df['p_adj'])

        n1 = group1.shape[0]
        n2 = group2.shape[0]
        title = f"{sample1} (n={n1}) vs {sample2} (n={n2})"

        sig_mask = plot_df['significant'].to_numpy()
        no_sig = not np.any(sig_mask)

        group1_color = color_map.get(sample1, '#1f77b4')
        group2_color = color_map.get(sample2, '#ff7f0e')
        if sample1 == missing_group_label:
            group1_color = '#bbbbbb'
        if sample2 == missing_group_label:
            group2_color = '#bbbbbb'

        def _hex_to_rgb(value):
            if not isinstance(value, str):
                return (120, 120, 120)
            value = value.strip().lstrip('#')
            if len(value) != 6:
                return (120, 120, 120)
            return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))

        def _rgba(hex_color, alpha):
            r, g, b = _hex_to_rgb(hex_color)
            return f"rgba({r}, {g}, {b}, {alpha:.3f})"

        # Intensity based on adjusted p-values (lower p => more intense)
        p_adj_vals = plot_df['p_adj'].to_numpy(dtype=float)
        p_adj_clip = np.clip(p_adj_vals, 1e-300, 1.0)
        logp = -np.log10(p_adj_clip)
        finite_mask = np.isfinite(logp)
        if finite_mask.any():
            min_lp = float(np.nanmin(logp[finite_mask]))
            max_lp = float(np.nanmax(logp[finite_mask]))
        else:
            min_lp = max_lp = 0.0
        denom = max_lp - min_lp if max_lp != min_lp else 1.0
        strength = (logp - min_lp) / denom
        strength = np.clip(strength, 0.0, 1.0)
        alpha_vals = 0.2 + 0.8 * strength

        # Size based on absolute difference between group means
        diff_vals = np.abs(plot_df['mean_b'].to_numpy() - plot_df['mean_a'].to_numpy())
        if np.isfinite(diff_vals).any():
            min_d = float(np.nanmin(diff_vals))
            max_d = float(np.nanmax(diff_vals))
        else:
            min_d = max_d = 0.0
        denom_d = max_d - min_d if max_d != min_d else 1.0
        size_strength = (diff_vals - min_d) / denom_d
        size_strength = np.clip(size_strength, 0.0, 1.0)
        size_vals = 6 + 10 * size_strength

        def build_customdata(frame, cols):
            if frame.empty:
                return np.empty((0, len(cols)))
            return frame[cols].to_numpy()

        fig = go.Figure()
        if plot_toggle:
            x_vals = plot_df['log2fc'].to_numpy() if no_sig else plot_df.loc[sig_mask, 'log2fc'].to_numpy()
            y_vals = plot_df['neg_log10_p'].to_numpy() if no_sig else plot_df.loc[sig_mask, 'neg_log10_p'].to_numpy()
            if no_sig:
                fig.add_trace(go.Scatter(
                    x=plot_df['log2fc'],
                    y=plot_df['neg_log10_p'],
                    mode='markers',
                    name='Not significant',
                    marker=dict(
                        color=[_rgba('#9aa0a6', a) for a in alpha_vals],
                        size=size_vals,
                        line=dict(width=0.6, color='#444'),
                    ),
                    customdata=build_customdata(
                        plot_df,
                        ['feature', 'mean_a', 'mean_b', 'p_value', 'p_adj'],
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{fc_hover}: %{{x:.3f}}<br>"
                        f"{sample1} mean: %{{customdata[1]:.3e}}<br>"
                        f"{sample2} mean: %{{customdata[2]:.3e}}<br>"
                        "p-value: %{customdata[3]:.3e}<br>"
                        "FDR: %{customdata[4]:.3e}<extra></extra>"
                    ),
                ))
            else:
                higher_in_1 = sig_mask & (plot_df['mean_a'] >= plot_df['mean_b'])
                higher_in_2 = sig_mask & (plot_df['mean_b'] > plot_df['mean_a'])
                fig.add_trace(go.Scatter(
                    x=plot_df.loc[higher_in_1, 'log2fc'],
                    y=plot_df.loc[higher_in_1, 'neg_log10_p'],
                    mode='markers',
                    name=str(sample1),
                    marker=dict(
                        color=[_rgba(group1_color, a) for a in alpha_vals[higher_in_1]],
                        size=size_vals[higher_in_1],
                        line=dict(width=0.6, color='#444'),
                    ),
                    customdata=build_customdata(
                        plot_df.loc[higher_in_1],
                        ['feature', 'mean_a', 'mean_b', 'p_value', 'p_adj'],
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{fc_hover}: %{{x:.3f}}<br>"
                        f"{sample1} mean: %{{customdata[1]:.3e}}<br>"
                        f"{sample2} mean: %{{customdata[2]:.3e}}<br>"
                        "p-value: %{customdata[3]:.3e}<br>"
                        "FDR: %{customdata[4]:.3e}<extra></extra>"
                    ),
                ))
                fig.add_trace(go.Scatter(
                    x=plot_df.loc[higher_in_2, 'log2fc'],
                    y=plot_df.loc[higher_in_2, 'neg_log10_p'],
                    mode='markers',
                    name=str(sample2),
                    marker=dict(
                        color=[_rgba(group2_color, a) for a in alpha_vals[higher_in_2]],
                        size=size_vals[higher_in_2],
                        line=dict(width=0.6, color='#444'),
                    ),
                    customdata=build_customdata(
                        plot_df.loc[higher_in_2],
                        ['feature', 'mean_a', 'mean_b', 'p_value', 'p_adj'],
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Log2FC: %{x:.3f}<br>"
                        f"{sample1} mean: %{{customdata[1]:.3e}}<br>"
                        f"{sample2} mean: %{{customdata[2]:.3e}}<br>"
                        "p-value: %{customdata[3]:.3e}<br>"
                        "FDR: %{customdata[4]:.3e}<extra></extra>"
                    ),
                ))
            fig.add_hline(y=-np.log10(p_threshold if p_threshold > 0 else min_positive), line_dash="dash", line_color="#999")
            fig.add_vline(x=0, line_dash="dash", line_color="#999")
            fig.update_layout(
                title=title,
                xaxis_title=fc_label,
                yaxis_title="-Log10(FDR)",
                template='plotly_white',
                margin=dict(l=40, r=20, t=80, b=60),
                clickmode='event',
                autosize=True,
            )
            if x_vals.size and y_vals.size:
                x_max = float(np.nanmax(x_vals))
                y_min = 0.0
                y_max = float(np.nanmax(y_vals))
                y_max += (y_max * 0.05)
                fig.update_xaxes(range=[-x_max - (x_max * 0.05), x_max + (x_max * 0.05)], scaleratio=1)
                fig.update_yaxes(range=[y_min, y_max])
        else:
            x_vals = plot_df['mean_a'].to_numpy() if no_sig else plot_df.loc[sig_mask, 'mean_a'].to_numpy()
            y_vals = plot_df['mean_b'].to_numpy() if no_sig else plot_df.loc[sig_mask, 'mean_b'].to_numpy()
            if no_sig:
                fig.add_trace(go.Scatter(
                    x=plot_df['mean_a'],
                    y=plot_df['mean_b'],
                    mode='markers',
                    name='Not significant',
                    marker=dict(
                        color=[_rgba('#9aa0a6', a) for a in alpha_vals],
                        size=size_vals,
                        line=dict(width=0.6, color='#444'),
                    ),
                    customdata=build_customdata(
                        plot_df,
                        ['feature', 'log2fc', 'p_value', 'p_adj'],
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{sample1} mean: %{{x:.3e}}<br>"
                        f"{sample2} mean: %{{y:.3e}}<br>"
                        f"{fc_hover}: %{{customdata[1]:.3f}}<br>"
                        "p-value: %{customdata[2]:.3e}<br>"
                        "FDR: %{customdata[3]:.3e}<extra></extra>"
                    ),
                ))
            else:
                higher_in_1 = sig_mask & (plot_df['mean_a'] >= plot_df['mean_b'])
                higher_in_2 = sig_mask & (plot_df['mean_b'] > plot_df['mean_a'])
                fig.add_trace(go.Scatter(
                    x=plot_df.loc[higher_in_1, 'mean_a'],
                    y=plot_df.loc[higher_in_1, 'mean_b'],
                    mode='markers',
                    name=str(sample1),
                    marker=dict(
                        color=[_rgba(group1_color, a) for a in alpha_vals[higher_in_1]],
                        size=size_vals[higher_in_1],
                        line=dict(width=0.6, color='#444'),
                    ),
                    customdata=build_customdata(
                        plot_df.loc[higher_in_1],
                        ['feature', 'log2fc', 'p_value', 'p_adj'],
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{sample1} mean: %{{x:.3e}}<br>"
                        f"{sample2} mean: %{{y:.3e}}<br>"
                        f"{fc_hover}: %{{customdata[1]:.3f}}<br>"
                        "p-value: %{customdata[2]:.3e}<br>"
                        "FDR: %{customdata[3]:.3e}<extra></extra>"
                    ),
                ))
                fig.add_trace(go.Scatter(
                    x=plot_df.loc[higher_in_2, 'mean_a'],
                    y=plot_df.loc[higher_in_2, 'mean_b'],
                    mode='markers',
                    name=str(sample2),
                    marker=dict(
                        color=[_rgba(group2_color, a) for a in alpha_vals[higher_in_2]],
                        size=size_vals[higher_in_2],
                        line=dict(width=0.6, color='#444'),
                    ),
                    customdata=build_customdata(
                        plot_df.loc[higher_in_2],
                        ['feature', 'log2fc', 'p_value', 'p_adj'],
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{sample1} mean: %{{x:.3e}}<br>"
                        f"{sample2} mean: %{{y:.3e}}<br>"
                        "Log2FC: %{customdata[1]:.3f}<br>"
                        "p-value: %{customdata[2]:.3e}<br>"
                        "FDR: %{customdata[3]:.3e}<extra></extra>"
                    ),
                ))
            min_val = float(np.nanmin([plot_df['mean_a'].min(), plot_df['mean_b'].min(), 0]))
            max_val = float(np.nanmax([plot_df['mean_a'].max(), plot_df['mean_b'].max(), 0]))
            if np.isfinite(min_val) and np.isfinite(max_val):
                fig.add_shape(
                    type='line',
                    x0=min_val,
                    y0=min_val,
                    x1=max_val,
                    y1=max_val,
                    line=dict(color='#999', dash='dash'),
                )
            fig.update_layout(
                title=title,
                xaxis_title=f"Signal Intensity in {sample1} Samples",
                yaxis_title=f"Signal Intensity in {sample2} Samples",
                template='plotly_white',
                margin=dict(l=40, r=20, t=80, b=60),
                clickmode='event',
            )
            if x_vals.size and y_vals.size:
                x_min = float(np.nanmin(x_vals))
                x_max = float(np.nanmax(x_vals))
                y_min = float(np.nanmin(y_vals))
                y_max = float(np.nanmax(y_vals))
                max_val = max(x_max, y_max, 0.0)
                max_val += (max_val * 0.05)
                min_val = min(x_min, y_min, 0.0)
                min_val -= (min_val * 0.05)
                fig.update_xaxes(range=[min_val, max_val], scaleratio=1)
                fig.update_yaxes(range=[min_val, max_val])

        fig.update_traces(hoverlabel=dict(namelength=-1))
        fig.update_layout(
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            uirevision=f"{sample1}-{sample2}-{metric}-{norm_value}-{fdr_method}",
            autosize=True,
            dragmode='zoom',
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=True, showline=True, linecolor='#444'),
            yaxis=dict(showgrid=False, zeroline=True, showline=True, linecolor='#444'),
        )
        return fig

    @app.callback(
        Output('comparison-chromatogram', 'figure'),
        Output('comparison-chromatogram-container', 'style'),
        Output('comparison-active-peak-store', 'data'),
        Input({'type': 'comparison-plot', 'index': 'main'}, 'clickData'),
        Input('comparison-sample1-select', 'value'),
        Input('comparison-sample2-select', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('comparison-log-scale-switch', 'checked'),
        Input('wdir', 'data'),
        prevent_initial_call=True,
    )
    def update_chromatogram(clickData, sample1, sample2, group_by_col, log_scale, wdir):
        default_chrom_style = {'width': '50%'}

        if not wdir or not sample1 or not sample2:
            return _empty_plot("Click a feature to view chromatograms."), default_chrom_style, None

        try:
            peak_label = clickData['points'][0]['customdata'][0]
        except (KeyError, IndexError, TypeError):
            return _empty_plot("Click a feature to view chromatograms."), default_chrom_style, None

        group_by_col = ensure_valid_group_field(group_by_col, allow_none=True, default=None)
        group_label = GROUP_LABELS.get(group_by_col, group_by_col or 'Group')
        missing_group_label = f"{group_label} (unset)"

        # Read-only optimization
        with analysis_read_connection(wdir, conn_factory=duckdb_connection) as conn:
            if conn is None:
                return _empty_plot("Unable to connect to workspace database."), default_chrom_style, None

            rt_info = conn.execute(
                "SELECT rt_min, rt_max FROM targets WHERE peak_label = ?",
                [peak_label],
            ).fetchone()
            rt_min, rt_max = rt_info if rt_info else (None, None)

            def fetch_group_samples(group_value, limit):
                if not group_by_col:
                    return []
                if group_value == missing_group_label:
                    query = f"""
                        SELECT ms_file_label, color
                        FROM samples
                        WHERE use_for_analysis = TRUE
                          AND ("{group_by_col}" IS NULL OR "{group_by_col}" = '')
                        ORDER BY random()
                        LIMIT {limit}
                    """
                    return conn.execute(query).fetchall()
                query = f"""
                    SELECT ms_file_label, color
                    FROM samples
                    WHERE use_for_analysis = TRUE
                      AND "{group_by_col}" = ?
                    ORDER BY random()
                    LIMIT {limit}
                """
                return conn.execute(query, [group_value]).fetchall()

            group1_samples = fetch_group_samples(sample1, 6)
            group2_samples = fetch_group_samples(sample2, 6)
            files_to_fetch = [row[0] for row in group1_samples + group2_samples]
            if not files_to_fetch:
                return _empty_plot("No chromatogram data found for selected groups."), default_chrom_style, None

            placeholders = ','.join(['?'] * len(files_to_fetch))
            chrom_query = f"""
                SELECT c.ms_file_label, c.scan_time, c.intensity, s.color
                FROM chromatograms c
                JOIN samples s ON c.ms_file_label = s.ms_file_label
                WHERE c.peak_label = ? AND c.ms_file_label IN ({placeholders})
            """
            chrom_data = conn.execute(chrom_query, [peak_label] + files_to_fetch).fetchall()
            if not chrom_data:
                return _empty_plot("No chromatogram data found."), default_chrom_style, None

            data_map = {row[0]: row for row in chrom_data}
            colors_df = conn.execute(
                f'SELECT ms_file_label, color, sample_type, "{group_by_col}" FROM samples'
            ).df()
            color_map = _build_color_map(
                colors_df, group_by_col, use_sample_colors=(group_by_col == 'sample_type')
            )

        fig = go.Figure()

        def add_group_traces(samples, group_value, group_color):
            show_legend = True
            for ms_file_label, _ in samples:
                if ms_file_label not in data_map:
                    continue
                _, scan_times, intensities, _ = data_map[ms_file_label]
                if len(intensities) > 0 and min(intensities) == max(intensities):
                    continue
                plot_intensities = np.log2(np.array(intensities) + 1) if log_scale else intensities
                fig.add_trace(go.Scatter(
                    x=scan_times,
                    y=plot_intensities,
                    mode='lines',
                    name=str(group_value),
                    legendgroup=str(group_value),
                    showlegend=show_legend,
                    line=dict(width=1.5, color=group_color),
                    opacity=0.6,
                    hovertemplate=(
                        f"<b>{ms_file_label}</b><br>"
                        "Scan Time: %{x:.2f}<br>"
                        f"Intensity: %{{y:.2e}}<extra>{group_value}</extra>"
                    ),
                ))
                show_legend = False

        group1_color = color_map.get(sample1, '#1f77b4')
        group2_color = color_map.get(sample2, '#ff7f0e')
        if sample1 == missing_group_label:
            group1_color = '#bbbbbb'
        if sample2 == missing_group_label:
            group2_color = '#bbbbbb'

        add_group_traces(group1_samples, sample1, group1_color)
        add_group_traces(group2_samples, sample2, group2_color)

        if rt_min is not None and rt_max is not None:
            fig.add_vrect(
                x0=rt_min,
                x1=rt_max,
                fillcolor="green",
                opacity=0.1,
                layer="below",
                line_width=0,
            )

        x_range_min = None
        x_range_max = None
        if rt_min is not None and rt_max is not None:
            padding = 5
            x_range_min = rt_min - padding
            x_range_max = rt_max + padding

            all_x = []
            for trace in fig.data:
                if hasattr(trace, 'x') and trace.x is not None:
                    all_x.extend(trace.x)
            if all_x:
                x_range_min = max(x_range_min, min(all_x))
                x_range_max = min(x_range_max, max(all_x))

        y_range = None
        if x_range_min is not None and x_range_max is not None:
            traces_data = [{'x': list(t.x), 'y': list(t.y)} for t in fig.data if hasattr(t, 'x') and hasattr(t, 'y')]
            y_range = _calc_y_range_numpy(traces_data, x_range_min, x_range_max, is_log=False)

        y_title = "Intensity (Log2)" if log_scale else "Intensity"
        fig.update_layout(
            title=dict(text=f"{peak_label} | {sample1} vs {sample2}", font=dict(size=14)),
            xaxis_title="Scan Time (s)",
            yaxis_title=y_title,
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            xaxis_tickfont=dict(size=12),
            yaxis_tickfont=dict(size=12),
            template="plotly_white",
            margin=dict(l=50, r=20, t=110, b=80),
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

        fig.update_xaxes(rangemode='tozero')
        fig.update_yaxes(rangemode='tozero')
        fig.update_layout(clickmode='event')
        fig.update_layout(autosize=True)

        return fig, default_chrom_style, peak_label

    @app.callback(
        Output('comparison-chromatogram', 'figure', allow_duplicate=True),
        Input('comparison-chromatogram', 'relayoutData'),
        State('comparison-chromatogram', 'figure'),
        prevent_initial_call=True,
    )
    def update_comparison_chromatogram_zoom(relayout, figure_state):
        if not relayout or not figure_state:
            raise PreventUpdate

        x_range = (relayout.get('xaxis.range[0]'), relayout.get('xaxis.range[1]'))
        y_range = (relayout.get('yaxis.range[0]'), relayout.get('yaxis.range[1]'))

        if y_range[0] is not None and y_range[1] is not None:
            raise PreventUpdate
        if x_range[0] is None or x_range[1] is None:
            raise PreventUpdate

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
        Output({'type': 'comparison-plot', 'index': 'main'}, 'figure', allow_duplicate=True),
        Input({'type': 'comparison-plot', 'index': 'main'}, 'clickData'),
        State({'type': 'comparison-plot', 'index': 'main'}, 'figure'),
        prevent_initial_call=True,
    )
    def highlight_comparison_point(click_data, figure_state):
        if not click_data or not figure_state:
            raise PreventUpdate

        try:
            point = click_data['points'][0]
            x_val = point.get('x')
            y_val = point.get('y')
        except (KeyError, IndexError, TypeError):
            raise PreventUpdate

        if x_val is None or y_val is None:
            raise PreventUpdate

        def _axis_min(layout_axis, data_key):
            if isinstance(layout_axis, dict):
                axis_range = layout_axis.get('range')
                if axis_range and len(axis_range) == 2 and axis_range[0] is not None:
                    return axis_range[0]
            vals = []
            for trace in figure_state.get('data', []):
                series = trace.get(data_key)
                if series:
                    vals.extend([v for v in series if v is not None])
            return min(vals) if vals else None

        layout = figure_state.get('layout', {})
        x_min = _axis_min(layout.get('xaxis', {}), 'x')
        y_min = _axis_min(layout.get('yaxis', {}), 'y')
        if x_min is None or y_min is None:
            raise PreventUpdate

        from dash import Patch
        fig_patch = Patch()
        existing_shapes = layout.get('shapes', [])
        preserved_shapes = [
            shape for shape in existing_shapes
            if shape.get('name') != 'comparison-click-crosshair'
        ]
        fig_patch['layout']['shapes'] = preserved_shapes + [
            {
                'type': 'line',
                'x0': x_val,
                'x1': x_val,
                'y0': y_min,
                'y1': y_val,
                'xref': 'x',
                'yref': 'y',
                'line': {'color': '#999', 'width': 1, 'dash': 'dot'},
                'name': 'comparison-click-crosshair',
            },
            {
                'type': 'line',
                'x0': x_min,
                'x1': x_val,
                'y0': y_val,
                'y1': y_val,
                'xref': 'x',
                'yref': 'y',
                'line': {'color': '#999', 'width': 1, 'dash': 'dot'},
                'name': 'comparison-click-crosshair',
            },
        ]

        return fig_patch


    @app.callback(
        Output('comparison-selection-store', 'data'),
        Output('comparison-selection-count', 'children'),
        Output('comparison-selection-list-container', 'children'),
        Output('analysis-notifications-container', 'children', allow_duplicate=True),
        Input('comparison-add-btn', 'nClicks'),
        Input('comparison-clear-btn', 'nClicks'),
        State('comparison-active-peak-store', 'data'),
        State('comparison-selection-store', 'data'),
        prevent_initial_call=True
    )
    def manage_selection_list(n_add, n_clear, active_peak, current_selection):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        current_selection = current_selection or []
        
        def _render_tags(items):
            if not items:
                return None
            return fac.AntdFlex(
                [fac.AntdTag(content=item, color='#3b8fa3') for item in items],
                wrap='wrap',
                gap='small'
            )

        if trigger_id == 'comparison-clear-btn':
            return [], "0 selected", None, None
        
        if trigger_id == 'comparison-add-btn':
            if not active_peak:
                return dash.no_update, dash.no_update, dash.no_update, fac.AntdNotification(
                    message="No feature selected",
                    description="Click a point on the plot first to select a feature.",
                    type="warning",
                    duration=3,
                    placement='bottomRight'
                )
            
            if active_peak in current_selection:
                return dash.no_update, dash.no_update, dash.no_update, fac.AntdNotification(
                    message="Already selected",
                    description=f"Feature '{active_peak}' is already in your list.",
                    type="info",
                    duration=2,
                    placement='bottomRight'
                )
            
            new_selection = current_selection + [active_peak]
            return new_selection, f"{len(new_selection)} selected", _render_tags(new_selection), fac.AntdNotification(
                message="Feature added",
                description=f"Added '{active_peak}' to selection list.",
                type="success",
                duration=2,
                placement='bottomRight'
            )
            
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


    @app.callback(
        Output('comparison-download-selection', 'data'),
        Input('comparison-download-btn', 'nClicks'),
        State('comparison-selection-store', 'data'),
        State('wdir', 'data'),
        State('analysis-metric-select', 'value'),
        prevent_initial_call=True
    )
    def download_selection_list(n_clicks, selection, wdir, metric):
        if not n_clicks or not selection:
            return dash.no_update
        
        if not wdir:
             df = pd.DataFrame({'peak_label': selection})
             return dcc.send_data_frame(df.to_csv, "selected_features.csv", index=False)

        metric = metric or 'peak_max'
        
        # Read-only optimization
        with analysis_read_connection(wdir, conn_factory=duckdb_connection) as conn:
             if conn is None:
                 return dash.no_update
             
             # Sanitize and format selection for SQL injection (safely for internal state)
             # Escaping single quotes in peak labels if necessary
             safe_selection = [s.replace("'", "''") for s in selection]
             in_clause = ", ".join([f"'{s}'" for s in safe_selection])
             
             try:
                 # 1. Get Targets Metadata
                 # We can use parameters here as it is a standard SELECT
                 placeholders = ','.join(['?'] * len(selection))
                 targets_query = f"SELECT * FROM targets WHERE peak_label IN ({placeholders})"
                 df_targets = conn.execute(targets_query, selection).df()
                 
                 # 2. Get Pivot (Using formatted string for IN clause to avoid Pivot+Param limitations)
                 pivot_query = f"""
                    PIVOT (
                        SELECT peak_label, ms_file_label, {metric}
                        FROM results
                        WHERE peak_label IN ({in_clause})
                    ) ON ms_file_label USING FIRST({metric})
                 """
                 df_pivot = conn.execute(pivot_query).df()
                 
                 # 3. Merge
                 final_df = pd.merge(df_targets, df_pivot, on='peak_label', how='left')
                 
             except duckdb.Error as exc:
                 logger.warning(
                     "Failed to build comparison selection pivot for workspace %s; falling back to targets-only export: %s",
                     wdir,
                     exc,
                 )
                 # Fallback: just targets when pivot/query fails.
                 final_df = conn.execute(f"SELECT * FROM targets WHERE peak_label IN ({placeholders})", selection).df()
        
        ws_name = get_workspace_name_from_wdir(wdir) or "workspace"
        filename = f"{date.today()}-MINT__{ws_name}_selected_{metric}.csv"
        return dcc.send_data_frame(final_df.to_csv, filename, index=False)
