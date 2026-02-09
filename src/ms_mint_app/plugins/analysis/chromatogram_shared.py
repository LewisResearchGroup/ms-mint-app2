"""Shared helpers for analysis tab chromatogram interactions."""

import hashlib
import random

from ._shared import (
    PreventUpdate,
    _build_color_map,
    _calc_y_range_numpy,
    dash,
    duckdb_connection,
    ensure_valid_group_field,
    go,
    logger,
    np,
    px,
    GROUP_LABELS,
)


DEFAULT_CHROM_CONTAINER_STYLE = {
    'display': 'block',
    'width': 'calc(38% - 6px)',
    'height': '410px',
}


def resolve_compound_options_and_selection(
    matrix,
    compound_options,
    current_value,
    trigger_id,
    selector_trigger_id,
    run_pca_fn,
    allow_legacy_list=False,
):
    """Return (sorted_options, selected_compound) for tab compound selector."""
    default_compound = None
    sorted_options = compound_options
    loadings_for_sort = None

    if sorted_options:
        try:
            pca_results = run_pca_fn(
                matrix,
                n_components=min(matrix.shape[0], matrix.shape[1], 5),
            )
            loadings = pca_results.get('loadings')
            if loadings is not None and 'PC1' in loadings.columns:
                loadings_for_sort = loadings
                default_compound = loadings['PC1'].abs().idxmax()
        except Exception:
            if sorted_options:
                default_compound = sorted_options[0]['value']

    if not default_compound and sorted_options:
        default_compound = sorted_options[0]['value']

    if loadings_for_sort is not None and 'PC1' in loadings_for_sort.columns:
        pc1_sorted = loadings_for_sort['PC1'].abs().sort_values(ascending=False)
        option_map = {opt['value']: opt for opt in compound_options}
        sorted_options = [option_map[val] for val in pc1_sorted.index if val in option_map]

    user_selected = trigger_id == selector_trigger_id
    selected_compound = None
    if user_selected and current_value:
        if allow_legacy_list and isinstance(current_value, list):
            selected_compound = current_value[0] if current_value else default_compound
        else:
            selected_compound = current_value
    else:
        if allow_legacy_list and isinstance(current_value, list):
            current_value = current_value[0] if current_value else None
        if current_value and current_value in matrix.columns:
            selected_compound = current_value
        else:
            selected_compound = default_compound

    return sorted_options, selected_compound


def update_chromatogram_from_click(
    clickData_list,
    peak_label,
    group_by_col,
    log_scale,
    current_selection,
    wdir,
    click_trigger_key,
    log_switch_trigger_id,
):
    """Shared chromatogram callback body used by violin and bar tabs."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    clickData = clickData_list[0] if clickData_list else None
    trigger_prop = ctx.triggered[0]['prop_id']
    trigger_id = trigger_prop.split('.')[0]
    ms_file_label = None

    if click_trigger_key in trigger_prop:
        try:
            ms_file_label = clickData['points'][0]['customdata'][0]
        except (KeyError, IndexError, TypeError):
            ms_file_label = None
    elif trigger_id == log_switch_trigger_id and current_selection:
        ms_file_label = current_selection

    if not ms_file_label and wdir and peak_label:
        with duckdb_connection(wdir, read_only=True) as conn:
            if conn:
                top_samples = conn.execute(
                    """
                    SELECT ms_file_label
                    FROM chromatograms
                    WHERE peak_label = ?
                    ORDER BY (list_max(intensity) - list_min(intensity)) DESC
                    LIMIT 5
                    """,
                    [peak_label],
                ).fetchall()
                if top_samples:
                    ms_file_label = random.choice(top_samples)[0]
                else:
                    random_sample = conn.execute(
                        """
                        SELECT DISTINCT c.ms_file_label
                        FROM chromatograms c
                        WHERE c.peak_label = ?
                        ORDER BY random()
                        LIMIT 1
                        """,
                        [peak_label],
                    ).fetchone()
                    if random_sample:
                        ms_file_label = random_sample[0]

    if not ms_file_label or not wdir or not peak_label:
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template="plotly_white",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig, dict(DEFAULT_CHROM_CONTAINER_STYLE), None

    group_by_col = ensure_valid_group_field(group_by_col, allow_none=True, default=None)

    with duckdb_connection(wdir, read_only=True) as conn:
        if conn is None:
            return dash.no_update, dash.no_update, dash.no_update

        rt_info = conn.execute(
            "SELECT rt_min, rt_max FROM targets WHERE peak_label = ?",
            [peak_label],
        ).fetchone()
        rt_min, rt_max = rt_info if rt_info else (None, None)

        neighbor_files = []
        group_val = None
        group_label = GROUP_LABELS.get(group_by_col, group_by_col or 'Sample')

        if group_by_col:
            try:
                row = conn.execute(
                    f'SELECT "{group_by_col}" FROM samples WHERE ms_file_label = ?',
                    [ms_file_label],
                ).fetchone()
                if row:
                    group_val = row[0]
                    if group_val is None:
                        neighbor_files = conn.execute(
                            f"""
                            SELECT ms_file_label, color
                            FROM samples
                            WHERE "{group_by_col}" IS NULL AND ms_file_label != ?
                            ORDER BY random()
                            LIMIT 10
                            """,
                            [ms_file_label],
                        ).fetchall()
                    else:
                        neighbor_files = conn.execute(
                            f"""
                            SELECT ms_file_label, color
                            FROM samples
                            WHERE "{group_by_col}" = ? AND ms_file_label != ?
                            ORDER BY random()
                            LIMIT 10
                            """,
                            [group_val, ms_file_label],
                        ).fetchall()
            except Exception as exc:
                logger.warning("Failed to fetch chromatogram neighbors: %s", exc)

        color_map = {}
        if group_by_col:
            try:
                colors_df = conn.execute(
                    f'SELECT ms_file_label, color, sample_type, "{group_by_col}" FROM samples'
                ).df()
                color_map = _build_color_map(
                    colors_df,
                    group_by_col,
                    use_sample_colors=(group_by_col == 'sample_type'),
                )
            except Exception as exc:
                logger.warning("Failed to build chromatogram color map: %s", exc)

        display_val = group_val
        if group_by_col and not group_val:
            display_val = f"{group_label} (unset)"

        group_color = None
        if group_by_col:
            if not group_val:
                group_color = '#bbbbbb'
            else:
                group_color = color_map.get(group_val)
                if not group_color:
                    palette = px.colors.qualitative.Plotly
                    digest = hashlib.md5(str(group_val).encode('utf-8')).hexdigest()
                    group_color = palette[int(digest, 16) % len(palette)]

        files_to_fetch = [ms_file_label] + [n[0] for n in neighbor_files]
        placeholders = ','.join(['?'] * len(files_to_fetch))
        chrom_data = conn.execute(
            f"""
            SELECT c.ms_file_label, c.scan_time, c.intensity, s.color
            FROM chromatograms c
            JOIN samples s ON c.ms_file_label = s.ms_file_label
            WHERE c.peak_label = ? AND c.ms_file_label IN ({placeholders})
            """,
            [peak_label] + files_to_fetch,
        ).fetchall()

        if not chrom_data:
            fig = go.Figure()
            fig.add_annotation(text="No chromatogram data found", showarrow=False)
            return fig, dict(DEFAULT_CHROM_CONTAINER_STYLE), ms_file_label

        data_map = {row[0]: row for row in chrom_data}
        fig = go.Figure()

        for n_label, n_color in neighbor_files:
            if n_label in data_map:
                _, scan_times, intensities, _ = data_map[n_label]
                if len(intensities) > 0 and min(intensities) == max(intensities):
                    continue
                if group_by_col:
                    n_color = group_color
                if log_scale:
                    intensities = np.log2(np.array(intensities) + 1)
                fig.add_trace(
                    go.Scatter(
                        x=scan_times,
                        y=intensities,
                        mode='lines',
                        name=str(display_val),
                        legendgroup=str(display_val),
                        showlegend=False,
                        line=dict(width=1, color=n_color),
                        opacity=0.4,
                        hovertemplate=(
                            f"<b>{n_label}</b><br>Scan Time: %{{x:.2f}}"
                            f"<br>Intensity: %{{y:.2e}}<extra>{display_val}</extra>"
                        ),
                    )
                )

        if ms_file_label in data_map:
            _, scan_times, intensities, main_color = data_map[ms_file_label]
            is_flat = len(intensities) > 0 and min(intensities) == max(intensities)
            if is_flat:
                fig.add_annotation(
                    text="Selected sample has no valid signal (flat line)",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                )
            else:
                if group_by_col:
                    main_color = group_color
                legend_name = str(display_val) if group_by_col else ms_file_label
                if log_scale:
                    intensities = np.log2(np.array(intensities) + 1)
                fig.add_trace(
                    go.Scatter(
                        x=scan_times,
                        y=intensities,
                        mode='lines',
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=True,
                        hovertemplate=(
                            f"<b>{ms_file_label}</b><br>Scan Time: %{{x:.2f}}"
                            f"<br>Intensity: %{{y:.2e}}<extra>{legend_name}</extra>"
                        ),
                        line=dict(width=2, color=main_color),
                        fill='tozeroy',
                        opacity=1.0,
                    )
                )

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
                data_x_min = min(all_x)
                data_x_max = max(all_x)
                x_range_min = max(x_range_min, data_x_min)
                x_range_max = min(x_range_max, data_x_max)

        y_range = None
        if x_range_min is not None and x_range_max is not None:
            traces_data = [
                {'x': list(t.x), 'y': list(t.y)}
                for t in fig.data
                if hasattr(t, 'x') and hasattr(t, 'y')
            ]
            y_range = _calc_y_range_numpy(traces_data, x_range_min, x_range_max, is_log=False)

        title_label = ms_file_label
        if len(title_label) > 50:
            title_label = title_label[:20] + "..." + title_label[-20:]

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
            height=410,
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
        return fig, dict(DEFAULT_CHROM_CONTAINER_STYLE), ms_file_label


def patch_chromatogram_zoom(relayout, figure_state):
    """Patch chromatogram y-axis to fit visible x-axis range."""
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
