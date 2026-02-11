"""Shared data prep and output-shape helpers for Analysis plugin callbacks."""

import logging

import dash
import duckdb
import pandas as pd
from dash.exceptions import PreventUpdate

from ...duckdb_manager import duckdb_connection
from ._shared import (
    GROUP_COLUMNS,
    GROUP_LABELS,
    _build_color_map,
    _clean_numeric,
    get_analysis_read_limits,
    normalize_matrices,
    prepare_metric_table,
)

logger = logging.getLogger(__name__)


_CONTENT_OUTPUT_FIELDS = (
    'clustermap_src',
    'pca_figure',
    'tsne_figure',
    'violin_children',
    'violin_options',
    'violin_value',
    'bar_children',
    'bar_options',
    'bar_value',
    'pca_cache',
    'tsne_cache',
    'violin_cache',
    'bar_cache',
)


def _build_content_outputs(**updates):
    data = {field: dash.no_update for field in _CONTENT_OUTPUT_FIELDS}
    data.update(updates)
    return tuple(data[field] for field in _CONTENT_OUTPUT_FIELDS)


def _no_update_outputs():
    return _build_content_outputs()


def _empty_content_outputs(invisible_fig):
    """Return update_content outputs for empty/unavailable analysis data."""
    return _build_content_outputs(
        clustermap_src=None,
        pca_figure=invisible_fig,
        tsne_figure=invisible_fig,
        violin_children=[],
        violin_options=[],
        violin_value=[],
        bar_children=[],
        bar_options=[],
        bar_value=[],
    )


def _return_clustermap(src):
    return _build_content_outputs(clustermap_src=src)


def _return_pca(fig, compound_options=dash.no_update, pca_cache_data=dash.no_update):
    return _build_content_outputs(
        pca_figure=fig,
        violin_options=compound_options,
        pca_cache=pca_cache_data,
    )


def _return_tsne(fig, tsne_cache_data=dash.no_update):
    return _build_content_outputs(
        tsne_figure=fig,
        tsne_cache=tsne_cache_data,
    )


def _return_violin(graphs, options, value, violin_cache_data=dash.no_update):
    return _build_content_outputs(
        violin_children=graphs,
        violin_options=options,
        violin_value=value,
        violin_cache=violin_cache_data,
    )


def _return_bar(graphs, options, value, bar_cache_data=dash.no_update):
    return _build_content_outputs(
        bar_children=graphs,
        bar_options=options,
        bar_value=value,
        bar_cache=bar_cache_data,
    )


def _prepare_matrix_data(
    wdir,
    metric,
    selected_group,
    grouping_fields,
    norm_value,
    invisible_fig,
    conn_factory=duckdb_connection,
):
    """Load and normalize analysis matrices shared by PCA/t-SNE/violin/bar/clustermap."""
    cpus, ram = get_analysis_read_limits()

    with conn_factory(wdir, n_cpus=cpus, ram=ram, read_only=True) as conn:
        if conn is None:
            return None, _empty_content_outputs(invisible_fig)

        try:
            results_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            if results_count == 0:
                return None, _empty_content_outputs(invisible_fig)
        except duckdb.Error as exc:
            logger.warning("Failed to check analysis results availability for workspace %s: %s", wdir, exc)
            return None, _empty_content_outputs(invisible_fig)

        df = prepare_metric_table(conn, wdir, metric)
        if df is None or df.empty:
            return None, _empty_content_outputs(invisible_fig)
        df.set_index('ms_file_label', inplace=True)

        group_field = selected_group if selected_group in df.columns else (
            'sample_type' if 'sample_type' in df.columns else None
        )
        group_label = GROUP_LABELS.get(group_field, 'Group')
        missing_group_label = f"{group_label} (unset)"
        metadata_cols = [col for col in ['ms_type'] + grouping_fields if col in df.columns]

        order_df_raw = conn.execute(
            "SELECT ms_file_label FROM samples ORDER BY ms_file_label"
        ).df()
        if order_df_raw.empty:
            return None, _empty_content_outputs(invisible_fig)

        order_df = order_df_raw["ms_file_label"].tolist()
        ordered_labels = [lbl for lbl in order_df if lbl in df.index]
        leftover_labels = [lbl for lbl in df.index if lbl not in ordered_labels]
        df = df.loc[ordered_labels + leftover_labels]

        # Sort samples by group, then alphabetically within each group for clustermap display
        if group_field and group_field in df.columns:
            df['_sort_group'] = df[group_field].fillna('')
            df['_sort_index'] = df.index.str.lower()
            df = df.sort_values(by=['_sort_group', '_sort_index'])
            df = df.drop(columns=['_sort_group', '_sort_index'])

        group_series = df[group_field] if group_field else pd.Series(df.index, index=df.index, name='group')
        if isinstance(group_series, pd.Series):
            group_series = group_series.replace("", pd.NA)
        group_series.name = group_field or 'group'
        group_series = group_series.fillna(missing_group_label)

        colors_df = conn.execute(
            f"SELECT ms_file_label, color, sample_type, {', '.join(GROUP_COLUMNS)} FROM samples"
        ).df()
        color_map = _build_color_map(colors_df, group_field, use_sample_colors=(group_field == 'sample_type'))
        if missing_group_label in group_series.values:
            if color_map is None:
                color_map = {}
            color_map.setdefault(missing_group_label, '#bbbbbb')

        raw_df = df.copy()
        df = df.drop(columns=[c for c in metadata_cols if c in df.columns], axis=1)

        compound_options = sorted(
            [
                {'label': c, 'value': c}
                for c in raw_df.columns
                if c not in metadata_cols
            ],
            key=lambda o: o['label'].lower(),
        )

        # Guard against NaN/inf and empty matrices (numeric only) before downstream plots
        df = _clean_numeric(df)
        raw_numeric_cols = [c for c in raw_df.columns if c not in metadata_cols]
        raw_numeric = _clean_numeric(raw_df[raw_numeric_cols])
        raw_df[raw_numeric_cols] = raw_numeric
        color_labels = group_series.reindex(df.index).fillna(missing_group_label)

        if df.empty or raw_numeric.empty:
            return None, _empty_content_outputs(invisible_fig)

        ndf, zdf = normalize_matrices(df, norm_value)
        if ndf.empty or zdf.empty:
            raise PreventUpdate

        # Choose matrix for violin/bar based on normalization selection
        if norm_value == 'zscore':
            violin_matrix = zdf
        elif norm_value in ('durbin', 'zscore_durbin'):
            violin_matrix = ndf
        else:
            violin_matrix = df

        if violin_matrix.empty:
            return None, _build_content_outputs(
                pca_figure=invisible_fig,
                tsne_figure=invisible_fig,
                violin_children=[],
                violin_options=[],
                violin_value=[],
                bar_children=[],
                bar_options=[],
                bar_value=[],
            )

    return {
        'ndf': ndf,
        'zdf': zdf,
        'violin_matrix': violin_matrix,
        'group_series': group_series,
        'group_label': group_label,
        'color_labels': color_labels,
        'color_map': color_map,
        'colors_df': colors_df,
        'compound_options': compound_options,
    }, None
