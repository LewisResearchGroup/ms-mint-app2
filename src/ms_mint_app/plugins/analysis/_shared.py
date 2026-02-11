"""Shared constants, imports, and utilities for Analysis tabs."""

import logging
from collections import OrderedDict
from contextlib import contextmanager
import dash
import base64
from io import BytesIO
from pathlib import Path
import time
import feffery_antd_components as fac
import numpy as np
import pandas as pd
from itertools import cycle
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly import colors as plotly_colors
import os
from sklearn.preprocessing import StandardScaler

from ...duckdb_manager import (
    duckdb_connection,
    create_pivot,
    get_physical_cores,
    calculate_optimal_params,
)
from ...sample_metadata import GROUP_COLUMNS, GROUP_LABELS

logger = logging.getLogger(__name__)

# === CONSTANTS ===

PCA_COMPONENT_OPTIONS = [
    {'label': f'PC{i}', 'value': f'PC{i}'}
    for i in range(1, 6)
]
TSNE_COMPONENT_OPTIONS = [
    {'label': f't-SNE-{i}', 'value': f't-SNE-{i}'}
    for i in range(1, 4)
]
NORM_OPTIONS = [
    {'label': 'None (raw)', 'value': 'none'},
    {'label': 'Z-score', 'value': 'zscore'},
    {'label': 'Rocke-Durbin', 'value': 'durbin'},
    {'label': 'Z-score + Rocke-Durbin', 'value': 'zscore_durbin'},
]
TAB_DEFAULT_NORM = {
    'clustermap': 'zscore',
    'pca': 'durbin',
    'tsne': 'zscore',
    'qc': 'durbin',
    'raincloud': 'durbin',
    'bar': 'durbin',
    'comparison': 'durbin',
}
GROUPING_FIELDS = ['sample_type'] + GROUP_COLUMNS
GROUP_SELECT_OPTIONS = [
    {'label': GROUP_LABELS.get(field, field.replace('_', ' ').title()), 'value': field}
    for field in GROUPING_FIELDS
]

METRIC_OPTIONS = [
    {'label': 'Peak Area', 'value': 'peak_area'},
    {'label': 'Peak Area (Top 3)', 'value': 'peak_area_top3'},
    {'label': 'Peak Max', 'value': 'peak_max'},
    {'label': 'Peak Mean', 'value': 'peak_mean'},
    {'label': 'Peak Median', 'value': 'peak_median'},
    {'label': 'Peak Area (EMG Fitted)', 'value': 'peak_area_fitted'},
    {'label': 'Concentration', 'value': 'scalir_conc'},
]

PLOTLY_HIGH_RES_CONFIG = {
    'toImageButtonOptions': {
        'format': 'svg',
        'filename': 'mint_plot',
        'scale': 1,  # Scale does not matter for SVG, but good to keep clean
        'height': None,
        'width': None,
    },
    'displayModeBar': True,
    'displaylogo': False,
    'responsive': True,
}


def get_download_config(image_format='svg', filename='mint_plot'):
    """Get Plotly config with specified download format and filename."""
    config = PLOTLY_HIGH_RES_CONFIG.copy()
    config['toImageButtonOptions'] = config['toImageButtonOptions'].copy()
    config['toImageButtonOptions']['format'] = image_format
    config['toImageButtonOptions']['filename'] = filename
    return config

allowed_metrics = {
    'peak_area',
    'peak_area_top3',
    'peak_max',
    'peak_mean',
    'peak_median',
    'peak_area_fitted',
    'scalir_conc',
}

# === UTILITY FUNCTIONS ===

def rocke_durbin(df: pd.DataFrame, c: float) -> pd.DataFrame:
    """Apply Rocke-Durbin transformation."""
    z = df.to_numpy(dtype=float)
    ef = np.log((z + np.sqrt(z ** 2 + c ** 2)) / 2.0)
    return pd.DataFrame(ef, index=df.index, columns=df.columns)


def _calc_y_range_numpy(data, x_left, x_right, is_log=False):
    """Calculate the Y-axis range for the given x-range using NumPy."""
    import math
    ys_all = []
    if not data:
        return None
        
    for trace in data:
        xs = trace.get('x')
        ys = trace.get('y')
        
        if xs is None or ys is None or len(xs) == 0:
            continue
            
        try:
            xs = np.array(xs, dtype=np.float64)
            ys = np.array(ys, dtype=np.float64)
        except Exception:
            continue
        
        mask = (xs >= x_left) & (xs <= x_right)
        ys_filtered = ys[mask]
        
        valid_mask = np.isfinite(ys_filtered)
        ys_filtered = ys_filtered[valid_mask]
        
        if len(ys_filtered) > 0:
            ys_all.append(ys_filtered)
            
    if not ys_all:
        return None
        
    ys_concat = np.concatenate(ys_all)
    
    if len(ys_concat) == 0:
        return None

    if is_log:
        pos_mask = ys_concat > 1
        ys_pos = ys_concat[pos_mask]
        if len(ys_pos) == 0:
            return None
        return [math.log10(np.min(ys_pos)), math.log10(np.max(ys_pos) * 1.05)]

    y_min = np.min(ys_concat)
    y_max = np.max(ys_concat)
    y_min = 0 if y_min > 0 else y_min
    return [y_min, y_max * 1.05]


_COLOR_MAP_CACHE_MAX_GROUPS = 16
_COLOR_MAP_MAX_VALUES_PER_GROUP = 512
_COLOR_MAP_CACHE: OrderedDict[str, dict] = OrderedDict()
_ANALYSIS_DB_LIMITS_LOGGED = False


def _cap_color_map_values(color_map: dict, active_values) -> dict:
    """Cap persisted color assignments while prioritizing currently active groups."""
    if len(color_map) <= _COLOR_MAP_MAX_VALUES_PER_GROUP:
        return color_map

    bounded = {}
    for val in active_values:
        if val in color_map and val not in bounded:
            bounded[val] = color_map[val]
            if len(bounded) >= _COLOR_MAP_MAX_VALUES_PER_GROUP:
                return bounded

    for val, color in color_map.items():
        if val in bounded:
            continue
        bounded[val] = color
        if len(bounded) >= _COLOR_MAP_MAX_VALUES_PER_GROUP:
            break
    return bounded


def _build_color_map(color_df: pd.DataFrame, group_col: str, *, use_sample_colors: bool = True) -> dict:
    if not group_col or group_col not in color_df.columns or color_df.empty:
        return {}

    working = color_df[[group_col, 'color']].copy()
    working = working[working[group_col].notna()]
    if use_sample_colors:
        working['color'] = working['color'].apply(
            lambda c: c if isinstance(c, str) and c.strip() and c.strip() != '#bbbbbb' else None
        )
    else:
        working['color'] = None

    # Start with explicit colors from data, then merge with cached assignments.
    explicit_map = (
        working.dropna(subset=['color'])
        .drop_duplicates(subset=[group_col])
        .set_index(group_col)['color']
        .to_dict()
    )

    cached_map = _COLOR_MAP_CACHE.pop(group_col, {}).copy()
    color_map = {**cached_map, **explicit_map}

    active_values = [val for val in working[group_col].dropna().unique()]
    missing = [val for val in active_values if val not in color_map]
    # Optionally force the largest group to a neutral gray for readability
    if missing and not use_sample_colors:
        top_group = working[group_col].value_counts().idxmax()
        if top_group in missing:
            color_map[top_group] = '#bbbbbb'
            missing = [val for val in missing if val != top_group]
    if missing:
        palette = plotly_colors.qualitative.Plotly
        used_colors = set(color_map.values())
        palette_cycle = cycle(palette)
        for val in sorted(missing, key=lambda v: str(v).lower()):
            color = next(palette_cycle)
            # Avoid reusing colors if possible; fall back if palette exhausted.
            attempts = 0
            while color in used_colors and attempts < len(palette):
                color = next(palette_cycle)
                attempts += 1
            color_map[val] = color
            used_colors.add(color)

    color_map = _cap_color_map_values(color_map, active_values)
    _COLOR_MAP_CACHE[group_col] = color_map.copy()
    if len(_COLOR_MAP_CACHE) > _COLOR_MAP_CACHE_MAX_GROUPS:
        _COLOR_MAP_CACHE.popitem(last=False)
    return color_map


def _clean_numeric(numeric_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = numeric_df.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if cleaned.isna().any().any():
        cleaned = cleaned.fillna(0)
    return cleaned


def prepare_metric_table(conn, wdir, metric):
    target_table = 'results'
    if metric == 'scalir_conc':
        scalir_path = Path(wdir) / "results" / "scalir" / "concentrations.csv"
        if not scalir_path.exists():
            return None
        try:
            conn.execute(
                f"CREATE OR REPLACE TEMP VIEW scalir_temp_conc AS SELECT * FROM read_csv_auto('{scalir_path}')"
            )
            conn.execute(
                """
                CREATE OR REPLACE TEMP VIEW scalir_results_view AS
                SELECT
                    r.ms_file_label,
                    r.peak_label,
                    s.pred_conc AS scalir_conc
                FROM results r
                LEFT JOIN scalir_temp_conc s
                ON r.ms_file_label = CAST(s.ms_file AS VARCHAR) AND r.peak_label = s.peak_label
                """
            )
            target_table = 'scalir_results_view'
        except Exception as exc:
            logger.error(f"Error preparing SCALiR data: {exc}")
            return None
    return _create_pivot_custom(conn, value=metric, table=target_table)


def normalize_matrices(df: pd.DataFrame, norm_value: str):
    df = _clean_numeric(df)
    if df.empty:
        return df, df
    scaler = StandardScaler()
    if norm_value == 'zscore':
        zdf = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
        return zdf, zdf
    if norm_value == 'durbin':
        ndf = _clean_numeric(rocke_durbin(df, c=10))
        return ndf, ndf
    if norm_value == 'zscore_durbin':
        z_tmp = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
        ndf = _clean_numeric(rocke_durbin(z_tmp, c=10))
        return ndf, ndf
    return df, df


def normalize_matrix(df: pd.DataFrame, norm_value: str):
    ndf, _ = normalize_matrices(df, norm_value)
    return ndf


def _create_pivot_custom(conn, value='peak_area', table='results'):
    """
    Local implementation of create_pivot to ensure table parameter is respected.
    """
    # Get ordered peak labels
    ordered_pl = [row[0] for row in conn.execute(f"""
        SELECT DISTINCT r.peak_label
        FROM {table} r
        JOIN targets t ON r.peak_label = t.peak_label
        ORDER BY t.ms_type
    """).fetchall()]

    group_cols_sql = ",\n                ".join([f"s.{col}" for col in GROUP_COLUMNS])

    query = f"""
        PIVOT (
            SELECT
                s.ms_type,
                s.sample_type,
                {group_cols_sql},
                r.ms_file_label,
                r.peak_label,
                r.{value}
            FROM {table} r
            JOIN samples s ON s.ms_file_label = r.ms_file_label
            WHERE s.use_for_analysis = TRUE
            ORDER BY s.ms_type, r.peak_label
        )
        ON peak_label
        USING FIRST({value})
        ORDER BY ms_type
    """
    df = conn.execute(query).df()
    if df.empty:
        return df
    meta_cols = ['ms_type', 'sample_type', *GROUP_COLUMNS, 'ms_file_label']
    # Start with metadata columns that exist in the dataframe
    final_cols = [col for col in meta_cols if col in df.columns]
    # Add pivot columns (targets) ONLY if they exist in the dataframe
    final_cols.extend([col for col in ordered_pl if col in df.columns])
    return df[final_cols]


def create_invisible_figure():
    """Create an invisible placeholder figure."""
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
        height=10,
    )
    return fig


def ensure_valid_group_field(group_field, *, allow_none=True, default=None):
    """Validate group field used in SQL identifier interpolation."""
    if group_field in (None, ''):
        if allow_none:
            return default
        raise PreventUpdate
    if group_field not in GROUPING_FIELDS:
        raise PreventUpdate
    return group_field


def get_analysis_read_limits() -> tuple[int, int]:
    """
    Return conservative DuckDB thread/memory limits for interactive Analysis reads.

    Mirrors the strategy used in optimization flows: reduce resources for UI-triggered
    queries to avoid CPU spikes from frequent callback-driven connections.
    """
    cpus, ram, _ = calculate_optimal_params()
    # Aggressive cap for interactive UI callbacks:
    # keep DB reads lightweight; rendering/math can still dominate CPU.
    cpus = max(1, cpus // 4)
    cpus = min(cpus, 2)
    ram = max(2, int(ram // 2))
    ram = min(ram, 6)
    return cpus, ram


@contextmanager
def analysis_read_connection(wdir, *, conn_factory=duckdb_connection):
    """Open a read-only DuckDB connection with Analysis-safe resource limits."""
    global _ANALYSIS_DB_LIMITS_LOGGED
    cpus, ram = get_analysis_read_limits()
    with conn_factory(wdir, n_cpus=cpus, ram=ram, read_only=True) as conn:
        if conn is not None and not _ANALYSIS_DB_LIMITS_LOGGED:
            try:
                actual_threads = conn.execute("SELECT current_setting('threads')").fetchone()[0]
                actual_mem = conn.execute("SELECT current_setting('memory_limit')").fetchone()[0]
                logger.info(
                    "Analysis DB read limits active: requested threads=%s, ram=%sGB; effective threads=%s, memory_limit=%s",
                    cpus,
                    ram,
                    actual_threads,
                    actual_mem,
                )
                _ANALYSIS_DB_LIMITS_LOGGED = True
            except Exception:
                pass
        yield conn
