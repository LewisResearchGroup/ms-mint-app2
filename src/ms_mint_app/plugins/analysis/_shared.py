"""Shared constants, imports, and utilities for Analysis tabs."""

import logging
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

from ...duckdb_manager import duckdb_connection, create_pivot, get_physical_cores
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
    'raincloud': 'durbin',
    'bar': 'durbin',
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


def _build_color_map(color_df: pd.DataFrame, group_col: str) -> dict:
    if not group_col or group_col not in color_df.columns or color_df.empty:
        return {}
    working = color_df[[group_col, 'color']].copy()
    working = working[working[group_col].notna()]
    working['color'] = working['color'].apply(
        lambda c: c if isinstance(c, str) and c.strip() and c.strip() != '#bbbbbb' else None
    )
    color_map = (
        working.dropna(subset=['color'])
        .drop_duplicates(subset=[group_col])
        .set_index(group_col)['color']
        .to_dict()
    )
    missing = [val for val in working[group_col].dropna().unique() if val not in color_map]
    if missing:
        palette = plotly_colors.qualitative.Plotly
        for val, color in zip(missing, cycle(palette)):
            color_map[val] = color
    return color_map


def _clean_numeric(numeric_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = numeric_df.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if cleaned.isna().any().any():
        cleaned = cleaned.fillna(0)
    return cleaned


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
