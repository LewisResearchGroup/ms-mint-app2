"""Analysis plugin orchestrator."""

import base64
from io import BytesIO, StringIO
import logging
from dataclasses import dataclass
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
import feffery_antd_components as fac
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from ...plugin_interface import PluginInterface
from ... import tools as T
from ...duckdb_manager import duckdb_connection, get_workspace_name_from_wdir
from ._shared import (
    METRIC_OPTIONS, NORM_OPTIONS, GROUP_SELECT_OPTIONS, TAB_DEFAULT_NORM,
    GROUPING_FIELDS, GROUP_LABELS, OPTIONAL_METRICS, allowed_metrics,
    _build_color_map,
    analysis_read_connection, check_optional_metric_availability, create_invisible_figure, get_download_config
)
from .data_pipeline import (
    _no_update_outputs as _no_update_outputs_full,
    _prepare_matrix_data,
    _return_bar,
    _return_clustermap,
    _return_pca,
    _return_tsne,
    _return_violin,
)
from . import qc, pca, tsne, violin, bar, clustermap, feature_comparison

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisUpdateContext:
    section_context: dict | None
    tab_key: str | None
    x_comp: str | None
    y_comp: str | None
    violin_comp_checks: list | None
    bar_comp_checks: list | None
    metric_value: str | None
    norm_value: str | None
    group_by: str | None
    regen_clicks: int | None
    tsne_regen_clicks: int | None
    cluster_rows: bool | None
    cluster_cols: bool | None
    fontsize_x: int | None
    fontsize_y: int | None
    wdir: str | None
    tsne_x_comp: str | None
    tsne_y_comp: str | None
    tsne_perplexity: int | None
    pca_cache: dict | None
    tsne_cache: dict | None
    violin_cache: dict | None
    bar_cache: dict | None = None
    pca_visible_groups: dict | None = None

# Analysis menu items for sidebar
ANALYSIS_MENU_ITEMS = [
    {'component': 'Item', 'props': {'key': 'pca', 'title': 'PCA', 'icon': 'antd-dot-chart'}},
    {'component': 'Item', 'props': {'key': 'tsne', 'title': 't-SNE', 'icon': 'antd-deployment-unit'}},
    {'component': 'Item', 'props': {'key': 'qc', 'title': 'QC', 'icon': 'antd-check-circle'}},
    {'component': 'Item', 'props': {'key': 'raincloud', 'title': 'Violin', 'icon': 'antd-control'}},
    {'component': 'Item', 'props': {'key': 'bar', 'title': 'Bar', 'icon': 'antd-bar-chart'}},
    {'component': 'Item', 'props': {'key': 'comparison', 'title': 'Comparison', 'icon': 'antd-swap'}},
    {'component': 'Item', 'props': {'key': 'clustermap', 'title': 'Clustermap', 'icon': 'antd-build'}},
]

_layout = fac.AntdLayout(
    [
        fac.AntdHeader(
            [
                fac.AntdFlex(
                    [
                        fac.AntdTitle(
                            'Analysis', level=4, style={'margin': '0', 'whiteSpace': 'nowrap'}
                        ),
                        fac.AntdTooltip(
                            fac.AntdIcon(
                                id='analysis-tour-icon',
                                icon='pi-info',
                                style={"cursor": "pointer", 'paddingLeft': '10px'},
                                **{'aria-label': 'Show tutorial'},
                            ),
                            title='Show tutorial'
                        ),
                    ],
                    align='center',
                ),
            ],
            style={'background': 'white', 'padding': '0px', 'lineHeight': '32px', 'height': '32px'}
        ),
        fac.AntdLayout(
            [
                fac.AntdSider(
                    [
                        fac.AntdFlex(
                            [
                                fac.AntdMenu(
                                    id='analysis-sidebar-menu',
                                    menuItems=ANALYSIS_MENU_ITEMS,
                                    mode='inline',
                                    defaultSelectedKey='pca',
                                    style={'border': 'none'},
                                ),
                            ],
                            vertical=True,
                            style={'height': '100%', 'paddingTop': '8px'}
                        ),
                        html.Div(
                            fac.AntdTooltip(
                                fac.AntdButton(
                                    id='analysis-sidebar-collapse',
                                    type='text',
                                    icon=fac.AntdIcon(
                                        id='analysis-sidebar-collapse-icon',
                                        icon='antd-right',
                                        style={'fontSize': '14px'},
                                    ),
                                    shape='default',
                                    **{'aria-label': 'Collapse/Expand Analysis Sidebar'},
                                ),
                                title='Collapse/Expand Analysis Sidebar'
                            ),
                            style={
                                'position': 'absolute',
                                'zIndex': 1,
                                'right': -8,
                                'bottom': 16,
                                'boxShadow': '2px 2px 5px 1px rgba(0,0,0,0.5)',
                                'background': 'white',
                                'borderRadius': '4px',
                            },
                        ),
                    ],
                    id='analysis-sidebar',
                    collapsible=True,
                    collapsed=True,
                    collapsedWidth=60,
                    width=180,
                    trigger=None,
                    style={'height': '100%', 'background': 'white'},
                    className="sidebar-mint"
                ),
                html.Div(
                    [
                        # Shared controls header - visible for all views
                        fac.AntdFlex(
                            [
                                html.Div(
                                    fac.AntdSpace(
                                        [
                                            fac.AntdText("Metric:", style={'fontWeight': 500, 'marginLeft': '15px'}),
                                            fac.AntdSelect(
                                                id='analysis-metric-select',
                                                options=METRIC_OPTIONS,
                                                value='peak_area',
                                                optionFilterProp='label',
                                                optionFilterMode='case-insensitive',
                                                allowClear=False,
                                                style={'width': 160},
                                            ),
                                        ],
                                        align='center',
                                        size='small',
                                    ),
                                    id='analysis-metric-wrapper',
                                    style={'display': 'flex'},
                                ),
                                html.Div(
                                    fac.AntdSpace(
                                        [
                                            fac.AntdText("Transformations:", style={'fontWeight': 500}),
                                            fac.AntdSelect(
                                                id='analysis-normalization-select',
                                                options=NORM_OPTIONS,
                                                value='durbin',
                                                optionFilterProp='label',
                                                optionFilterMode='case-insensitive',
                                                allowClear=False,
                                                style={'width': 160},
                                            ),
                                        ],
                                        align='center',
                                        size='small',
                                    ),
                                    id='analysis-normalization-wrapper',
                                    style={'display': 'flex'},
                                ),
                                fac.AntdSpace(
                                    [
                                        fac.AntdText("Group by:", style={'fontWeight': 500}),
                                        fac.AntdSelect(
                                            id='analysis-grouping-select',
                                            options=GROUP_SELECT_OPTIONS,
                                            value='sample_type',
                                            allowClear=False,
                                            style={'width': 160},
                                        ),
                                    ],
                                    align='center',
                                    size='small',
                                ),
                                html.Div(
                                    fac.AntdSpace(
                                        [
                                            fac.AntdText("Target:", style={'fontWeight': 500}),
                                            fac.AntdSelect(
                                                id='qc-target-select',
                                                options=[],
                                                value=None,
                                                allowClear=False,
                                                optionFilterProp='label',
                                                optionFilterMode='case-insensitive',
                                                style={'width': 350},
                                            ),
                                        ],
                                        align='center',
                                        size='small',
                                    ),
                                    id='qc-target-wrapper',
                                    style={'display': 'flex'},
                                ),
                            ],
                            wrap=True,
                            gap='middle',
                            align='center',
                            id='analysis-metric-container',
                            style={'padding': '12px 16px', 'borderBottom': '1px solid #f0f0f0'},
                        ),
                        # QC content
                        html.Div(
                            qc.create_layout(),
                            id='analysis-qc-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # PCA content
                        html.Div(
                            pca.create_layout(),
                            id='analysis-pca-container',
                            style={'display': 'block', 'padding': '16px'}
                        ),
                        # t-SNE content
                        html.Div(
                            tsne.create_layout(),
                            id='analysis-tsne-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # Violin content
                        html.Div(
                            violin.create_layout(),
                            id='analysis-violin-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # Bar content
                        html.Div(
                            bar.create_layout(),
                            id='analysis-bar-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # Comparison content
                        html.Div(
                            feature_comparison.create_layout(),
                            id='analysis-comparison-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # Clustermap content
                        html.Div(
                            clustermap.create_layout(),
                            id='analysis-clustermap-container',
                            style={'display': 'none', 'padding': '16px', 'height': 'calc(100vh - 150px)'}
                        ),
                    ],
                    className='ant-layout-content css-1v28nim',
                    style={'background': 'white', 'overflowY': 'auto', 'height': 'calc(100vh - 80px)'}
                ),
            ],
            style={'height': 'calc(100vh - 100px)', 'background': 'white'},
        ),
        # Hidden store for maintaining tab state compatibility with callbacks
        dcc.Store(id='analysis-tabs', data={'activeKey': 'pca'}),
        dcc.Store(
            id='analysis-shared-settings',
            data={'metric': 'peak_area', 'group_by': 'sample_type'},
        ),
        dcc.Store(id='analysis-pca-cache', data=None),
        dcc.Store(id='analysis-pca-visible-groups', data=None),
        dcc.Store(id='analysis-tsne-cache', data=None),
        dcc.Store(id='analysis-violin-cache', data=None),
        dcc.Store(id='analysis-bar-cache', data=None),
        fac.AntdTour(
            locale='en-us',
            steps=[],
            id='analysis-tour',
            open=False,
            current=0,
        ),
        html.Div(id="analysis-notifications-container"),
    ]
)

class AnalysisPlugin(PluginInterface):
    def __init__(self):
        self._layout = _layout

    def layout(self):
        return self._layout

    def callbacks(self, app, fsc=None, cache=None):
        callbacks(app, fsc, cache)

    def outputs(self):
        return None

def layout():
    return _layout


def _no_update_outputs():
    # Backward-compatible helper for direct calls/tests.
    return _no_update_outputs_full()[:-1]

def _normalize_pca_visible_groups(pca_visible_groups, selected_group):
    if not isinstance(pca_visible_groups, dict):
        return None
    if pca_visible_groups.get('group_by') != selected_group:
        return None
    groups = pca_visible_groups.get('visible_groups')
    if not isinstance(groups, list):
        return None
    normalized = [str(group) for group in groups if group is not None and str(group) != ""]
    return list(dict.fromkeys(normalized))


def _pca_cache_key(wdir, metric, norm_value, selected_group=None, visible_groups=None):
    group_token = 'ALL'
    if visible_groups is not None:
        if not visible_groups:
            group_token = 'EMPTY'
        else:
            group_token = ','.join(sorted(str(group) for group in visible_groups))
    return f"{wdir}|{metric}|{norm_value}|{selected_group}|{group_token}"

def _tsne_cache_key(wdir, metric, norm_value, selected_group, perplexity):
    return f"{wdir}|{metric}|{norm_value}|{selected_group}|{perplexity}"

def _violin_cache_key(wdir, metric, norm_value, selected_group):
    return f"{wdir}|{metric}|{norm_value}|{selected_group}"

def _bar_cache_key(wdir, metric, norm_value, selected_group):
    return f"{wdir}|{metric}|{norm_value}|{selected_group}"


def _resolve_metric_options_for_workspace(wdir):
    """Return metric select options with unavailable metrics disabled for this workspace."""
    options = [dict(opt) for opt in METRIC_OPTIONS]
    unavailable = set()

    if not wdir:
        unavailable.update(OPTIONAL_METRICS)
    else:
        with analysis_read_connection(wdir) as conn:
            if conn is None:
                unavailable.update(OPTIONAL_METRICS)
            else:
                for opt_metric in OPTIONAL_METRICS:
                    is_available, _ = check_optional_metric_availability(conn, wdir, opt_metric)
                    if not is_available:
                        unavailable.add(opt_metric)

    for opt in options:
        if opt.get("value") in unavailable:
            opt["disabled"] = True

    available_metrics = [opt.get("value") for opt in options if not opt.get("disabled")]
    return options, available_metrics


def _serialize_pca_results(results):
    loadings = results.get('loadings')
    return {
        'scores': results['scores'].to_dict(orient='split'),
        'loadings': loadings.to_dict(orient='split') if isinstance(loadings, pd.DataFrame) else None,
        'explained_variance_ratio': {
            'index': list(results['explained_variance_ratio'].index),
            'data': results['explained_variance_ratio'].tolist(),
        },
        'cumulative_variance_ratio': {
            'index': list(results['cumulative_variance_ratio'].index),
            'data': results['cumulative_variance_ratio'].tolist(),
        },
    }


def _deserialize_pca_results(payload):
    scores = payload.get('scores') or {}
    scores_df = pd.DataFrame(scores.get('data', []), columns=scores.get('columns', []), index=scores.get('index', []))
    loadings = payload.get('loadings')
    loadings_df = None
    if loadings:
        loadings_df = pd.DataFrame(loadings.get('data', []), columns=loadings.get('columns', []), index=loadings.get('index', []))
    explained = payload.get('explained_variance_ratio') or {}
    cumulative = payload.get('cumulative_variance_ratio') or {}
    return {
        'scores': scores_df,
        'loadings': loadings_df,
        'explained_variance_ratio': pd.Series(explained.get('data', []), index=explained.get('index', [])),
        'cumulative_variance_ratio': pd.Series(cumulative.get('data', []), index=cumulative.get('index', [])),
    }

def _serialize_tsne_results(scores_df):
    return {
        'scores': scores_df.to_dict(orient='split'),
    }


def _deserialize_tsne_results(payload):
    scores = payload.get('scores') or {}
    scores_df = pd.DataFrame(scores.get('data', []), columns=scores.get('columns', []), index=scores.get('index', []))
    return scores_df

def _serialize_violin_series(series_df):
    return {
        'series': series_df.to_dict(orient='split'),
        'index_name': series_df.index.name or 'ms_file_label',
    }


def _deserialize_violin_series(payload):
    series = payload.get('series') or {}
    series_df = pd.DataFrame(series.get('data', []), columns=series.get('columns', []), index=series.get('index', []))
    series_df.index.name = payload.get('index_name') or 'ms_file_label'
    return series_df


def _serialize_bar_series(series_df):
    return {
        'series': series_df.to_dict(orient='split'),
        'index_name': series_df.index.name or 'ms_file_label',
    }


def _deserialize_bar_series(payload):
    series = payload.get('series') or {}
    series_df = pd.DataFrame(series.get('data', []), columns=series.get('columns', []), index=series.get('index', []))
    series_df.index.name = payload.get('index_name') or 'ms_file_label'
    return series_df


def _normalize_compound_selection(value):
    """Normalize legacy list-valued selectors to a single compound value."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _build_group_context_from_cache(samples_meta_records, selected_group, target_index):
    """Build grouping series/color map from cached sample metadata."""
    samples_meta = pd.DataFrame(samples_meta_records or [])
    if not samples_meta.empty and 'ms_file_label' in samples_meta.columns:
        samples_meta = samples_meta.set_index('ms_file_label')

    group_field = selected_group if selected_group in samples_meta.columns else (
        'sample_type' if 'sample_type' in samples_meta.columns else None
    )
    group_label = GROUP_LABELS.get(group_field, 'Group')
    missing_group_label = f"{group_label} (unset)"

    if group_field and group_field in samples_meta.columns:
        group_series = samples_meta[group_field].reindex(target_index)
    else:
        group_series = pd.Series(target_index, index=target_index, name='group')

    if isinstance(group_series, pd.Series):
        group_series = group_series.replace("", pd.NA)
    group_series.name = group_field or 'group'
    group_series = group_series.fillna(missing_group_label)

    if samples_meta.empty:
        color_map = {}
    else:
        color_source = samples_meta.reset_index()
        color_map = _build_color_map(
            color_source, group_field, use_sample_colors=(group_field == 'sample_type')
        )
    if missing_group_label in group_series.values:
        color_map.setdefault(missing_group_label, '#bbbbbb')

    return group_series, color_map, group_label


def _extract_all_groups_from_samples_meta(samples_meta_records, selected_group, fallback_index):
    samples_meta = pd.DataFrame(samples_meta_records or [])
    if samples_meta.empty:
        return list(dict.fromkeys(fallback_index))

    group_field = selected_group if selected_group in samples_meta.columns else (
        'sample_type' if 'sample_type' in samples_meta.columns else None
    )
    if not group_field or group_field not in samples_meta.columns:
        return list(dict.fromkeys(fallback_index))

    group_label = GROUP_LABELS.get(group_field, 'Group')
    missing_group_label = f"{group_label} (unset)"
    values = (
        samples_meta[group_field]
        .replace("", pd.NA)
        .fillna(missing_group_label)
        .tolist()
    )
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def _handle_pca_cached_group_change(
    tab_key, triggered_only_group, pca_cache, cache_key, selected_group, x_comp, y_comp, visible_groups
):
    if not (tab_key == 'pca' and triggered_only_group and pca_cache and pca_cache.get('key') == cache_key):
        return None
    try:
        results = _deserialize_pca_results(pca_cache.get('results', {}))
        group_series, color_map, group_label = _build_group_context_from_cache(
            pca_cache.get('samples_meta', []),
            selected_group,
            results['scores'].index,
        )
        all_groups = _extract_all_groups_from_samples_meta(
            pca_cache.get('samples_meta', []),
            selected_group,
            results['scores'].index,
        )
        fig = pca.generate_pca_figure(
            None, group_series, color_map, group_label, x_comp, y_comp,
            pca_results=results, all_groups=all_groups, visible_groups=visible_groups
        )
        return _return_pca(fig)
    except Exception as exc:
        logger.warning("PCA cache fast-path failed (group=%s): %s", selected_group, exc)
        return None


def _handle_pca_cached_render(
    tab_key, triggered_props, pca_cache, cache_key, selected_group, x_comp, y_comp, visible_groups
):
    """
    Re-render PCA directly from cached PCA results when data matrix inputs are unchanged.

    This covers common UI paths like:
    - switching away from PCA and coming back
    - changing PCA axis selectors (x/y)
    """
    if not (tab_key == 'pca' and pca_cache and pca_cache.get('key') == cache_key):
        return None

    cache_safe_triggers = {
        'analysis-sidebar-menu.currentKey',
        'pca-x-comp.value',
        'pca-y-comp.value',
        'section-context.data',
        'analysis-metric-select.value',
        'analysis-normalization-select.value',
        'analysis-grouping-select.value',
        'wdir.data',
    }
    if triggered_props and not all(prop in cache_safe_triggers for prop in triggered_props):
        return None

    try:
        results = _deserialize_pca_results(pca_cache.get('results', {}))
        if results['scores'].empty:
            return None
        group_series, color_map, group_label = _build_group_context_from_cache(
            pca_cache.get('samples_meta', []),
            selected_group,
            results['scores'].index,
        )
        all_groups = _extract_all_groups_from_samples_meta(
            pca_cache.get('samples_meta', []),
            selected_group,
            results['scores'].index,
        )
        fig = pca.generate_pca_figure(
            None, group_series, color_map, group_label, x_comp, y_comp,
            pca_results=results, all_groups=all_groups, visible_groups=visible_groups
        )
        logger.debug(
            "PCA cache hit for key=%s (triggers=%s)",
            cache_key,
            ",".join(triggered_props) if triggered_props else "none",
        )
        return _return_pca(fig)
    except Exception as exc:
        logger.warning("PCA cache render fast-path failed: %s", exc)
        return None


def _handle_tsne_cached_group_change(
    tab_key, triggered_only_group, tsne_cache, tsne_cache_key, selected_group,
    tsne_x_comp, tsne_y_comp, tsne_perplexity
):
    if not (tab_key == 'tsne' and triggered_only_group and tsne_cache and tsne_cache.get('key') == tsne_cache_key):
        return None
    try:
        scores_df = _deserialize_tsne_results(tsne_cache.get('results', {}))
        if scores_df.empty:
            raise ValueError("Empty t-SNE cache")
        group_series, color_map, group_label = _build_group_context_from_cache(
            tsne_cache.get('samples_meta', []),
            selected_group,
            scores_df.index,
        )
        fig = tsne.generate_tsne_figure(
            None, group_series, color_map, group_label, tsne_x_comp, tsne_y_comp, tsne_perplexity,
            tsne_scores=scores_df
        )
        return _return_tsne(fig)
    except Exception as exc:
        logger.warning("t-SNE cache fast-path failed (group=%s): %s", selected_group, exc)
        return None


def _handle_violin_cached_group_change(
    tab_key, triggered_only_group, violin_cache, violin_cache_key, selected_group,
    metric, norm_value, wdir
):
    if not (tab_key == 'raincloud' and triggered_only_group and violin_cache and violin_cache.get('key') == violin_cache_key):
        return None
    try:
        series_df = _deserialize_violin_series(violin_cache.get('results', {}))
        if series_df.empty:
            raise ValueError("Empty violin cache")
        group_series, color_map, group_label = _build_group_context_from_cache(
            violin_cache.get('samples_meta', []),
            selected_group,
            series_df.index,
        )
        selected_compound = violin_cache.get('selected_compound')
        violin_options = violin_cache.get('options', [])
        graphs = violin._build_violin_graphs(
            series_df, group_series, color_map, group_label, metric, norm_value, selected_compound,
            filename=f"{T.today()}-MINT__{get_workspace_name_from_wdir(wdir)}-Analysis-Violin"
        )
        return _return_violin(graphs, violin_options, selected_compound)
    except Exception as exc:
        logger.warning("Violin cache fast-path failed (group=%s): %s", selected_group, exc)
        return None


def _handle_violin_cached_render(
    tab_key, triggered_props, violin_cache, violin_cache_key, selected_group, metric, norm_value, wdir,
    current_violin_value=None
):
    if not (tab_key == 'raincloud' and violin_cache and violin_cache.get('key') == violin_cache_key):
        return None
    cache_safe_triggers = {
        'analysis-sidebar-menu.currentKey',
        'violin-comp-checks.value',
        'analysis-grouping-select.value',
        'analysis-metric-select.value',
        'analysis-normalization-select.value',
        'section-context.data',
        'wdir.data',
    }
    if triggered_props and not all(prop in cache_safe_triggers for prop in triggered_props):
        return None
    if 'violin-comp-checks.value' in (triggered_props or []):
        cached_value = _normalize_compound_selection(violin_cache.get('selected_compound'))
        requested_value = _normalize_compound_selection(current_violin_value)
        if requested_value and requested_value != cached_value:
            # Cached payload contains only one compound series, so selector changes need a full recompute.
            return None
    try:
        series_df = _deserialize_violin_series(violin_cache.get('results', {}))
        if series_df.empty:
            return None
        group_series, color_map, group_label = _build_group_context_from_cache(
            violin_cache.get('samples_meta', []),
            selected_group,
            series_df.index,
        )
        selected_compound = violin_cache.get('selected_compound')
        violin_options = violin_cache.get('options', [])
        graphs = violin._build_violin_graphs(
            series_df, group_series, color_map, group_label, metric, norm_value, selected_compound,
            filename=f"{T.today()}-MINT__{get_workspace_name_from_wdir(wdir)}-Analysis-Violin"
        )
        return _return_violin(graphs, violin_options, selected_compound)
    except Exception as exc:
        logger.warning("Violin cache render fast-path failed: %s", exc)
        return None


def _handle_bar_cached_group_change(
    tab_key, triggered_only_group, bar_cache, bar_cache_key, selected_group,
    metric, norm_value, wdir
):
    if not (tab_key == 'bar' and triggered_only_group and bar_cache and bar_cache.get('key') == bar_cache_key):
        return None
    try:
        series_df = _deserialize_bar_series(bar_cache.get('results', {}))
        if series_df.empty:
            raise ValueError("Empty bar cache")
        group_series, color_map, group_label = _build_group_context_from_cache(
            bar_cache.get('samples_meta', []),
            selected_group,
            series_df.index,
        )
        selected_compound = bar_cache.get('selected_compound')
        bar_options = bar_cache.get('options', [])
        graphs, options, value = bar.generate_bar_plots(
            series_df, group_series, color_map, group_label, metric, norm_value,
            selected_compound, bar_options,
            filename=f"{T.today()}-MINT__{get_workspace_name_from_wdir(wdir)}-Analysis-Bar"
        )
        return _return_bar(graphs, options, value)
    except Exception as exc:
        logger.warning("Bar cache fast-path failed (group=%s): %s", selected_group, exc)
        return None


def _handle_bar_cached_render(
    tab_key, triggered_props, bar_cache, bar_cache_key, selected_group, metric, norm_value, wdir,
    current_bar_value=None
):
    if not (tab_key == 'bar' and bar_cache and bar_cache.get('key') == bar_cache_key):
        return None
    cache_safe_triggers = {
        'analysis-sidebar-menu.currentKey',
        'bar-comp-checks.value',
        'analysis-grouping-select.value',
        'analysis-metric-select.value',
        'analysis-normalization-select.value',
        'section-context.data',
        'wdir.data',
    }
    if triggered_props and not all(prop in cache_safe_triggers for prop in triggered_props):
        return None
    if 'bar-comp-checks.value' in (triggered_props or []):
        cached_value = _normalize_compound_selection(bar_cache.get('selected_compound'))
        requested_value = _normalize_compound_selection(current_bar_value)
        if requested_value and requested_value != cached_value:
            # Cached payload contains only one compound series, so selector changes need a full recompute.
            return None
    try:
        series_df = _deserialize_bar_series(bar_cache.get('results', {}))
        if series_df.empty:
            return None
        group_series, color_map, group_label = _build_group_context_from_cache(
            bar_cache.get('samples_meta', []),
            selected_group,
            series_df.index,
        )
        selected_compound = bar_cache.get('selected_compound')
        bar_options = bar_cache.get('options', [])
        graphs, options, value = bar.generate_bar_plots(
            series_df, group_series, color_map, group_label, metric, norm_value,
            selected_compound, bar_options,
            filename=f"{T.today()}-MINT__{get_workspace_name_from_wdir(wdir)}-Analysis-Bar"
        )
        return _return_bar(graphs, options, value)
    except Exception as exc:
        logger.warning("Bar cache render fast-path failed: %s", exc)
        return None


def _handle_clustermap_tab(
    triggered_props, zdf, color_labels, color_map, group_label, norm_value,
    cluster_rows, cluster_cols, fontsize_x, fontsize_y, wdir, metric
):
    triggered_prop = triggered_props[0].split('.')[0] if triggered_props else None
    src = clustermap.generate_clustermap(
        zdf, color_labels, color_map, group_label, norm_value, cluster_rows, cluster_cols,
        fontsize_x, fontsize_y, wdir, metric, triggered_prop, norm_value
    )
    return _return_clustermap(src)


def _handle_pca_tab(
    ndf, color_labels, color_map, group_label, x_comp, y_comp, cache_key, colors_df,
    compound_options, all_groups, visible_groups
):
    pca_results = pca.run_pca_samples_in_cols(ndf, n_components=min(ndf.shape[0], ndf.shape[1], 5))
    fig = pca.generate_pca_figure(
        ndf, color_labels, color_map, group_label, x_comp, y_comp,
        pca_results=pca_results, all_groups=all_groups, visible_groups=visible_groups
    )
    pca_cache_data = {
        'key': cache_key,
        'results': _serialize_pca_results(pca_results),
        'samples_meta': colors_df.to_dict(orient='records') if not colors_df.empty else [],
    }
    return _return_pca(fig, compound_options=compound_options, pca_cache_data=pca_cache_data)


def _handle_tsne_tab(
    ndf, color_labels, color_map, group_label, tsne_x_comp, tsne_y_comp, tsne_perplexity,
    tsne_cache_key, colors_df
):
    tsne_scores = tsne.run_tsne_samples_in_cols(ndf, tsne_perplexity)
    fig = tsne.generate_tsne_figure(
        ndf, color_labels, color_map, group_label, tsne_x_comp, tsne_y_comp, tsne_perplexity,
        tsne_scores=tsne_scores
    )
    tsne_cache_data = {
        'key': tsne_cache_key,
        'results': _serialize_tsne_results(tsne_scores) if tsne_scores is not None else {},
        'samples_meta': colors_df.to_dict(orient='records') if not colors_df.empty else [],
    } if tsne_scores is not None else None
    return _return_tsne(fig, tsne_cache_data=tsne_cache_data)


def _handle_raincloud_tab(
    violin_matrix, group_series, color_map, group_label, metric, norm_value,
    violin_comp_checks, compound_options, base_name, violin_cache_key, colors_df
):
    file_name = f"{base_name}-Violin"
    graphs, options, val = violin.generate_violin_plots(
        violin_matrix, group_series, color_map, group_label, metric, norm_value,
        violin_comp_checks, compound_options, filename=file_name
    )
    violin_cache_data = None
    if val and val in violin_matrix.columns:
        series_df = violin_matrix[[val]].copy()
        violin_cache_data = {
            'key': violin_cache_key,
            'results': _serialize_violin_series(series_df),
            'selected_compound': val,
            'options': options,
            'samples_meta': colors_df.to_dict(orient='records') if not colors_df.empty else [],
        }
    return _return_violin(graphs, options, val, violin_cache_data=violin_cache_data)


def _handle_bar_tab(
    violin_matrix, group_series, color_map, group_label, metric, norm_value,
    bar_comp_checks, compound_options, base_name, bar_cache_key, colors_df
):
    file_name = f"{base_name}-Bar"
    graphs, options, val = bar.generate_bar_plots(
        violin_matrix, group_series, color_map, group_label, metric, norm_value,
        bar_comp_checks, compound_options, filename=file_name
    )
    bar_cache_data = None
    if val and val in violin_matrix.columns:
        series_df = violin_matrix[[val]].copy()
        bar_cache_data = {
            'key': bar_cache_key,
            'results': _serialize_bar_series(series_df),
            'selected_compound': val,
            'options': options,
            'samples_meta': colors_df.to_dict(orient='records') if not colors_df.empty else [],
        }
    return _return_bar(graphs, options, val, bar_cache_data=bar_cache_data)


def _analysis_tour_steps(active_tab: str):
    # Common steps for all views
    common_steps = [
        {
            'title': 'Analysis overview',
            'description': 'Use the controls above to choose a metric, transformation, and grouping.',
        },
        {
            'title': 'Metric',
            'description': 'Pick the quantitative metric to visualize.',
            'targetSelector': '#analysis-metric-select',
        },
        {
            'title': 'Transformations',
            'description': 'Normalize or transform the values before plotting.',
            'targetSelector': '#analysis-normalization-select',
        },
        {
            'title': 'Group by',
            'description': 'Group samples for coloring and summaries.',
            'targetSelector': '#analysis-grouping-select',
        },
        {
            'title': 'Analysis Types',
            'description': 'Switch between PCA, Violin, and Clustermap views.',
            'targetSelector': '#analysis-sidebar-menu',
        },
    ]

    # QC-specific override for common steps
    if active_tab == 'qc':
        common_steps = [
            {
                'title': 'Analysis overview',
                'description': 'Use the controls above to group samples and select targets for QC.',
            },
            {
                'title': 'Group by',
                'description': 'Group samples for coloring in the plots.',
                'targetSelector': '#analysis-grouping-select',
            },
            {
                'title': 'Select Target',
                'description': 'Choose the specific target compound to visualize in the QC plots.',
                'targetSelector': '[id="qc-target-select"]', # Use attribute selector for component IDs
            },
            {
                'title': 'Analysis Types',
                'description': 'Switch between different analysis views.',
                'targetSelector': '#analysis-sidebar-menu',
            },
        ]
    
    # View-specific steps
    if active_tab == 'qc':
        view_steps = [
            {
                'title': 'QC Scatter Plots',
                'description': 'View Retention Time (RT) and Peak Area drifts over acquisition order. Useful for identifying run order effects.',
                'targetSelector': '#qc-spinner',
            },
            {
                'title': 'Chromatogram',
                'description': 'See the raw chromatogram for the selected sample. Click on a point in the scatter plots to select a sample.',
                'targetSelector': '#qc-chromatogram-container',
            },
        ]
    elif active_tab == 'pca':
        view_steps = [
            {
                'title': 'PCA Axes',
                'description': 'Select which principal components to display on the X and Y axes.',
                'targetSelector': '#pca-x-comp',
            },
            {
                'title': 'PCA Plot',
                'description': 'Interactive scatter plot showing samples in PCA space. Hover for details, use the legend to filter groups.',
                'targetSelector': '#pca-graph',
            },

        ]
    elif active_tab == 'tsne':
        view_steps = [
            {
                'title': 't-SNE Axes',
                'description': 'Select which t-SNE dimensions to display.',
                'targetSelector': '#tsne-x-comp',
            },
            {
                'title': 't-SNE Plot',
                'description': 'Interactive scatter plot showing samples in t-SNE space.',
                'targetSelector': '#tsne-graph',
            },
        ]
    elif active_tab == 'raincloud':
        view_steps = [
            {
                'title': 'Target Selection',
                'description': 'Choose which target compound to display in the violin plot.',
                'targetSelector': '#violin-comp-checks',
            },
            {
                'title': 'Violin Plot',
                'description': 'Shows the distribution of values for each group. Click on individual points to see the chromatogram.',
                'targetSelector': '#violin-graphs',
            },
            {
                'title': 'Chromatogram',
                'description': 'View the chromatogram for the selected sample.',
                'targetSelector': '#violin-chromatogram-container',
            },
        ]
    elif active_tab == 'bar':
        view_steps = [
            {
                'title': 'Target Selection',
                'description': 'Choose which target compound to display in the bar plot.',
                'targetSelector': '#bar-comp-checks',
            },
            {
                'title': 'Bar Plot',
                'description': 'Compare mean values across groups with error bars. Click on a bar to select a sample from that group.',
                'targetSelector': '#bar-graphs',
            },
            {
                'title': 'Chromatogram',
                'description': 'View the chromatogram for the selected sample.',
                'targetSelector': '#bar-chromatogram-container',
            },
        ]
    elif active_tab == 'comparison':
        view_steps = [
            {
                'title': 'Sample Selection',
                'description': 'Choose two groups to compare and set the significance threshold.',
                'targetSelector': '#comparison-sample1-select',
            },
            {
                'title': 'Comparison Plot',
                'description': 'Scatter or Volcano plot highlighting significant features. Click a point to view chromatograms.',
                'targetSelector': '[id="{\\"type\\":\\"comparison-plot\\",\\"index\\":\\"main\\"}"]',
            },
            {
                'title': 'Chromatogram',
                'description': 'Compare chromatograms for the selected feature across both groups.',
                'targetSelector': '#comparison-chromatogram-container',
            },
        ]
    elif active_tab == 'clustermap':
        view_steps = [
            {
                'title': 'Font Size Controls',
                'description': 'Adjust the font size for row and column labels.',
                'targetSelector': '#clustermap-fontsize-x-slider',
            },
            {
                'title': 'Regenerate',
                'description': 'Click to regenerate the clustermap with current settings.',
                'targetSelector': '#clustermap-regenerate-btn',
            },
            {
                'title': 'Clustermap',
                'description': 'Heatmap with hierarchical clustering of samples and targets.',
                'targetSelector': '#clustermap-image',
            },
        ]
    else:
        view_steps = []
    
    return common_steps + view_steps


def callbacks(app, fsc=None, cache=None):

    # Register tab-specific callbacks
    qc.register_callbacks(app)
    pca.register_callbacks(app)
    tsne.register_callbacks(app)
    violin.register_callbacks(app)
    bar.register_callbacks(app)
    feature_comparison.register_callbacks(app)
    clustermap.register_callbacks(app)

    @app.callback(
        Output("analysis-notifications-container", "children"),
        Input('section-context', 'data'),
        Input("wdir", "data"),
        Input('analysis-metric-select', 'value'),
        prevent_initial_call=True,
    )
    def warn_missing_workspace(section_context, wdir, metric_value):
        if not section_context or section_context.get('page') != 'Analysis':
            return dash.no_update
        if not wdir:
            return fac.AntdNotification(
                message="Activate a workspace",
                description="Please select or create a workspace first.",
                type="warning",
                duration=4,
                placement='bottom',
                showProgress=True,
                stack=True,
            )
        
        # Check if results table is empty
        # Read-only optimization
        with analysis_read_connection(wdir) as conn:
            if conn is None:
                return []
            try:
                results_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
                if results_count == 0:
                    return fac.AntdNotification(
                        message="No data available",
                        description="Please run MINT processing first by navigating to the Processing tab and clicking Run MINT.",
                        type="info",
                        duration=6,
                        placement='bottom',
                        showProgress=True,
                        stack=True,
                    )

                # Metric-specific guidance: optional metrics may not be computed/generated yet.
                if metric_value in OPTIONAL_METRICS:
                    is_available, reason = check_optional_metric_availability(conn, wdir, metric_value)
                    if not is_available and metric_value == 'peak_area_fitted':
                        if reason == 'no_values':
                            return fac.AntdNotification(
                                message="No EMG values found",
                                description=(
                                    "Peak Area (EMG Fitted) was selected, but no fitted EMG values exist yet. "
                                    "Run Processing with EMG enabled, then return to Analysis."
                                ),
                                type="info",
                                duration=6,
                                placement='bottom',
                                showProgress=True,
                                stack=True,
                            )
                        return fac.AntdNotification(
                            message="EMG metric unavailable",
                            description=(
                                "Peak Area (EMG Fitted) is not available in this workspace yet. "
                                "Run Processing with EMG enabled, then refresh Analysis."
                            ),
                            type="info",
                            duration=6,
                            placement='bottom',
                            showProgress=True,
                            stack=True,
                        )
                    if not is_available and metric_value == 'scalir_conc':
                        if reason == 'missing_column':
                            return fac.AntdNotification(
                                message="Concentration file invalid",
                                description="SCALiR output is missing 'pred_conc'. Re-run SCALiR.",
                                type="warning",
                                duration=6,
                                placement='bottom',
                                showProgress=True,
                                stack=True,
                            )
                        if reason == 'no_values':
                            return fac.AntdNotification(
                                message="No concentration values found",
                                description="Concentration is selected, but SCALiR has no pred_conc values yet.",
                                type="info",
                                duration=6,
                                placement='bottom',
                                showProgress=True,
                                stack=True,
                            )
                        return fac.AntdNotification(
                            message="Concentration metric unavailable",
                            description=(
                                "Concentration values are not available yet. "
                                "Run SCALiR first to generate concentrations."
                            ),
                            type="info",
                            duration=6,
                            placement='bottom',
                            showProgress=True,
                            stack=True,
                        )
            except Exception as exc:
                logger.warning("Failed to query results count for analysis notification (wdir=%s): %s", wdir, exc)
        
        return []

    # Sidebar collapse toggle callback
    @app.callback(
        Output('analysis-sidebar', 'collapsed'),
        Output('analysis-sidebar-collapse-icon', 'icon'),
        Input('analysis-sidebar-collapse', 'nClicks'),
        State('analysis-sidebar', 'collapsed'),
        prevent_initial_call=True,
    )
    def toggle_analysis_sidebar(n_clicks, is_collapsed):
        if n_clicks is None:
            raise PreventUpdate
        new_collapsed = not is_collapsed
        icon = 'antd-right' if new_collapsed else 'antd-left'
        return new_collapsed, icon

    # Menu selection -> content visibility + sync to analysis-tabs store
    @app.callback(
        Output('analysis-qc-container', 'style'),
        Output('analysis-pca-container', 'style'),
        Output('analysis-tsne-container', 'style'),
        Output('analysis-violin-container', 'style'),
        Output('analysis-bar-container', 'style'),
        Output('analysis-comparison-container', 'style'),
        Output('analysis-clustermap-container', 'style'),
        Output('analysis-tabs', 'data'),
        Input('analysis-sidebar-menu', 'currentKey'),
        prevent_initial_call=False,
    )
    def update_analysis_content_visibility(current_key):
        # Default to 'pca' if no key selected
        active_key = current_key or 'pca'
        
        qc_style = {'display': 'block', 'padding': '16px'} if active_key == 'qc' else {'display': 'none', 'padding': '16px'}
        pca_style = {'display': 'block', 'padding': '16px'} if active_key == 'pca' else {'display': 'none', 'padding': '16px'}
        tsne_style = {'display': 'block', 'padding': '16px'} if active_key == 'tsne' else {'display': 'none', 'padding': '16px'}
        violin_style = {'display': 'block', 'padding': '16px'} if active_key == 'raincloud' else {'display': 'none', 'padding': '16px'}
        bar_style = {'display': 'block', 'padding': '16px'} if active_key == 'bar' else {'display': 'none', 'padding': '16px'}
        comparison_style = {'display': 'block', 'padding': '16px'} if active_key == 'comparison' else {'display': 'none', 'padding': '16px'}
        # Clustermap needs explicit height for spinner centering to work on first access
        clustermap_style = {
            'display': 'block' if active_key == 'clustermap' else 'none',
            'padding': '16px',
            'height': 'calc(100vh - 150px)',  # Explicit height for flexbox centering
        }
        
        # Sync to legacy analysis-tabs store for backward compatibility with other callbacks
        tabs_data = {'activeKey': active_key}
        
        return qc_style, pca_style, tsne_style, violin_style, bar_style, comparison_style, clustermap_style, tabs_data

    @app.callback(
        Output('analysis-normalization-select', 'value', allow_duplicate=True),
        Input('analysis-sidebar-menu', 'currentKey'),
        prevent_initial_call=True,
    )
    def set_norm_default_for_tab(active_tab):
        return TAB_DEFAULT_NORM.get(active_tab, 'zscore')

    @app.callback(
        Output('analysis-shared-settings', 'data'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-grouping-select', 'value'),
        State('analysis-shared-settings', 'data'),
        prevent_initial_call=False,
    )
    def persist_shared_settings(metric_value, group_by_value, stored):
        data = stored.copy() if isinstance(stored, dict) else {}
        if metric_value:
            data['metric'] = metric_value
        if group_by_value:
            data['group_by'] = group_by_value
        if not data:
            data = {'metric': 'peak_area', 'group_by': 'sample_type'}
        return data

    @app.callback(
        Output('analysis-metric-select', 'value'),
        Output('analysis-grouping-select', 'value'),
        Output('analysis-metric-select', 'options'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('section-context', 'data'),
        Input('wdir', 'data'),
        State('analysis-shared-settings', 'data'),
        State('analysis-metric-select', 'value'),
        State('analysis-grouping-select', 'value'),
        prevent_initial_call=False,
    )
    def sync_shared_controls_on_tab_switch(active_tab, section_context, wdir, stored, current_metric, current_group):
        if not section_context or section_context.get('page') != 'Analysis':
            return dash.no_update, dash.no_update, dash.no_update
        metric_options, available_metrics = _resolve_metric_options_for_workspace(wdir)
        if not isinstance(stored, dict):
            stored = {}
        metric_value = stored.get('metric') or current_metric
        group_by_value = stored.get('group_by') or current_group
        if metric_value not in available_metrics:
            if 'peak_area' in available_metrics:
                metric_value = 'peak_area'
            elif available_metrics:
                metric_value = available_metrics[0]
            else:
                metric_value = current_metric or 'peak_area'
        metric_out = dash.no_update if metric_value == current_metric else metric_value
        group_out = dash.no_update if group_by_value == current_group else group_by_value
        return metric_out, group_out, metric_options

    @app.callback(
        Output('analysis-tour', 'current'),
        Output('analysis-tour', 'open'),
        Output('analysis-tour', 'steps'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('analysis-tour-icon', 'nClicks'),
        State('analysis-tour', 'open'),
        prevent_initial_call=False,
    )
    def analysis_tour_open(active_tab, n_clicks, is_open):
        ctx = dash.callback_context
        steps = _analysis_tour_steps(active_tab)
        if not ctx.triggered:
            return 0, False, steps
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'analysis-tour-icon':
            return 0, True, steps
        if trigger == 'analysis-sidebar-menu' and is_open:
            return 0, True, steps
        return dash.no_update, dash.no_update, steps

    @app.callback(
        Output('qc-rt-graph', 'config'),
        Output('qc-mz-graph', 'config'),
        Output('qc-chromatogram', 'config'),
        Output('pca-graph', 'config'),
        Output('tsne-graph', 'config'),
        Output('violin-chromatogram', 'config'),
        Output('bar-chromatogram', 'config'),
        Output({'type': 'comparison-plot', 'index': 'main'}, 'config'),
        Output('comparison-chromatogram', 'config'),
        Input('wdir', 'data'),
    )
    def update_graph_configs(wdir):
        """Update download config with standardized filename."""
        ws_name = get_workspace_name_from_wdir(wdir) if wdir else "workspace"
        date_str = T.today()
        base_name = f"{date_str}-MINT__{ws_name}-Analysis"
        
        return (
            get_download_config(filename=f"{base_name}-QC-RT"),
            get_download_config(filename=f"{base_name}-QC-MZ"),
            get_download_config(filename=f"{base_name}-QC-Chromatogram"),
            get_download_config(filename=f"{base_name}-PCA"),
            get_download_config(filename=f"{base_name}-tSNE"),
            get_download_config(filename=f"{base_name}-Violin-Chromatogram"),
            get_download_config(filename=f"{base_name}-Bar-Chromatogram"),
            get_download_config(filename=f"{base_name}-Comparison-Plot"),
            get_download_config(filename=f"{base_name}-Comparison-Chromatogram"),
        )

    @app.callback(
        Output('clustermap-image', 'src'),
        Output('pca-graph', 'figure'),
        Output('tsne-graph', 'figure'),
        Output('violin-graphs', 'children'),
        Output('violin-comp-checks', 'options'),
        Output('violin-comp-checks', 'value'),
        Output('bar-graphs', 'children'),
        Output('bar-comp-checks', 'options'),
        Output('bar-comp-checks', 'value'),
        Output('analysis-pca-cache', 'data'),
        Output('analysis-tsne-cache', 'data'),
        Output('analysis-violin-cache', 'data'),
        Output('analysis-bar-cache', 'data'),

        Input('section-context', 'data'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('pca-x-comp', 'value'),
        Input('pca-y-comp', 'value'),
        Input('violin-comp-checks', 'value'),
        Input('bar-comp-checks', 'value'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('clustermap-regenerate-btn', 'nClicks'),
        Input('tsne-regenerate-btn', 'nClicks'),
        Input('clustermap-cluster-rows', 'checked'),
        Input('clustermap-cluster-cols', 'checked'),
        Input('clustermap-fontsize-x-slider', 'value'),
        Input('clustermap-fontsize-y-slider', 'value'),
        Input('wdir', 'data'),
        Input('tsne-x-comp', 'value'),
        Input('tsne-y-comp', 'value'),
        Input('tsne-perplexity-slider', 'value'),
        Input('analysis-pca-visible-groups', 'data'),
        State('analysis-pca-cache', 'data'),
        State('analysis-tsne-cache', 'data'),
        State('analysis-violin-cache', 'data'),
        State('analysis-bar-cache', 'data'),
        prevent_initial_call=False,
    )
    def update_content_wrapper(section_context, tab_key, x_comp, y_comp, violin_comp_checks, bar_comp_checks, metric_value, norm_value,
                        group_by, regen_clicks, tsne_regen_clicks, cluster_rows, cluster_cols, fontsize_x, fontsize_y, wdir,
                        tsne_x_comp, tsne_y_comp, tsne_perplexity, pca_visible_groups, pca_cache, tsne_cache, violin_cache, bar_cache=None):
        return update_content(
            section_context, tab_key, x_comp, y_comp, violin_comp_checks, bar_comp_checks, metric_value, norm_value,
            group_by, regen_clicks, tsne_regen_clicks, cluster_rows, cluster_cols, fontsize_x, fontsize_y, wdir,
            tsne_x_comp, tsne_y_comp, tsne_perplexity, pca_cache, tsne_cache, violin_cache, bar_cache,
            include_bar_cache=True,
            pca_visible_groups=pca_visible_groups,
        )


def update_content(section_context, tab_key, x_comp, y_comp, violin_comp_checks, bar_comp_checks, metric_value, norm_value,
                    group_by, regen_clicks, tsne_regen_clicks, cluster_rows, cluster_cols, fontsize_x, fontsize_y, wdir,
                    tsne_x_comp, tsne_y_comp, tsne_perplexity, pca_cache, tsne_cache, violin_cache, bar_cache=None,
                    include_bar_cache=False, pca_visible_groups=None):

        ctx = AnalysisUpdateContext(
            section_context=section_context,
            tab_key=tab_key,
            x_comp=x_comp,
            y_comp=y_comp,
            violin_comp_checks=violin_comp_checks,
            bar_comp_checks=bar_comp_checks,
            metric_value=metric_value,
            norm_value=norm_value,
            group_by=group_by,
            regen_clicks=regen_clicks,
            tsne_regen_clicks=tsne_regen_clicks,
            cluster_rows=cluster_rows,
            cluster_cols=cluster_cols,
            fontsize_x=fontsize_x,
            fontsize_y=fontsize_y,
            wdir=wdir,
            tsne_x_comp=tsne_x_comp,
            tsne_y_comp=tsne_y_comp,
            tsne_perplexity=tsne_perplexity,
            pca_cache=pca_cache,
            tsne_cache=tsne_cache,
            violin_cache=violin_cache,
            bar_cache=bar_cache,
            pca_visible_groups=pca_visible_groups,
        )
        result = _update_content_from_context(ctx)
        if include_bar_cache:
            return result
        # Keep backward compatibility for direct function callers/tests that still expect
        # the legacy 12-output shape (without bar cache).
        return result[:-1] if isinstance(result, tuple) and len(result) == 13 else result


def _update_content_from_context(ctx: AnalysisUpdateContext):
        if not ctx.section_context or ctx.section_context.get('page') != 'Analysis':
            raise PreventUpdate
        
        # Prevent double-firing when switching tabs forces a normalization update.
        from dash import callback_context
        triggered_props = [t['prop_id'] for t in callback_context.triggered] if callback_context.triggered else []
        if 'analysis-sidebar-menu.currentKey' in triggered_props:
            default_norm = TAB_DEFAULT_NORM.get(ctx.tab_key)
            if default_norm and ctx.norm_value != default_norm:
                raise PreventUpdate
        
        if not ctx.wdir:
            raise PreventUpdate
        
        invisible_fig = create_invisible_figure()
        
        grouping_fields = GROUPING_FIELDS
        selected_group = ctx.group_by if ctx.group_by in grouping_fields else GROUPING_FIELDS[0]

        # QC and comparison have dedicated callbacks; skip matrix prep for these tabs.
        if ctx.tab_key in {'qc', 'comparison'}:
            return _no_update_outputs_full()

        # Robust metric selection (needed early for cache keys)
        metric = 'peak_area'
        if ctx.metric_value == 'scalir_conc' or ctx.metric_value in allowed_metrics:
            metric = ctx.metric_value

        norm_value = ctx.norm_value or TAB_DEFAULT_NORM.get(ctx.tab_key, 'zscore')
        visible_groups = _normalize_pca_visible_groups(ctx.pca_visible_groups, selected_group)
        cache_key = _pca_cache_key(ctx.wdir, metric, norm_value, selected_group, visible_groups)
        tsne_perplexity_value = ctx.tsne_perplexity if ctx.tsne_perplexity else 30
        tsne_cache_key = _tsne_cache_key(ctx.wdir, metric, norm_value, selected_group, tsne_perplexity_value)
        violin_cache_key = _violin_cache_key(ctx.wdir, metric, norm_value, selected_group)
        bar_cache_key = _bar_cache_key(ctx.wdir, metric, norm_value, selected_group)
        triggered_only_group = (
            bool(triggered_props)
            and all(prop.startswith('analysis-grouping-select') for prop in triggered_props)
        )

        cached_result = _handle_pca_cached_group_change(
            ctx.tab_key, triggered_only_group, ctx.pca_cache, cache_key, selected_group,
            ctx.x_comp, ctx.y_comp, visible_groups
        )
        if cached_result is not None:
            return cached_result

        cached_result = _handle_pca_cached_render(
            ctx.tab_key, triggered_props, ctx.pca_cache, cache_key, selected_group,
            ctx.x_comp, ctx.y_comp, visible_groups
        )
        if cached_result is not None:
            return cached_result

        cached_result = _handle_tsne_cached_group_change(
            ctx.tab_key, triggered_only_group, ctx.tsne_cache, tsne_cache_key, selected_group,
            ctx.tsne_x_comp, ctx.tsne_y_comp, ctx.tsne_perplexity
        )
        if cached_result is not None:
            return cached_result

        cached_result = _handle_violin_cached_group_change(
            ctx.tab_key, triggered_only_group, ctx.violin_cache, violin_cache_key, selected_group,
            metric, norm_value, ctx.wdir
        )
        if cached_result is not None:
            return cached_result

        cached_result = _handle_violin_cached_render(
            ctx.tab_key, triggered_props, ctx.violin_cache, violin_cache_key, selected_group,
            metric, norm_value, ctx.wdir, ctx.violin_comp_checks
        )
        if cached_result is not None:
            return cached_result

        cached_result = _handle_bar_cached_group_change(
            ctx.tab_key, triggered_only_group, ctx.bar_cache, bar_cache_key, selected_group,
            metric, norm_value, ctx.wdir
        )
        if cached_result is not None:
            return cached_result

        cached_result = _handle_bar_cached_render(
            ctx.tab_key, triggered_props, ctx.bar_cache, bar_cache_key, selected_group,
            metric, norm_value, ctx.wdir, ctx.bar_comp_checks
        )
        if cached_result is not None:
            return cached_result
        
        ws_name = get_workspace_name_from_wdir(ctx.wdir) if ctx.wdir else "workspace"
        date_str = T.today()
        base_name = f"{date_str}-MINT__{ws_name}-Analysis"

        matrix_data, early_return = _prepare_matrix_data(
            wdir=ctx.wdir,
            metric=metric,
            selected_group=selected_group,
            grouping_fields=grouping_fields,
            norm_value=norm_value,
            invisible_fig=invisible_fig,
            conn_factory=duckdb_connection,
        )
        if early_return is not None:
            if metric in OPTIONAL_METRICS:
                logger.debug(
                    "Metric %s unavailable/empty for tab %s; preserving prior analysis view state.",
                    metric,
                    ctx.tab_key,
                )
                return _no_update_outputs_full()
            return early_return

        ndf = matrix_data['ndf']
        zdf = matrix_data['zdf']
        violin_matrix = matrix_data['violin_matrix']
        group_series = matrix_data['group_series']
        group_label = matrix_data['group_label']
        color_labels = matrix_data['color_labels']
        color_map = matrix_data['color_map']
        colors_df = matrix_data['colors_df']
        compound_options = matrix_data['compound_options']
        all_groups = list(dict.fromkeys(color_labels.tolist()))

        if ctx.tab_key == 'pca' and visible_groups is not None:
            keep_mask = color_labels.isin(visible_groups)
            filtered_index = color_labels.index[keep_mask]
            ndf = ndf.loc[filtered_index]
            color_labels = color_labels.loc[filtered_index]

            if ndf.shape[0] < 2:
                fig = pca.generate_pca_placeholder_figure(
                    color_map,
                    group_label,
                    ctx.x_comp,
                    ctx.y_comp,
                    all_groups=all_groups,
                    visible_groups=visible_groups,
                    message="Select at least 2 samples to compute PCA.",
                )
                return _return_pca(fig, compound_options=compound_options)

        tab_handlers = {
            'clustermap': lambda: _handle_clustermap_tab(
                triggered_props, zdf, color_labels, color_map, group_label, norm_value,
                ctx.cluster_rows, ctx.cluster_cols, ctx.fontsize_x, ctx.fontsize_y, ctx.wdir, metric
            ),
            'pca': lambda: _handle_pca_tab(
                ndf, color_labels, color_map, group_label, ctx.x_comp, ctx.y_comp,
                cache_key, colors_df, compound_options, all_groups, visible_groups
            ),
            'tsne': lambda: _handle_tsne_tab(
                ndf, color_labels, color_map, group_label, ctx.tsne_x_comp, ctx.tsne_y_comp,
                ctx.tsne_perplexity, tsne_cache_key, colors_df
            ),
            'raincloud': lambda: _handle_raincloud_tab(
                violin_matrix, group_series, color_map, group_label, metric, norm_value,
                ctx.violin_comp_checks, compound_options, base_name, violin_cache_key, colors_df
            ),
            'bar': lambda: _handle_bar_tab(
                violin_matrix, group_series, color_map, group_label, metric, norm_value,
                ctx.bar_comp_checks, compound_options, base_name, bar_cache_key, colors_df
            ),
        }
        handler = tab_handlers.get(ctx.tab_key)
        if handler is None:
            return _no_update_outputs_full()
        return handler()
