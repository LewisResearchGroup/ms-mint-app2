"""Clustermap tab for Analysis plugin."""

from ._shared import (
    fac, html, dcc, logger,
    Input, Output, State, PreventUpdate,
    NORM_OPTIONS, Path, time,
    dash
)
from ..scalir import slugify_label
from ... import tools as T
from ...duckdb_manager import get_workspace_name_from_wdir


def create_layout():
    """Return the Clustermap tab layout component."""
    return html.Div([
        fac.AntdFlex(
            [
                # Left side: Options panel
                html.Div(
                    [
                        fac.AntdText("Clustering", strong=True, style={'fontSize': '14px', 'marginBottom': '16px', 'display': 'block'}),
                        fac.AntdFlex(
                            [
                                fac.AntdSwitch(
                                    id='clustermap-cluster-rows',
                                    checked=True,
                                    checkedChildren='On',
                                    unCheckedChildren='Off',
                                ),
                                fac.AntdText("Cluster Rows", style={'marginLeft': '8px'}),
                            ],
                            align='center',
                            style={'marginBottom': '24px'},
                        ),
                        fac.AntdFlex(
                            [
                                fac.AntdSwitch(
                                    id='clustermap-cluster-cols',
                                    checked=False,
                                    checkedChildren='On',
                                    unCheckedChildren='Off',
                                ),
                                fac.AntdText("Cluster Columns", style={'marginLeft': '8px'}),
                            ],
                            align='center',
                            style={'marginBottom': '8px'},
                        ),
                        fac.AntdDivider(style={'margin': '16px 0'}),
                        fac.AntdText("Fontsize", strong=True, style={'fontSize': '14px', 'marginBottom': '16px', 'display': 'block'}),
                        fac.AntdText("X-axis:", style={'fontWeight': 500, 'fontSize': '12px', 'display': 'block', 'marginBottom': '8px'}),
                        fac.AntdSlider(
                            id='clustermap-fontsize-x-slider',
                            min=0,
                            max=20,
                            step=1,
                            value=5,
                            marks={0: '0', 10: '10', 20: '20'},
                            style={'width': '100%', 'marginBottom': '24px'},
                        ),
                        fac.AntdText("Y-axis:", style={'fontWeight': 500, 'fontSize': '12px', 'display': 'block', 'marginBottom': '8px'}),
                        fac.AntdSlider(
                            id='clustermap-fontsize-y-slider',
                            min=0,
                            max=20,
                            step=1,
                            value=5,
                            marks={0: '0', 10: '10', 20: '20'},
                            style={'width': '100%', 'marginBottom': '24px'},
                        ),
                        fac.AntdFlex(
                            [
                                fac.AntdButton(
                                    "Regenerate",
                                    id='clustermap-regenerate-btn',
                                    type='default',
                                    style={'flex': '1'},
                                ),
                                fac.AntdTooltip(
                                    fac.AntdIcon(
                                        icon='antd-question-circle',
                                        style={'marginLeft': '8px', 'color': 'gray', 'fontSize': '14px'}
                                    ),
                                    title='Regenerate the clustermap with current settings',
                                    placement='right'
                                ),
                            ],
                            align='center',
                            style={'marginBottom': '16px'},
                        ),
                        fac.AntdDivider(style={'margin': '16px 0'}),
                        fac.AntdFlex(
                            [
                                fac.AntdButton(
                                    "Save PNG",
                                    id='clustermap-save-png-btn',
                                    type='default',
                                    style={'flex': '1'},
                                ),
                                fac.AntdTooltip(
                                    fac.AntdIcon(
                                        icon='antd-question-circle',
                                        style={'marginLeft': '8px', 'color': 'gray', 'fontSize': '14px'}
                                    ),
                                    title='Download the clustermap as a PNG image',
                                    placement='right'
                                ),
                            ],
                            align='center',
                        ),
                        dcc.Download(id='clustermap-download'),
                    ],
                    style={
                        'width': '250px',
                        'minWidth': '250px',
                        'padding': '16px',
                        'flexShrink': 0,
                    },
                ),
                # Right side: Clustermap image
                html.Div(
                    fac.AntdSpin(
                        html.Div(
                            fac.AntdImage(
                                id='clustermap-image',
                                preview={'mask': 'Click to Zoom'},
                                locale='en-us',
                                fallback='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
                                style={
                                    'maxWidth': '100%',
                                    'maxHeight': 'calc(100vh - 200px)',
                                    'objectFit': 'contain',
                                    'cursor': 'zoom-in',
                                },
                            ),
                            style={
                                'display': 'flex',
                                'justifyContent': 'center',
                                'alignItems': 'center',
                                'width': '100%',
                            },
                        ),
                        id='clustermap-spinner',
                        spinning=True,
                        text='Loading clustermap...',
                        style={
                            'minHeight': 'calc(100vh - 250px)',
                            'width': '100%',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                        },
                    ),
                    style={
                        'flex': '1',
                        'width': '100%',
                        'minHeight': 'calc(100vh - 200px)',
                    },
                ),
            ],
            style={'height': '100%'},
        ),
    ], style={'height': '100%'})


def generate_clustermap(zdf, color_labels, color_map, group_label, norm_value, cluster_rows, cluster_cols, fontsize_x, fontsize_y, wdir, metric, triggered_prop, provided_norm):
    """Generate the clustermap figure."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.patches as mpatches
    from io import BytesIO
    import base64

    matplotlib.use('Agg')
    # Use base font_scale for other elements, apply specific sizes to tick labels
    sns.set_theme(style='white', font_scale=0.5)
    sample_colors = None
    if color_map:
        sample_colors = [color_map.get(lbl, '#bbbbbb') for lbl in color_labels]
    
    norm_label = next((o['label'] for o in NORM_OPTIONS if o['value'] == norm_value), norm_value)
    
    vmin = -2.00 if norm_value == 'zscore' else None
    vmax = 2.05 if norm_value == 'zscore' else None
    fig = sns.clustermap(
                            zdf.T,
                            method='ward', metric='euclidean', 
                            cmap='vlag', center=0, vmin=vmin, vmax=vmax,
                            standard_scale=None,
                            row_cluster=cluster_rows if cluster_rows is not None else True,
                            col_cluster=cluster_cols if cluster_cols is not None else False, 
                            dendrogram_ratio=0.1,
                            figsize=(8, 8),
                            cbar_kws={"orientation": "horizontal"},
                            cbar_pos=(0.00, 0.95, 0.075, 0.01),
                            col_colors=sample_colors,
                            row_colors=['#ffffff'] * len(zdf.T.index),
                            colors_ratio=(0.0015, 0.015)
                        )
    # Ensure white backgrounds across panels
    fig.fig.patch.set_facecolor('white')
    fig.ax_heatmap.set_facecolor('white')
    fig.ax_col_dendrogram.set_facecolor('white')
    fig.ax_row_dendrogram.set_facecolor('white')
    # Apply custom font sizes to x and y tick labels
    x_fontsize = fontsize_x if fontsize_x else 5
    y_fontsize = fontsize_y if fontsize_y else 5
    fig.ax_heatmap.tick_params(axis='x', labelsize=x_fontsize, length=0, rotation=90)
    fig.ax_heatmap.tick_params(axis='y', labelsize=y_fontsize, length=0)
    
    fig.ax_heatmap.set_xlabel('Samples', fontsize=7, labelpad=8)
    fig.ax_cbar.tick_params(which='both', axis='both', width=0.3, length=2, labelsize=4)
    fig.ax_cbar.set_title(norm_label, fontsize=6, pad=4)
    # Legend for grouping colors (top right)
    if color_map:
        used_types = [lbl for lbl in color_labels if lbl in color_map]
        handles = [
            mpatches.Patch(color=color_map[stype], label=stype)
            for stype in dict.fromkeys(used_types)  # preserve order, unique
            if stype in color_map
        ]
        if handles:
            fig.ax_heatmap.legend(
                handles=handles,
                title=group_label,
                bbox_to_anchor=(-0.15, 1.025),
                loc='upper right',
                ncol=1,
                frameon=False,
                fontsize=5,
                title_fontsize=5,
                labelspacing=0.75,
                
            )

    buf = BytesIO()
    # Save a high-resolution copy to disk for durability/exports
    try:
        # Avoid double-saving when the norm dropdown hasn't populated yet (provided_norm is None)
        # Also skip the immediate tab-change trigger; wait for the follow-up with the final norm value.
        if wdir and provided_norm is not None and triggered_prop != 'analysis-sidebar-menu':
            cm_dir = Path(wdir) / "analysis" / "clustermap"
            cm_dir.mkdir(parents=True, exist_ok=True)
            safe_metric = slugify_label(metric)
            file_name = f"{safe_metric}_{norm_value}_clustermap.png"
            save_path = cm_dir / file_name
            should_save = True
            if save_path.exists():
                last_write = save_path.stat().st_mtime
                should_save = (time.time() - last_write) > 30
            if should_save:
                fig.savefig(save_path, format="png", dpi=600)
                logger.info(f"Saved high-res Clustermap: {save_path}")
    except Exception:
        logger.error("Failed to save Clustermap image.", exc_info=True)
        pass
    fig.savefig(buf, format="png", dpi=300)
    # Avoid accumulating open figures across callbacks
    plt.close(fig.fig)
    
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
    return fig_bar_matplotlib


def register_callbacks(app):
    """Register Clustermap callbacks."""

    @app.callback(
        Output('clustermap-spinner', 'spinning'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('clustermap-image', 'src'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        prevent_initial_call=False,
    )
    def toggle_clustermap_spinner(active_tab, bar_src, metric_value, norm_value):
        from dash import callback_context

        if active_tab != 'clustermap':
            return False

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        # When user switches to clustermap or changes metric/normalization, force spinner on even if previous image exists
        if trigger in ('analysis-sidebar-menu', 'analysis-metric-select', 'analysis-normalization-select'):
            return True

        # Otherwise, keep spinning until image src is set
        return bar_src is None

    @app.callback(
        Output('clustermap-download', 'data'),
        Input('clustermap-save-png-btn', 'nClicks'),
        State('clustermap-image', 'src'),
        State('wdir', 'data'),
        prevent_initial_call=True,
    )
    def save_clustermap_png(n_clicks, img_src, wdir):
        if not n_clicks or not img_src:
            raise PreventUpdate
        
        ws_name = get_workspace_name_from_wdir(wdir) if wdir else "workspace"
        date_str = T.today()
        filename = f"{date_str}-MINT__{ws_name}-Analysis-Clustermap.png"

        # Extract base64 data from src
        if ',' in img_src:
            img_data = img_src.split(',')[1]
        else:
            img_data = img_src
            
        return dict(
            content=img_data,
            filename=filename,
            type='image/png',
            base64=True,
        )
