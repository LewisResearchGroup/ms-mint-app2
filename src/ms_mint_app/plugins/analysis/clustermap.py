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


def _artifact_paths(wdir, metric, norm_value):
    """Return workspace artifact paths for the current clustermap."""
    cm_dir = Path(wdir) / "analysis" / "clustermap"
    safe_metric = slugify_label(metric)
    stem = f"{safe_metric}_{norm_value}_clustermap"
    return cm_dir, cm_dir / f"{stem}.png", cm_dir / f"{stem}.csv"


def build_clustermap_download(img_src, wdir, metric_value, norm_value, filename):
    """Package the current clustermap PNG and matrix CSV into a zip download."""
    import base64
    from io import BytesIO
    from zipfile import ZIP_DEFLATED, ZipFile

    metric_value = metric_value or 'peak_area'
    norm_value = norm_value or 'zscore'
    safe_metric = slugify_label(metric_value)
    png_name = f"{safe_metric}_{norm_value}_clustermap.png"
    csv_name = f"{safe_metric}_{norm_value}_clustermap.csv"

    if ',' in img_src:
        img_data = img_src.split(',')[1]
    else:
        img_data = img_src
    png_bytes = base64.b64decode(img_data)

    csv_bytes = None
    if wdir:
        _, _, csv_path = _artifact_paths(wdir, metric_value, norm_value)
        if csv_path.exists():
            csv_bytes = csv_path.read_bytes()
    if csv_bytes is None:
        raise FileNotFoundError(f"Missing clustermap CSV artifact for {metric_value}/{norm_value} in {wdir}")

    def write_bundle(buffer: BytesIO):
        with ZipFile(buffer, mode='w', compression=ZIP_DEFLATED) as zf:
            zf.writestr(png_name, png_bytes)
            zf.writestr(csv_name, csv_bytes)

    return dcc.send_bytes(write_bundle, filename)


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
                        fac.AntdText("Cbar + legend:", style={'fontWeight': 500, 'fontSize': '12px', 'display': 'block', 'marginBottom': '8px'}),
                        fac.AntdSlider(
                            id='clustermap-fontsize-aux-slider',
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
                                    "Save PNG + Data",
                                    id='clustermap-save-png-btn',
                                    type='default',
                                    style={'flex': '1'},
                                ),
                                fac.AntdTooltip(
                                    fac.AntdIcon(
                                        icon='antd-question-circle',
                                        style={'marginLeft': '8px', 'color': 'gray', 'fontSize': '14px'}
                                    ),
                                    title='Download the clustermap PNG and clustered matrix data',
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


def generate_clustermap(zdf, color_labels, color_map, group_label, norm_value, cluster_rows, cluster_cols, fontsize_x, fontsize_y, fontsize_aux, wdir, metric):
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
    # Seaborn can recreate tick label artists during clustermap layout, so set
    # tick-label font sizes explicitly on the final artists.
    x_fontsize = fontsize_x if fontsize_x else 5
    y_fontsize = fontsize_y if fontsize_y else 5
    aux_fontsize = fontsize_aux if fontsize_aux else 5
    axis_label_fontsize = max(x_fontsize + 1, y_fontsize + 1, 6)  # Ensure axis labels are at least slightly larger than ticks
    fig.ax_heatmap.tick_params(axis='x', length=0, rotation=90)
    fig.ax_heatmap.tick_params(axis='y', length=0)
    for tick in fig.ax_heatmap.get_xticklabels():
        tick.set_fontsize(x_fontsize)
    for tick in fig.ax_heatmap.get_yticklabels():
        tick.set_fontsize(y_fontsize)
    
    fig.ax_heatmap.set_xlabel('Samples', fontsize=axis_label_fontsize, labelpad=10)
    if fig.ax_heatmap.get_ylabel():
        fig.ax_heatmap.set_ylabel(fig.ax_heatmap.get_ylabel(), fontsize=axis_label_fontsize)
    fig.ax_cbar.tick_params(which='both', axis='both', width=0.3, length=2, labelsize=aux_fontsize - 1)
    fig.ax_cbar.set_title(norm_label, fontsize=aux_fontsize, pad=4)
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
                fontsize=aux_fontsize - 1,
                title_fontsize=aux_fontsize,
                labelspacing=0.75,
                
            )

    buf = BytesIO()
    # Save a high-resolution copy to disk for durability/exports
    try:
        if wdir:
            cm_dir, png_path, csv_path = _artifact_paths(wdir, metric, norm_value)
            cm_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path, format="png", dpi=600)
            fig.data2d.to_csv(csv_path, index=True)
            logger.info("Saved clustermap artifacts: %s and %s", png_path, csv_path)
    except Exception:
        logger.error("Failed to save clustermap artifacts.", exc_info=True)
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
        State('analysis-metric-select', 'value'),
        State('analysis-normalization-select', 'value'),
        prevent_initial_call=True,
    )
    def save_clustermap_png(n_clicks, img_src, wdir, metric_value, norm_value):
        if not n_clicks or not img_src:
            raise PreventUpdate
        
        ws_name = get_workspace_name_from_wdir(wdir) if wdir else "workspace"
        date_str = T.today()
        filename = f"{date_str}-MINT__{ws_name}-Analysis-Clustermap.zip"
        try:
            return build_clustermap_download(img_src, wdir, metric_value, norm_value, filename)
        except FileNotFoundError:
            logger.warning(
                "Clustermap CSV artifact missing for download: metric=%s norm=%s wdir=%s",
                metric_value, norm_value, wdir,
            )
            raise PreventUpdate
