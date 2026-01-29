import importlib
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import ms_mint_app


def make_dirs():
    tmpdir = tempfile.gettempdir()
    tmpdir = os.path.join(tmpdir, "MINT")
    tmpdir = os.getenv("MINT_DATA_DIR", default=tmpdir)
    cachedir = os.path.join(tmpdir, ".cache")
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(cachedir, exist_ok=True)
    logging.info(f"MAKEDIRS: {tmpdir}, {cachedir}")
    return Path(tmpdir), Path(cachedir)


TMPDIR, CACHEDIR = make_dirs()

config = {
    "DEBUG": True,  # some Flask specific configs
    "CACHE_TYPE": "simple",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300,
}

logging.info(f'CACHEDIR: {CACHEDIR}')
logging.info(f'TMPDIR: {TMPDIR}')


def load_plugins(plugin_dir, package_name):
    logging.info('Loading plugins')
    from .plugin_interface import PluginInterface
    plugins = {}

    for file in os.listdir(plugin_dir):
        if file.endswith(".py") and not file.startswith("__"):
            module_name = file[:-3]
            module_path = f"{package_name}.{module_name}"
            module = importlib.import_module(module_path)

            for name, cls in module.__dict__.items():
                if isinstance(cls, type) and issubclass(cls, PluginInterface) and cls is not PluginInterface:
                    plugin_instance = cls()
                    plugins[plugin_instance.label] = plugin_instance

    return plugins


icons = {
    'Workspaces': 'antd-home',
    'MS-Files': 'pi-stack',
    'Targets': 'antd-unordered-list',
    'Optimization': 'antd-desktop',
    'Processing': 'antd-hourglass',
    'Quality Control': 'antd-line-chart',
    'Analysis': 'antd-monitor',
    'MS2 Browser': 'antd-home'
}


def _build_layout(*, plugins, file_explorer, initial_page_children=None, initial_section_context=None):
    import feffery_antd_components as fac
    from dash import html, dcc

    return fac.AntdLayout(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="tmpdir", data=str(TMPDIR)),
            dcc.Store(id="wdir"),
            dcc.Store(id="page-load-id", storage_type="memory"),
            dcc.Store(id="page-heartbeat-store"),
            dcc.Store(id="workspace-status", data={
                'ms_files_count': 0,
                'targets_count': 0,
                'chromatograms_count': 0,
                'selected_targets_count': 0,
                'optimization_samples_count': 0
            }),
            dcc.Store(id='section-context', data=initial_section_context),
            dcc.Store(id='workspace-busy-tracker', data=None),  # Tracks if busy notification was shown
            dcc.Interval(id="page-load-trigger", n_intervals=0, interval=200, max_intervals=1),
            dcc.Interval(id="page-heartbeat-interval", n_intervals=0, interval=4000),
            dcc.Interval(id="progress-interval", n_intervals=0, interval=20000, disabled=False),

            file_explorer.layout(),

            fac.AntdSider(
                [
                    html.Div(
                        fac.AntdTooltip(
                            fac.AntdButton(
                                id='main-sidebar-collapse',
                                type='text',
                                icon=fac.AntdIcon(
                                    id='main-sidebar-collapse-icon',
                                    icon='antd-left',
                                    style={'fontSize': '14px'}, ),
                                shape='default',
                                **{'aria-label': 'Collapse/Expand Sidebar'},
                            ),
                            title='Collapse/Expand Sidebar'
                        ),
                        style={
                            'position': 'absolute',
                            'zIndex': 1,
                            'bottom': 150,
                            'right': -10,
                            'boxShadow': 'rgb(0 0 0 / 20%) 0px 4px 10px 0px',
                            'background': 'white',
                            'borderRadius': '4px',
                        },
                    ),
                    fac.AntdFlex(
                        [
                                    fac.AntdFlex(
                                        [
                                            html.Div(id='corruption-notifications-container'),
                                            html.Div(id='notifications-container'),
                                            fac.AntdFlex(
                                                [
                                                    fac.AntdAvatar(
                                                id='logo',
                                                mode='image',
                                                shape='square',
                                                src='assets/MINT-logo.jpg',
                                                style={'width': '50%', 'height': 'auto'},
                                            ),
                                            fac.AntdMenu(
                                                id='logout-menu',
                                                menuItems=[
                                                    {
                                                        'component': 'Item',
                                                        'props': {
                                                            'key': 'logout',
                                                            'title': 'Logout',
                                                            'icon': 'antd-logout',
                                                            'href': "/logout",
                                                        },
                                                    }
                                                ],
                                                style={'display': 'none'},
                                                className='ant-menu-inline-collapsed ant-menu-vertical',
                                            )
                                        ],
                                        justify='space-between',
                                        align='center',
                                        wrap=True,
                                        style={'paddingLeft': '8px'},
                                    ),
                                    fac.AntdDivider(
                                        size='small'
                                    ),
                                    fac.AntdFlex(
                                        [
                                            fac.AntdText('Workspace:', strong=True, style={'whiteSpace': 'nowrap'}),
                                            fac.AntdCopyText(
                                                id="ws-wdir-name",
                                                locale='en-us',
                                                style={'minWidth': 0, 'flex': 1, 'textAlign': 'right'},
                                                beforeIcon=fac.AntdText(
                                                    code=True,
                                                    id="ws-wdir-name-text",
                                                    style={
                                                        'whiteSpace': 'nowrap',
                                                        'overflow': 'hidden',
                                                        'textOverflow': 'ellipsis',
                                                        'maxWidth': '100%',
                                                        'display': 'inline-block',
                                                    },
                                                ),
                                                afterIcon=fac.AntdIcon(icon='antd-like')
                                            )
                                        ],
                                        justify='space-between',
                                        align='center',
                                        style={'padding': '0 4px 0 8px', 'flexWrap': 'nowrap'},
                                        id='active-workspace-container'
                                    ),
                                    fac.AntdDivider(
                                        size='small',
                                        id='workspace-divider'
                                    ),
                                    fac.AntdMenu(
                                        menuItems=[
                                            {
                                                'component': 'Item',
                                                'props': {
                                                    'key': plugin_id,
                                                    'title': plugin_id,
                                                    'icon': icons.get(plugin_id, 'antd-home'),
                                                },
                                            }
                                            for plugin_id, plugin_instance in plugins.items()
                                        ],
                                        mode='inline',
                                        style={'overflow': 'hidden auto'},
                                        currentKey='Workspaces',
                                        id='sidebar-menu',
                                    ),
                                ],
                                vertical=True,
                                gap=2,
                                style={'width': '100%'}
                            ),
                            fac.AntdFlex(
                                [
                                    fac.AntdMenu(
                                        menuItems=[
                                            {
                                                'component': 'Item',
                                                'props': {
                                                    'key': 'docs',
                                                    'title': 'Docs',
                                                    'icon': 'antd-question-circle',
                                                    'href': "https://lewisresearchgroup.github.io/ms-mint-app/gui/",
                                                    'target': '_blank',
                                                },
                                            },
                                            {
                                                'component': 'Item',
                                                'props': {
                                                    'key': 'issues',
                                                    'title': 'Issues',
                                                    'icon': 'antd-exclamation-circle',
                                                    'href': "https://github.com/LewisResearchGroup/ms-mint-app/issues/new",
                                                    'target': '_blank',
                                                },
                                            },

                                        ],
                                        mode='horizontal',
                                        id='doc-issues-menu',
                                        style={'justifyContent': 'center'}
                                    ),
                                    html.Div(
                                        [
                                            fac.AntdFlex(
                                                [
                                                    fac.AntdText('version:', strong=True),
                                                    fac.AntdText(str(ms_mint_app.__version__), code=True),
                                                ],
                                                justify='space-between',
                                                align='center',
                                                wrap=False,
                                                style={'whiteSpace': 'nowrap', 'overflow': 'hidden'},
                                            ),
                                        ],
                                        style={'margin': '10px 4px 10px 8px'},
                                        id='version-info',
                                    ),
                                ],
                                vertical=True,
                                gap=2,
                                style={'width': '100%'}
                            ),

                        ],
                        justify='space-between',
                        align='center',
                        vertical=True,
                        style={'height': '100%'}
                    )
                ],
                collapsible=True,
                collapsedWidth=60,
                trigger=None,
                width=250,
                style={'backgroundColor': 'white'},
                id='main-sidebar',
                className="sidebar-mint"
            ),
            fac.AntdContent(
                id='page-content',
                style={'backgroundColor': 'white', 'padding': '1rem 2rem'},
                children=initial_page_children,
            ),
        ],
        style={'height': '100vh'},
    )


def register_callbacks(app, cache, fsc, args, *, plugins, file_explorer):
    from dash.dependencies import Input, Output, State
    import dash
    from dash import html
    from dash.exceptions import PreventUpdate
    from flask_login import current_user
    import feffery_antd_components as fac

    from .duckdb_manager import (
        duckdb_connection_mint,
        ensure_exploration_workspace,
        is_workspace_corrupted,
        is_workspace_busy_probe,
        get_busy_notification,
        clear_busy_flag,
        mark_page_load_active,
    )

    logging.info("Register callbacks")
    upload_root = os.getenv("MINT_DATA_DIR", tempfile.gettempdir())
    upload_dir = str(Path(upload_root) / "MINT-Uploads")
    UPLOAD_FOLDER_ROOT = upload_dir

    file_explorer.callbacks(app=app, fsc=fsc, cache=cache)

    # Eagerly register callbacks for all plugins.
    # Dash expects callbacks to be registered before the client tries to use them.
    for label, plugin in plugins.items():
        logging.info(f"Loading callbacks of plugin {label}")
        if label in ['MS-Files']:
            plugin.callbacks(app=app, fsc=fsc, cache=cache, args=args)
        else:
            plugin.callbacks(app=app, fsc=fsc, cache=cache)

    app.clientside_callback(
        """(nClicks, collapsed) => {
            return [!collapsed, collapsed ? 'antd-left' : 'antd-right'];
        }""",
        [
            Output('main-sidebar', 'collapsed'),
            Output('main-sidebar-collapse-icon', 'icon'),
        ],
        Input('main-sidebar-collapse', 'nClicks'),
        State('main-sidebar', 'collapsed'),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """(n, current) => {
            if (current) {
                return current;
            }
            try {
                if (window.crypto && window.crypto.randomUUID) {
                    return window.crypto.randomUUID();
                }
            } catch (e) {
                // ignore and fall through to timestamp-based id
            }
            return `page-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        }""",
        Output('page-load-id', 'data'),
        Input('page-load-trigger', 'n_intervals'),
        State('page-load-id', 'data'),
        prevent_initial_call=False,
    )

    @app.callback(
        Output('page-heartbeat-store', 'data'),
        Input('wdir', 'data'),
        Input('page-load-id', 'data'),
        Input('page-heartbeat-interval', 'n_intervals'),
        prevent_initial_call=True,
    )
    def page_heartbeat(wdir, page_load_id, _n):
        if not wdir or not page_load_id:
            raise PreventUpdate
        mark_page_load_active(wdir, page_load_id)
        return {'page_load_id': page_load_id, 'ts': time.time()}

    @app.callback(
        Output('page-content', 'children'),
        Output('section-context', 'data'),

        Input('tmpdir', 'data'),
        Input('sidebar-menu', 'currentKey'),
        prevent_initial_call=True
    )
    def menu_navigation(tmpdir, currentKey):
        if not currentKey or currentKey not in plugins:
            raise PreventUpdate
        section_context = {
            'page': currentKey,
            'time': time.time()
        }
        try:
            return plugins[currentKey].layout(), section_context
        except Exception:
            logging.exception("Failed to render layout for %s", currentKey)
            return (
                html.Div(
                    [
                        html.H4(f"Failed to render page: {currentKey}"),
                        html.Pre("See server logs for details."),
                    ]
                ),
                section_context,
            )

    @app.callback(
        Output('corruption-notifications-container', 'children'),
        Output('workspace-busy-tracker', 'data'),

        Input('section-context', 'data'),
        Input('wdir', 'data'),
        prevent_initial_call=False
    )
    def check_corruption_on_navigation(section_context, wdir):
        """Show notification when navigating to a tab or when workspace changes if corrupted/busy."""
        if not section_context or section_context.get('page') != 'Workspaces':
            raise PreventUpdate
        if not wdir:
            raise PreventUpdate
        if is_workspace_corrupted(wdir):
            return fac.AntdNotification(
                message="[!] Database Corrupted",
                description="This workspace's database is corrupted. Please go to Workspaces tab and delete this workspace, then restore from backup or recreate it.",
                type="error",
                duration=15,
                placement='bottom',
                showProgress=True,
            ), None
        if is_workspace_busy_probe(wdir):
            busy = get_busy_notification(wdir)
            if busy:
                # Mark that we showed a busy notification for this workspace
                return fac.AntdNotification(**busy), {'wdir': wdir, 'busy': True, 'ts': time.time()}
        raise PreventUpdate

    @app.callback(
        Output('corruption-notifications-container', 'children', allow_duplicate=True),
        Output('workspace-busy-tracker', 'data', allow_duplicate=True),

        Input('page-heartbeat-interval', 'n_intervals'),
        Input('workspace-busy-tracker', 'data'),
        State('wdir', 'data'),
        State('section-context', 'data'),
        prevent_initial_call=True
    )
    def check_database_ready(n_intervals, busy_tracker, wdir, section_context):
        """
        Polls for busy/ready state.
        1. If busy, ensures we are tracking it.
        2. If previously busy and now free, notifies the user.
        """
        if not section_context or section_context.get('page') != 'Workspaces':
            raise PreventUpdate

        if not wdir:
            # logging.info("No wdir.")
            raise PreventUpdate

        if not busy_tracker:
            is_busy = is_workspace_busy_probe(wdir)
            if is_busy:
                busy = get_busy_notification(wdir)
                if busy:
                    return fac.AntdNotification(**busy), {'wdir': wdir, 'busy': True, 'ts': time.time()}
            raise PreventUpdate
        
        # Only check for the same workspace that was busy
        tracked_wdir = busy_tracker.get('wdir')
        if tracked_wdir != wdir:
            logging.debug("Wdir mismatch: tracked=%s, current=%s", tracked_wdir, wdir)
            raise PreventUpdate
        
        # Check if still busy
        is_busy = is_workspace_busy_probe(wdir)
        # logging.info(f"Checking database ready: trigger={trigger}, wdir={wdir}, busy={is_busy}, tracker={busy_tracker}")
        
        if is_busy:
            # If not tracking yet, start tracking!
            if not busy_tracker or not busy_tracker.get('busy'):
                logging.debug("Polling detected busy state for %s. Starting tracking.", wdir)
                return dash.no_update, {'wdir': wdir, 'busy': True, 'ts': time.time()}
            return dash.no_update, busy_tracker # Keep tracking

        # Database is now available! Clear the busy flag and notify user
        if busy_tracker and busy_tracker.get('busy'):
            logging.debug("Database ready! sending notification for %s", wdir)
            clear_busy_flag(wdir)
            return fac.AntdNotification(
                message="Database Ready",
                description="The workspace database is now available. You can continue your work.",
                type="success",
                duration=5,
                placement='bottom',
                showProgress=True,
                key=str(time.time()) # Force re-render
            ), None

        return dash.no_update, None


    @app.callback(
        Output('logo', 'style'),
        Output('logout-menu', 'style', allow_duplicate=True),
        Output('active-workspace-container', 'style'),
        Output('workspace-divider', 'style'),
        Output('doc-issues-menu', 'mode'),
        Output('version-info', 'style'),

        Input('main-sidebar', 'collapsed'),
        prevent_initial_call=True
    )
    def toggle_sidebar(collapsed):
        logo_style = {'width': '100%', 'height': 'auto'} if collapsed else {'width': '50%', 'height': 'auto'}
        logout_menu_style = {'width': '100%', 'display': 'none'} if collapsed else {'width': '60px', 'display': 'none'}
        active_workspace_container_style = {'display': 'none', 'padding': '0 4px'} if collapsed else {'display': 'flex',
                                                                                                      'padding': '0 4px'}
        ws_divider_style = {'display': 'none'} if collapsed else {'display': 'block'}
        doc_issues_menu_mode = 'vertical' if collapsed else 'horizontal'
        version_info_style = {'display': 'none'} if collapsed else {'display': 'block', 'margin': '4px'}

        return (logo_style, logout_menu_style, active_workspace_container_style, ws_divider_style,
                doc_issues_menu_mode, version_info_style)

    def user_tmpdir():
        uid = current_user.get_id() if hasattr(app.server, "login_manager") and current_user.is_authenticated else None
        base = TMPDIR / ("User" if uid else "Local")
        sub = uid or ""
        path = base / sub
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @app.callback(
        Output("tmpdir", "data"),
        Output("logout-menu", "style"),

        Input("url", "pathname"),
        State("main-sidebar", "collapsed"),
    )
    def init_session(_pathname, collapsed):
        logout_menu_style = {"width": "100%" if collapsed else "60px"}

        if hasattr(app.server, "login_manager") and current_user.is_authenticated:
            logout_menu_style["display"] = "block"
        else:
            logout_menu_style["display"] = "none"

        ud = user_tmpdir()

        if not Path(ud, 'mint.db').exists():
            with duckdb_connection_mint(ud):
                logging.info("Created Workspaces DB...")

        try:
            ensure_exploration_workspace(Path(ud))
        except Exception:
            logging.exception("Failed to initialize exploration workspace")
        return ud, logout_menu_style

    logging.info("Done registering callbacks")


def create_app(**kwargs):
    logging.info('Create application')
    logging.info(f'ms-mint-app: {ms_mint_app.__version__}')

    from dash import DiskcacheManager, Dash
    import feffery_utils_components  # noqa: F401
    from dash_extensions.enrich import FileSystemCache
    from flask_caching import Cache

    from .plugin_manager import PluginManager
    from .plugins.explorer import FileExplorer

    if 'REDIS_URL' in os.environ:
        # Use Redis & Celery if REDIS_URL set as an env variable
        from celery import Celery
        from dash import CeleryManager
        celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
        background_callback_manager = CeleryManager(celery_app)

    else:
        # Diskcache for non-production apps when developing locally
        from uuid import uuid4
        import diskcache
        import shutil
        launch_uid = uuid4()

        try:
            cache = diskcache.Cache(CACHEDIR)
        except Exception as e:
            logging.error(f"Failed to initialize DiskCache at {CACHEDIR}: {e}")
            logging.warning("Attempting to clear cache directory and retry...")
            try:
                shutil.rmtree(CACHEDIR)
                os.makedirs(CACHEDIR, exist_ok=True)
                cache = diskcache.Cache(CACHEDIR)
                logging.info("Successfully re-initialized DiskCache after clearing.")
            except Exception as e2:
                logging.critical(f"Failed to recover DiskCache: {e2}")
                raise e2

        background_callback_manager = DiskcacheManager(
            cache, expire=60,
        )

    plugin_manager = PluginManager()
    plugins = plugin_manager.get_plugins()
    logging.info(f'Plugins: {plugins.keys()}')

    file_explorer = FileExplorer()

    initial_section_context = {'page': 'Workspaces', 'time': time.time()}
    initial_page_children = None
    if 'Workspaces' in plugins and hasattr(plugins['Workspaces'], 'ensure_loaded'):
        plugins['Workspaces'].ensure_loaded()
        initial_page_children = plugins['Workspaces'].layout()

    layout = _build_layout(
        plugins=plugins,
        file_explorer=file_explorer,
        initial_page_children=initial_page_children,
        initial_section_context=initial_section_context,
    )

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        kwargs['assets_folder'] = os.path.join(sys._MEIPASS, 'ms_mint_app', 'assets')

    app = Dash(
        __name__,
        background_callback_manager=background_callback_manager,
        **kwargs,
    )

    app.layout = layout
    app.title = "MINT"
    app.config["suppress_callback_exceptions"] = True

    upload_root = os.getenv("MINT_DATA_DIR", tempfile.gettempdir())
    CACHE_DIR = str(Path(upload_root) / "MINT-Cache")

    logging.info('Defining filesystem cache')
    cache = Cache(
        app.server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": CACHE_DIR}
    )

    fsc = FileSystemCache(str(CACHEDIR))
    logging.info('Done creating app')
    return app, cache, fsc, plugins, file_explorer
