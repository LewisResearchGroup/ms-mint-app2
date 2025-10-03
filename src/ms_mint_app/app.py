import importlib
import logging
import os
import tempfile
from pathlib import Path

import feffery_antd_components as fac
from dash import html, dcc, DiskcacheManager, Dash
from dash.dependencies import Input, Output, State
from dash_extensions.enrich import FileSystemCache
from flask_caching import Cache
from flask_login import current_user

import ms_mint
import ms_mint_app
from .plugin_interface import PluginInterface
from .plugin_manager import PluginManager
from .plugins.explorer import FileExplorer


def make_dirs():
    tmpdir = tempfile.gettempdir()
    tmpdir = os.path.join(tmpdir, "MINT")
    tmpdir = os.getenv("MINT_DATA_DIR", default=tmpdir)
    cachedir = os.path.join(tmpdir, ".cache")
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(cachedir, exist_ok=True)
    print("MAKEDIRS:", tmpdir, cachedir)
    return Path(tmpdir), Path(cachedir)


TMPDIR, CACHEDIR = make_dirs()

config = {
    "DEBUG": True,  # some Flask specific configs
    "CACHE_TYPE": "simple",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300,
}

logging.info(f'CACHEDIR: {CACHEDIR}')
logging.info(f'TMPDIR: {TMPDIR}')

## Diskcache
from uuid import uuid4
import diskcache

def load_plugins(plugin_dir, package_name):
    logging.info('Loading plugins')
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


# Assuming 'plugins' is a subdirectory in the same directory as this script
plugin_manager = PluginManager()
plugins = plugin_manager.get_plugins()

logging.info(f'Plugins: {plugins.keys()}')

_layout = fac.AntdLayout(
    [
        dcc.Store(id="tmpdir", data=str(TMPDIR)),
        dcc.Store(id="wdir"),
        dcc.Interval(id="progress-interval", n_intervals=0, interval=20000, disabled=False),

        file_explorer.layout(),

        fac.AntdSider(
            [
                fac.AntdFlex(
                    [
                        fac.AntdFlex(
                            [
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
                                ),
                                fac.AntdDivider(
                                    size='small'
                                ),
                                fac.AntdFlex(
                                    [
                                        fac.AntdText('Workspace:', strong=True),
                                        fac.AntdCopyText(
                                            id="ws-wdir-name",
                                            locale='en-us',
                                            beforeIcon=fac.AntdText(code=True, id="ws-wdir-name-text"),
                                            afterIcon=fac.AntdIcon(icon='antd-like')
                                        )
                                    ],
                                    justify='space-between',
                                    align='center',
                                    style={'padding': '0 4px'},
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
                                                'icon': 'antd-home',
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
                                    style={'justifyContent': 'center', 'width': '100%'}
                                ),
                                html.Div(
                                    [
                                        fac.AntdFlex(
                                            [
                                                fac.AntdText('ms-mint:', strong=True),
                                                fac.AntdText(str(ms_mint.__version__), code=True),

                                            ],
                                            justify='space-between',
                                            align='center',
                                        ),
                                        fac.AntdFlex(
                                            [
                                                fac.AntdText('ms-mint-app:', strong=True),
                                                fac.AntdText(str(ms_mint_app.__version__), code=True),

                                            ],
                                            justify='space-between',
                                            align='center',
                                        ),
                                    ],
                                    style={'margin': '4px'},
                                    id='version-info',
                                ),
                            ],
                            vertical=True,
                            gap=2,
                        ),

                    ],
                    justify='space-between',
                    align='center',
                    vertical=True,
                    style={'height': '100%'}
                )
            ],
            collapsible=True,
            width=250,
            style={'backgroundColor': 'white'},
            id='sidebar',
        ),
        fac.AntdContent(
            id='page-content',
            style={'backgroundColor': 'white', 'padding': '2rem'},
        ),
    ],
    style={'height': '100vh'},
)


def register_callbacks(app, cache, fsc, args):
    logging.info("Register callbacks")
    upload_root = os.getenv("MINT_DATA_DIR", tempfile.gettempdir())
    upload_dir = str(Path(upload_root) / "MINT-Uploads")
    UPLOAD_FOLDER_ROOT = upload_dir

    file_explorer.callbacks(app=app, fsc=fsc, cache=cache)

    for label, plugin in plugins.items():
        logging.info(f"Loading callbacks of plugin {label}")
        if label in ['MS-Files']:
            plugin.callbacks(app=app, fsc=fsc, cache=cache, args=args)
        else:
            plugin.callbacks(app=app, fsc=fsc, cache=cache)

    @app.callback(
        Output('page-content', 'children'),
        Input('sidebar-menu', 'currentKey')
    )
    def menu_navigation(currentKey):
        return plugins[currentKey].layout()

    @app.callback(
        Output('logo', 'style'),
        Output('logout-menu', 'style', allow_duplicate=True),
        Output('active-workspace-container', 'style'),
        Output('workspace-divider', 'style'),
        Output('doc-issues-menu', 'mode'),
        Output('version-info', 'style'),

        Input('sidebar', 'collapsed'),
        prevent_initial_call=True
    )
    def toggle_sidebar(collapsed):
        logo_style = {'width': '100%', 'height': 'auto'} if collapsed else {'width': '50%', 'height': 'auto'}
        logout_menu_style = {'width': '100%', 'display': 'none'} if collapsed else {'width': '60px', 'display': 'none'}
        active_workspace_container_style = {'display': 'none', 'padding': '0 4px'} if collapsed else {'display': 'flex', 'padding': '0 4px'}
        ws_divider_style = {'display': 'none'} if collapsed else {'display': 'block'}
        doc_issues_menu_mode = 'vertical' if collapsed else 'horizontal'
        version_info_style = {'display': 'none'} if collapsed else {'display': 'block', 'margin': '4px'}

        return (logo_style, logout_menu_style, active_workspace_container_style, ws_divider_style,
                doc_issues_menu_mode, version_info_style)

    @app.callback(
        Output("tmpdir", "data"),
        Output("logout-menu", "style"),

        Input("progress-interval", "n_intervals"),
        State('sidebar', 'collapsed'),
        # prevent_initial_call=True
    )
    def update_tmpdir(x, collapsed):
        logout_menu_style = {'width': '100%'} if collapsed else {'width': '60px'}

        if hasattr(app.server, "login_manager"):
            username = current_user.username
            tmpdir = str(TMPDIR / "User" / username)
            logout_menu_style['display'] = 'block'
            return tmpdir, logout_menu_style
        logout_menu_style['display'] = 'none'
        return str(TMPDIR / "Local"), logout_menu_style

    logging.info("Done registering callbacks")


def create_app(**kwargs):
    logging.info('Create application')
    logging.info(f'ms-mint: {ms_mint.__version__}, ({ms_mint.__file__})')
    logging.info(f'ms-mint-app: {ms_mint_app.__version__}, ({ms_mint.__file__})')

    if 'REDIS_URL' in os.environ:
        # Use Redis & Celery if REDIS_URL set as an env variable
        from celery import Celery
        celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
        background_callback_manager = CeleryManager(celery_app)

    else:
        # Diskcache for non-production apps when developing locally
        launch_uid = uuid4()
        cache = diskcache.Cache(CACHEDIR)
        background_callback_manager = DiskcacheManager(
            cache, expire=60,
        )

    app = Dash(
        __name__,
        background_callback_manager=background_callback_manager,
        **kwargs,
    )

    app.layout = _layout
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
    return app, cache, fsc
