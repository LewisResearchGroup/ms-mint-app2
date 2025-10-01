import os
import tempfile
import logging
import importlib

import pandas as pd

from pathlib import Path as P


import dash

from dash import html, dcc, DiskcacheManager

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.dcc import Download
from dash_extensions.enrich import FileSystemCache

import feffery_antd_components as fac
from .plugin_manager import PluginManager
from .plugin_interface import PluginInterface

import dash_bootstrap_components as dbc

from flask_caching import Cache
from flask_login import current_user

import ms_mint
import ms_mint_app

from . import tools as T
from . import messages

import dash_uploader as du


def make_dirs():
    tmpdir = tempfile.gettempdir()
    tmpdir = os.path.join(tmpdir, "MINT")
    tmpdir = os.getenv("MINT_DATA_DIR", default=tmpdir)
    cachedir = os.path.join(tmpdir, ".cache")
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(cachedir, exist_ok=True)
    print("MAKEDIRS:", tmpdir, cachedir)
    return P(tmpdir), P(cachedir)


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


pd.options.display.max_colwidth = 1000


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

# Collect outputs:
_outputs = html.Div(
    id="outputs",
    children=[plugin.outputs() for plugin in plugins.values() if plugin.outputs is not None],
    style={"visibility": "hidden"},
)

#logging.info(f'Outputs: {_outputs}')

logout_button = (
    dbc.Button(
        "Logout",
        id="logout-button",
        style={"marginRight": "10px", "visibility": "hidden"},
    ),
)
logout_button = html.A(href="/logout", children=logout_button)
SIDEBAR_WIDTH = "250px"
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'fontWeight': 'bold',
    'borderLeft': 'none'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
}


_layout = fac.AntdLayout(
    [
        dcc.Store(id="tmpdir", data=str(TMPDIR)),
        dcc.Store(id="wdir"),

        fac.AntdSider(
            [
                fac.AntdFlex([
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
                fac.AntdFlex([
                    fac.AntdText('Workspace:', strong=True),
                    fac.AntdCopyText(
                        id="ws-wdir-name",
                        locale='en-us',
                        beforeIcon=fac.AntdText(code=True, id="ws-wdir-name-text"),
                        afterIcon=fac.AntdIcon(icon='antd-like')
                    )],
                    justify='space-between',
                    align='center',
                    style={'height': '50px'},
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
                    style={'height': '100%', 'overflow': 'hidden auto'},
                    currentKey='Workspaces',
                    id='sidebar-menu',
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
    upload_dir = str(P(upload_root) / "MINT-Uploads")
    UPLOAD_FOLDER_ROOT = upload_dir
    du.configure_upload(app, UPLOAD_FOLDER_ROOT)

    messages.callbacks(app=app, fsc=fsc, cache=cache)

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
        Output('logout-menu', 'style'),
        Output('active-workspace-container', 'style'),
        Output('workspace-divider', 'style'),
        Input('sidebar', 'collapsed')
    )
    def toggle_sidebar(collapsed):
        logo_style = {'width': '100%', 'height': 'auto'} if collapsed else {'width': '50%', 'height': 'auto'}
        logout_menu_style = {'width': '100%', 'display': 'none'} if collapsed else {'width': '60px', 'display': 'none'}
        active_workspace_container_style = {'display': 'none'} if collapsed else {'display': 'flex'}
        ws_divider_style = {'display': 'none'} if collapsed else {'display': 'block'}

        return logo_style, logout_menu_style, active_workspace_container_style, ws_divider_style

    @app.callback(
        Output("tab-content", "children"),
        Input("tab", "value"),
        State("wdir", "data"),
    )
    def render_content(tab, wdir):
        func = plugins[tab].layout
        if tab != "Workspaces" and wdir == "":
            return dbc.Alert(
                "Please, create and activate a workspace.", color="warning"
            )
        elif (
            tab in ["Metadata", "Peak Optimization", "Processing"]
            and len(T.get_ms_fns(wdir)) == 0
        ):
            return dbc.Alert("Please import MS files.", color="warning")
        elif tab in ["Processing"] and (len(T.get_targets(wdir)) == 0):
            return dbc.Alert("Please, define targets.", color="warning")
        elif tab in ["Analysis"] and not P(T.get_results_fn(wdir)).is_file():
            return dbc.Alert("Please, create results (Processing).", color="warning")
        if func is not None:
            return func()
        else:
            raise PreventUpdate

    @app.callback(
        Output("tmpdir", "data"),
        Output("logout-button", "style"),
        Input("progress-interval", "n_intervals"),
    )
    def upate_tmpdir(x):
        if hasattr(app.server, "login_manager"):
            username = current_user.username
            tmpdir = str(TMPDIR / "User" / username)
            return tmpdir, {"visibility": "visible"}
        return str(TMPDIR / "Local"), {"visibility": "hidden"}
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

    app = dash.Dash(
        __name__,
        background_callback_manager=background_callback_manager,
        external_stylesheets=[
            dbc.themes.MINTY,
            "https://codepen.io/chriddyp/pen/bWLwgP.css",
        ],
        **kwargs,
    )

    app.layout = _layout
    app.title = "MINT"
    app.config["suppress_callback_exceptions"] = True

    upload_root = os.getenv("MINT_DATA_DIR", tempfile.gettempdir())
    CACHE_DIR = str(P(upload_root) / "MINT-Cache")

    logging.info('Defining filesystem cache')
    cache = Cache(
        app.server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": CACHE_DIR}
    )

    fsc = FileSystemCache(str(CACHEDIR))
    logging.info('Done creating app')
    return app, cache, fsc

