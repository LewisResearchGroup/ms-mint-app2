#!/usr/bin/env python

import os
import sys
import subprocess
import argparse
import importlib.metadata as importlib_metadata
import logging
import json
from urllib.parse import urlparse

from waitress import serve
from os.path import expanduser
from pathlib import Path as P
from collections import namedtuple
from multiprocessing import freeze_support

import ms_mint_app


def _looks_like_remote(target: str) -> bool:
    return target.startswith(("http://", "https://", "git@", "ssh://", "git+"))


def _normalize_remote_target(target: str) -> str:
    if target.startswith("git+"):
        return target

    if target.startswith("https://github.com/") and "/tree/" in target:
        base, _, branch = target.partition("/tree/")
        branch = branch.strip("/")
        if branch:
            if not base.endswith(".git"):
                base = base.rstrip("/") + ".git"
            return f"git+{base}@{branch}"
    return target


def update_repo(repo_path: str, install: bool = True) -> bool:
    """Try to update/install a repo from a local path or remote URL."""
    if not repo_path:
        return False

    repo_path = repo_path.strip()

    if _looks_like_remote(repo_path):
        target = _normalize_remote_target(repo_path)
        logging.info("Installing from remote source %s", target)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", target],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return True
        except subprocess.CalledProcessError as exc:
            output = exc.stdout.decode(errors="ignore") if exc.stdout else str(exc)
            logging.warning("Remote install failed for %s: %s", target, output)
            return False

    repo_path = os.path.expanduser(repo_path)
    if not os.path.isdir(repo_path):
        logging.warning("Repo path %s does not exist", repo_path)
        return False

    logging.info("Updating repo at %s", repo_path)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", repo_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        output = exc.stdout.decode(errors="ignore") if exc.stdout else str(exc)
        logging.warning("Failed for %s: %s", repo_path, output)
        return False

    if install:
        logging.info("Installing %s in editable mode", repo_path)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", repo_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            output = exc.stdout.decode(errors="ignore") if exc.stdout else str(exc)
            logging.warning("pip install -e %s failed: %s", repo_path, output)
            return False

    return True


def _can_import_ms_mint_app() -> bool:
    """Check whether ms_mint_app imports cleanly after an update."""
    try:
        import importlib

        importlib.import_module("ms_mint_app.app")
        return True
    except Exception as exc:  # noqa: BLE001
        logging.warning("Post-update import check failed: %s", exc)
        return False


welcome = r"""
 __________________________________________________________________________________________________________
/___/\\\\____________/\\\\__/\\\\\\\\\\\__/\\\\\_____/\\\__/\\\\\\\\\\\\\\\_______________/\\\_____________\
|___\/\\\\\\________/\\\\\\_\/////\\\///__\/\\\\\\___\/\\\_\///////\\\/////______________/\\\\\\\__________|
|____\/\\\//\\\____/\\\//\\\_____\/\\\_____\/\\\/\\\__\/\\\_______\/\\\__________________/\\\\\\\\\________|
|_____\/\\\\///\\\/\\\/_\/\\\_____\/\\\_____\/\\\//\\\_\/\\\_______\/\\\_________________\//\\\\\\\________|
|______\/\\\__\///\\\/___\/\\\_____\/\\\_____\/\\\\//\\\\/\\\_______\/\\\__________________\//\\\\\________|
|_______\/\\\____\///_____\/\\\_____\/\\\_____\/\\\_\//\\\/\\\_______\/\\\___________________\//\\\________|
|________\/\\\_____________\/\\\_____\/\\\_____\/\\\__\//\\\\\\_______\/\\\____________________\///________|
|_________\/\\\_____________\/\\\__/\\\\\\\\\\\_\/\\\___\//\\\\\_______\/\\\_____________________/\\\______|
|__________\///______________\///__\///////////__\///_____\/////________\///_____________________\///______|
\__________________________________________________________________________________________________________/
       \
        \   ^__^
         \  (@@)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
"""


def _create_get_distribution(is_frozen, true_get_distribution, _Dist):
    def _get_distribution(dist):
        if is_frozen and dist == "flask-compress":
            return _Dist("1.5.0")
        try:
            return _Dist(true_get_distribution(dist))
        except importlib_metadata.PackageNotFoundError:
            return _Dist("0.0.0")
    return _get_distribution


def main():
    freeze_support()

    HOME = expanduser("~")
    DATADIR = str(P(HOME) / "MINT")
    
    # Define local variables
    is_frozen = hasattr(sys, "_MEIPASS")
    true_get_distribution = importlib_metadata.version
    _Dist = namedtuple("_Dist", ["version"])
    
    # Create the distribution getter function with the necessary context
    get_distribution = _create_get_distribution(is_frozen, true_get_distribution, _Dist)

    parser = argparse.ArgumentParser(description="MINT frontend.")

    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="do not start the browser",
    )
    parser.add_argument(
        "--version", default=False, action="store_true", help="print current version"
    )
    parser.add_argument(
        "--data-dir", default=DATADIR, help="target directory for MINT data"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="start MINT server in debug mode",
    )
    parser.add_argument("--port", type=int, default=9999, help="Port to use")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host binding address"
    )
    parser.add_argument(
        "--serve-path",
        default=None,
        type=str,
        help="(deprecated) serve app at a different path e.g. '/mint/' to serve the app at 'localhost:9999/mint/'",
    )
    parser.add_argument(
        "--ncpu",
        default=None,
        type=int,
        help='Number of CPUs to use',
    )
    parser.add_argument(
        "--local",
        default=True,
        action="store_true",
        help="run MINT locally and use the File System Access API to get local files without uploading them to the "
             "server",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("MINT_CONFIG_PATH") or str(P(expanduser("~"), ".mint_config.json")),
        help="Path to JSON config with repo settings (auto-created if missing)",
    )
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path or VCS URL for ms-mint-app to update before launching",
    )
    parser.add_argument(
        "--fallback-repo-path",
        default=None,
        help="Fallback repository path to try if the primary update fails",
    )
    parser.add_argument(
        "--skip-update",
        default=True,
        action="store_true",
        help="Skip updating repositories before launching the app",
    )

    args = parser.parse_args()

    if args.version:
        print("Mint version:", ms_mint_app.__version__)
        exit()

    url = f"http://{args.host}:{args.port}"

    def wait_and_open_browser():
        import socket
        import time
        import webbrowser

        def wait_for_server(host, port, timeout=30):
            start_time = time.time()
            while True:
                try:
                    with socket.create_connection((host, port), timeout=1):
                        return True
                except OSError:
                    time.sleep(0.5)
                    if time.time() - start_time > timeout:
                        return False

        if wait_for_server(args.host, args.port):
            try:
                if sys.platform in ["win32", "nt"]:
                    os.startfile(url)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", url])
                else:
                    subprocess.Popen(["xdg-open", url])
            except Exception:
                print(f"Please open your browser manually: {url}")

    if not args.no_browser:
        import threading
        threading.Thread(target=wait_and_open_browser).start()

    if args.data_dir is not None:
        os.environ["MINT_DATA_DIR"] = args.data_dir

    if args.serve_path is not None:
        os.environ["MINT_SERVE_PATH"] = args.serve_path

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    config_repo = None
    config_fallback = None

    if args.config:
        config_path = os.path.expanduser(args.config)
        if not os.path.isfile(config_path):
            user_home = P.home()
            default_fallback = str(user_home / "ms-mint-app")
            default_cfg = {
                "repo_path": "git+https://github.com/Valdes-Tresanco-MS/ms-mint-app@db-migration",
                "fallback_repo_path": default_fallback,
            }
            try:
                with open(config_path, "w", encoding="utf-8") as fh:
                    json.dump(default_cfg, fh, indent=2)
                logging.info("Created default config at %s", config_path)
            except OSError as exc:
                logging.warning("Unable to create config file %s: %s", config_path, exc)
        else:
            logging.info("Using config file %s", config_path)

        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                config_repo = cfg.get("repo_path")
                config_fallback = cfg.get("fallback_repo_path")
            except Exception as exc:
                logging.warning("Could not read config %s: %s", config_path, exc)

    env_repo = os.environ.get("MINT_REPO_PATH")
    env_fallback = os.environ.get("MINT_FALLBACK_REPO_PATH")

    repo_path = args.repo_path or config_repo or env_repo
    fallback_repo = args.fallback_repo_path or config_fallback or env_fallback

    updated = False
    if not args.skip_update:
        if repo_path:
            updated = update_repo(repo_path)
        if not updated and fallback_repo:
            updated = update_repo(fallback_repo)
    else:
        logging.info("Skipping repository updates per --skip-update")

    if updated and not os.environ.get("MINT_ALREADY_UPDATED"):
        if _can_import_ms_mint_app():
            logging.info("Update succeeded; restarting to load fresh code")
            os.environ["MINT_ALREADY_UPDATED"] = "1"
            if is_frozen:
                os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
            else:
                os.execv(
                    sys.executable,
                    [sys.executable, "-m", "ms_mint_app.scripts.Mint", *sys.argv[1:]],
                )
        else:
            logging.warning("Update reported success but import failed; skipping restart; proceeding with existing install")

    print(welcome)
    print("Loading app...")

    from ms_mint_app.app import create_app, register_callbacks

    app, cache, fsc = create_app()
    register_callbacks(app, cache, fsc, args)

    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True

    print("Configuration done starting server...")

    if args.debug:
        app.run(
            debug=args.debug,
            port=args.port,
            host=args.host,
            dev_tools_hot_reload=False,
            dev_tools_hot_reload_interval=3000,
            dev_tools_hot_reload_max_retry=30,
        )
    else:
        serve(app.server, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
