import os
import sys
from types import SimpleNamespace

import pytest


def _install_stub_modules(monkeypatch, tmp_path):
    class _AppStub:
        def __init__(self):
            self.css = SimpleNamespace(config=SimpleNamespace(serve_locally=False))
            self.scripts = SimpleNamespace(config=SimpleNamespace(serve_locally=False))
            self.server = object()

        def run(self, **_kwargs):
            return None

    app_stub = _AppStub()

    def _create_app():
        return app_stub, object(), object(), object(), object()

    def _register_callbacks(*_args, **_kwargs):
        return None

    app_module = SimpleNamespace(
        create_app=_create_app,
        register_callbacks=_register_callbacks,
    )

    class _SingleInstance:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def ensure_single(self, force=False):
            return None

    instance_module = SimpleNamespace(SingleInstance=_SingleInstance)

    monkeypatch.setitem(sys.modules, "ms_mint_app.app", app_module)
    monkeypatch.setitem(sys.modules, "ms_mint_app.instance", instance_module)
    monkeypatch.setitem(sys.modules, "waitress", SimpleNamespace(serve=lambda *_a, **_k: None))

    return app_stub


def test_main_creates_default_config(monkeypatch, tmp_path):
    from ms_mint_app.scripts import Mint

    config_path = tmp_path / "mint_config.json"
    home_dir = tmp_path / "home"
    home_dir.mkdir()

    _install_stub_modules(monkeypatch, tmp_path)

    monkeypatch.setattr(Mint, "init_global_logging", lambda: None)
    monkeypatch.setattr(Mint, "serve", lambda *_a, **_k: None)
    monkeypatch.setattr(Mint, "expanduser", lambda *_a, **_k: str(home_dir))
    monkeypatch.setattr(Mint.P, "home", lambda: Mint.P(home_dir))
    monkeypatch.setenv("MINT_CONFIG_PATH", str(config_path))

    argv = [
        "Mint.py",
        "--no-browser",
        "--port",
        "0",
        "--host",
        "127.0.0.1",
        "--config",
        str(config_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    Mint.main()

    assert config_path.exists()
    assert os.environ.get("MINT_DATA_DIR")


def test_main_prefers_config_data_dir(monkeypatch, tmp_path):
    from ms_mint_app.scripts import Mint

    config_path = tmp_path / "mint_config.json"
    data_dir = tmp_path / "custom_data"
    data_dir.mkdir()

    config_path.write_text(
        '{"repo_path": null, "fallback_repo_path": null, "data_dir": "%s"}'
        % data_dir.as_posix(),
        encoding="utf-8",
    )

    _install_stub_modules(monkeypatch, tmp_path)

    monkeypatch.setattr(Mint, "init_global_logging", lambda: None)
    monkeypatch.setattr(Mint, "serve", lambda *_a, **_k: None)
    monkeypatch.setattr(Mint, "expanduser", lambda *_a, **_k: str(tmp_path / "home"))
    monkeypatch.setattr(Mint.P, "home", lambda: Mint.P(tmp_path / "home"))

    argv = [
        "Mint.py",
        "--no-browser",
        "--port",
        "0",
        "--host",
        "127.0.0.1",
        "--config",
        str(config_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    Mint.main()

    assert os.environ.get("MINT_DATA_DIR") == data_dir.as_posix()
