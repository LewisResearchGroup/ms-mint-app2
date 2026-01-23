import os
import sys
from pathlib import Path

from ms_mint_app.app import make_dirs, load_plugins


def test_make_dirs_respects_env(tmp_path, monkeypatch):
    mint_dir = tmp_path / "MINT"
    monkeypatch.setenv("MINT_DATA_DIR", str(mint_dir))

    tmpdir, cachedir = make_dirs()

    assert tmpdir == mint_dir
    assert cachedir == mint_dir / ".cache"
    assert tmpdir.exists()
    assert cachedir.exists()


def test_load_plugins_discovers_plugin(tmp_path, monkeypatch):
    pkg_dir = tmp_path / "dummy_plugins"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    plugin_module = pkg_dir / "dummy.py"
    plugin_module.write_text(
        """
from ms_mint_app.plugin_interface import PluginInterface

class DummyPlugin(PluginInterface):
    def __init__(self):
        self._label = "Dummy"
        self._order = 99

    def callbacks(self, app, fsc, cache):
        return None

    def layout(self):
        return "layout"

    @property
    def outputs(self):
        return []
"""
    )

    sys.path.insert(0, str(tmp_path))
    try:
        plugins = load_plugins(str(pkg_dir), "dummy_plugins")
        assert "Dummy" in plugins
        assert plugins["Dummy"].layout() == "layout"
    finally:
        sys.path.remove(str(tmp_path))
