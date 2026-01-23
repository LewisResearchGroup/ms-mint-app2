import os
import sys
from pathlib import Path

import ms_mint_app.prebuild_cache as prebuild_cache


def test_setup_bundled_matplotlib_cache_non_frozen(tmp_path, monkeypatch):
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("MINT_DATA_DIR", raising=False)

    if hasattr(sys, "_MEIPASS"):
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    prebuild_cache.setup_bundled_matplotlib_cache()

    assert os.environ.get("MPLCONFIGDIR") is None


def test_setup_bundled_matplotlib_cache_frozen(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets" / "matplotlib_cache_linux"
    assets_dir.mkdir(parents=True)
    (assets_dir / "fontlist-v390.json").write_text("{}")

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setenv("MINT_DATA_DIR", str(tmp_path / "MINT_DATA"))

    prebuild_cache.setup_bundled_matplotlib_cache()

    user_cache = Path(os.environ["MPLCONFIGDIR"])
    assert user_cache.exists()
    assert (user_cache / "fontlist-v390.json").exists()
