import importlib

from ms_mint_app.scripts.Mint import (
    _looks_like_remote,
    _normalize_remote_target,
    update_repo,
    _can_import_ms_mint_app,
)


def test_looks_like_remote():
    assert _looks_like_remote("https://github.com/foo/bar")
    assert _looks_like_remote("git@github.com:foo/bar.git")
    assert not _looks_like_remote("/local/path")


def test_normalize_remote_target():
    url = "https://github.com/foo/bar/tree/dev"
    normalized = _normalize_remote_target(url)
    assert normalized == "git+https://github.com/foo/bar.git@dev"

    already = "git+https://github.com/foo/bar.git@dev"
    assert _normalize_remote_target(already) == already


def test_update_repo_invalid_path(tmp_path):
    assert update_repo("") is False
    assert update_repo(str(tmp_path / "missing")) is False


def test_can_import_ms_mint_app(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda name: None)
    assert _can_import_ms_mint_app() is True

    def _raise(name):
        raise RuntimeError("fail")

    monkeypatch.setattr(importlib, "import_module", _raise)
    assert _can_import_ms_mint_app() is False
