import contextlib

import duckdb
import pytest

from ms_mint_app.duckdb_manager import _create_tables
from ms_mint_app.plugins import target_optimization as topt


@contextlib.contextmanager
def _conn_context(conn):
    yield conn


def _seed_samples(conn):
    conn.execute(
        "INSERT INTO samples (ms_file_label, label, sample_type, ms_type, use_for_optimization) "
        "VALUES "
        "('S1', 'S1', 'TypeA', 'ms1', TRUE),"
        "('S2', 'S2', 'TypeA', 'ms1', TRUE),"
        "('S3', 'S3', 'TypeB', 'ms2', TRUE)"
    )


def test_update_sample_type_tree_requires_section():
    with pytest.raises(topt.PreventUpdate):
        topt._update_sample_type_tree({}, None, None, None, "all", "/tmp", "section-context")


def test_update_sample_type_tree_missing_wdir():
    tree_data, checked, expanded, show, hide = topt._update_sample_type_tree(
        {"page": "Optimization"},
        None,
        None,
        None,
        "all",
        None,
        "section-context",
    )

    assert tree_data == []
    assert checked == []
    assert expanded == []
    assert show["display"] == "none"
    assert hide["display"] == "block"


def test_update_sample_type_tree_no_chromatograms():
    tree_data, checked, expanded, show, hide = topt._update_sample_type_tree(
        {"page": "Optimization"},
        None,
        None,
        None,
        "all",
        "/tmp",
        "section-context",
        workspace_status={"chromatograms_count": 0},
    )

    assert tree_data == []
    assert checked == []
    assert expanded == []
    assert show["display"] == "none"
    assert hide["display"] == "block"


def test_update_sample_type_tree_db_error(monkeypatch):
    monkeypatch.setattr(topt, "duckdb_connection", lambda _wdir: _conn_context(None))

    tree_data, checked, expanded, show, hide = topt._update_sample_type_tree(
        {"page": "Optimization"},
        None,
        None,
        None,
        "all",
        "/tmp",
        "section-context",
        workspace_status={"chromatograms_count": 1},
    )

    assert tree_data == []
    assert checked == []
    assert expanded == []
    assert show["display"] == "none"
    assert hide["display"] == "block"


def test_update_sample_type_tree_mark_action(monkeypatch):
    conn = duckdb.connect(":memory:")
    _create_tables(conn)
    _seed_samples(conn)

    monkeypatch.setattr(topt, "duckdb_connection", lambda _wdir: _conn_context(conn))

    tree_data, checked, expanded, show, hide = topt._update_sample_type_tree(
        {"page": "Optimization"},
        None,
        None,
        None,
        "ms1",
        "/tmp",
        "mark-tree-action",
        workspace_status={"chromatograms_count": 1},
    )

    assert tree_data is topt.dash.no_update
    assert expanded is topt.dash.no_update
    assert sorted(checked) == ["S1", "S2"]
    assert show["display"] == "flex"
    assert hide["display"] == "none"


def test_update_sample_type_tree_section_context(monkeypatch):
    conn = duckdb.connect(":memory:")
    _create_tables(conn)
    _seed_samples(conn)

    monkeypatch.setattr(topt, "duckdb_connection", lambda _wdir: _conn_context(conn))
    monkeypatch.setattr(topt, "proportional_min1_selection", lambda *_a, **_k: (None, ["S1"]))

    tree_data, checked, expanded, show, hide = topt._update_sample_type_tree(
        {"page": "Optimization"},
        None,
        None,
        None,
        "ms1",
        "/tmp",
        "section-context",
        workspace_status={"chromatograms_count": 1},
    )

    assert isinstance(tree_data, list)
    assert checked == ["S1"]
    assert expanded is topt.dash.no_update
    assert show["display"] == "flex"
    assert hide["display"] == "none"
