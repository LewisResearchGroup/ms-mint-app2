import contextlib

import duckdb
import pytest

from ms_mint_app.duckdb_manager import _create_tables
from ms_mint_app.plugins import processing as proc


@contextlib.contextmanager
def _conn_context(conn):
    yield conn


def test_build_delete_modal_requires_selection(monkeypatch):
    monkeypatch.setattr(proc.fac, "AntdFlex", lambda children, vertical=False: {"children": children})
    monkeypatch.setattr(proc.fac, "AntdText", lambda text, **kwargs: {"text": text, **kwargs})

    with pytest.raises(proc.PreventUpdate):
        proc._build_delete_modal_content("processing-delete-selected", [])


def test_build_delete_modal_selected(monkeypatch):
    monkeypatch.setattr(proc.fac, "AntdFlex", lambda children, vertical=False: {"children": children})
    monkeypatch.setattr(proc.fac, "AntdText", lambda text, **kwargs: {"text": text, **kwargs})

    visible, content = proc._build_delete_modal_content(
        "processing-delete-selected",
        [{"peak_label": "Peak1"}],
    )

    assert visible is True
    assert "selected results" in content["children"][1]["text"]


def test_build_delete_modal_all(monkeypatch):
    monkeypatch.setattr(proc.fac, "AntdFlex", lambda children, vertical=False: {"children": children})
    monkeypatch.setattr(proc.fac, "AntdText", lambda text, **kwargs: {"text": text, **kwargs})

    visible, content = proc._build_delete_modal_content("processing-delete-all", [])

    assert visible is True
    assert "ALL results" in content["children"][1]["text"]


def test_load_peaks_from_results_validation(monkeypatch):
    with pytest.raises(proc.PreventUpdate):
        proc._load_peaks_from_results("", [])

    monkeypatch.setattr(proc, "duckdb_connection", lambda _wdir: _conn_context(None))
    with pytest.raises(proc.PreventUpdate):
        proc._load_peaks_from_results("/tmp", [])


def test_load_peaks_from_results_options(monkeypatch):
    conn = duckdb.connect(":memory:")
    _create_tables(conn)
    conn.execute(
        "INSERT INTO results (peak_label, ms_file_label, peak_area) VALUES ('Peak1', 'S1', 1.0), ('Peak2', 'S2', 2.0)"
    )

    monkeypatch.setattr(proc, "duckdb_connection", lambda _wdir: _conn_context(conn))

    options, selected = proc._load_peaks_from_results("/tmp", None)

    assert options[0]["value"] == "Peak1"
    assert selected == ["Peak1"]


def test_load_peaks_from_results_filters_invalid(monkeypatch):
    conn = duckdb.connect(":memory:")
    _create_tables(conn)
    conn.execute(
        "INSERT INTO results (peak_label, ms_file_label, peak_area) VALUES ('Peak1', 'S1', 1.0)"
    )

    monkeypatch.setattr(proc, "duckdb_connection", lambda _wdir: _conn_context(conn))

    options, selected = proc._load_peaks_from_results("/tmp", ["Missing", "Peak1"])

    assert selected == ["Peak1"]
