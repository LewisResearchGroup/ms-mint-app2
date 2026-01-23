import contextlib
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from ms_mint_app.duckdb_manager import _create_tables
from ms_mint_app.plugins import processing as proc


@contextlib.contextmanager
def _conn_context(conn):
    yield conn


def _seed_results(conn):
    conn.execute(
        "INSERT INTO samples (ms_file_label, ms_type) VALUES ('S1', 'ms1'), ('S2', 'ms1')"
    )
    conn.execute(
        "INSERT INTO results (peak_label, ms_file_label, peak_area) "
        "VALUES ('Peak1', 'S1', 10.0), ('Peak1', 'S2', 11.0), ('Peak2', 'S1', 12.0)"
    )


def test_download_all_results_requires_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(proc.fac, "AntdNotification", lambda **kwargs: kwargs)

    download, notice = proc._download_all_results(str(tmp_path), "WS", [])

    assert download is proc.dash.no_update
    assert notice.get("type") == "warning"


def test_download_all_results_invalid_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(proc.fac, "AntdNotification", lambda **kwargs: kwargs)

    download, notice = proc._download_all_results(str(tmp_path), "WS", ["not_a_column"])

    assert download is proc.dash.no_update
    assert notice.get("type") == "warning"


def test_download_all_results_fallback_small_file(monkeypatch, tmp_path):
    tmp_csv = tmp_path / "results.csv"
    tmp_csv.write_text("peak_label,ms_file_label,peak_area\nPeak1,S1,1\n")

    monkeypatch.setattr(proc, "_generate_csv_from_db", lambda _wdir, _ws, _cols: str(tmp_csv))
    monkeypatch.setattr(proc.dcc, "send_file", lambda path, filename: {"path": path, "filename": filename})
    monkeypatch.setattr(proc.T, "today", lambda: "2026-01-23")

    download, notice = proc._download_all_results(str(tmp_path), "WS", ["peak_area"])

    assert notice is proc.dash.no_update
    assert download["path"] == str(tmp_csv)
    assert download["filename"].startswith("2026-01-23-MINT__WS")


def test_download_dense_matrix_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(proc.fac, "AntdNotification", lambda **kwargs: kwargs)

    download, notice = proc._download_dense_matrix(str(tmp_path), "WS", [], [], [])

    assert download is proc.dash.no_update
    assert notice.get("type") == "warning"


def test_download_dense_matrix_success(monkeypatch, tmp_path):
    monkeypatch.setattr(proc, "duckdb_connection", lambda _wdir: _conn_context(object()))
    monkeypatch.setattr(proc, "create_pivot", lambda *_a, **_k: pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(proc.dcc, "send_data_frame", lambda func, filename, **kwargs: {"filename": filename})
    monkeypatch.setattr(proc.T, "today", lambda: "2026-01-23")

    download, notice = proc._download_dense_matrix(
        str(tmp_path),
        "WS",
        rows=["ms_file_label"],
        cols=["peak_label"],
        value=["peak_area"],
    )

    assert notice is proc.dash.no_update
    assert download["filename"] == "2026-01-23-MINT__WS-peak_area_results.csv"


def test_delete_selected_results_success(monkeypatch, tmp_path):
    conn = duckdb.connect(":memory:")
    _create_tables(conn)
    _seed_results(conn)

    results_dir = Path(tmp_path) / "results"
    results_dir.mkdir()
    backup = results_dir / "results_backup.csv"
    backup.write_text("backup")

    monkeypatch.setattr(proc, "duckdb_connection", lambda _wdir: _conn_context(conn))

    selected = [
        {"peak_label": "Peak1", "ms_file_label": "S1"},
        {"peak_label": "Peak1", "ms_file_label": "S2"},
    ]

    notice, action, removed = proc._delete_selected_results(str(tmp_path), selected)

    assert notice is None
    assert action["status"] == "success"
    assert removed == [1, 2]
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1
    assert not backup.exists()


def test_delete_selected_results_empty():
    notice, action, removed = proc._delete_selected_results("/tmp", [])

    assert notice is None
    assert action["status"] == "failed"
    assert removed == [0, 0]


def test_delete_all_results_success(monkeypatch, tmp_path):
    conn = duckdb.connect(":memory:")
    _create_tables(conn)
    _seed_results(conn)

    results_dir = Path(tmp_path) / "results"
    results_dir.mkdir()
    backup = results_dir / "results_backup.csv"
    backup.write_text("backup")

    monkeypatch.setattr(proc, "duckdb_connection", lambda _wdir: _conn_context(conn))

    notice, action, removed = proc._delete_all_results(str(tmp_path))

    assert notice is None
    assert action["status"] == "success"
    assert removed == [2, 2]
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0
    assert not backup.exists()


def test_delete_all_results_missing_db(monkeypatch):
    monkeypatch.setattr(proc, "duckdb_connection", lambda _wdir: _conn_context(None))

    with pytest.raises(proc.PreventUpdate):
        proc._delete_all_results("/tmp")
