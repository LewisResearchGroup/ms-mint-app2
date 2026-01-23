import contextlib

import duckdb

from ms_mint_app.duckdb_manager import _create_tables
from ms_mint_app.plugins import target_optimization as topt


@contextlib.contextmanager
def _conn_context(conn):
    yield conn


def _seed_target_rows(conn):
    conn.execute(
        "INSERT INTO samples (ms_file_label, color, label, sample_type, use_for_optimization) "
        "VALUES ('S1', '#ff0000', 'Sample1', 'TypeA', TRUE)"
    )
    conn.execute(
        "INSERT INTO targets (peak_label, ms_type, bookmark) VALUES ('Peak1', 'ms1', FALSE)"
    )
    conn.execute(
        "INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type) "
        "VALUES ('Peak1', 'S1', [1.0], [10.0], 'ms1')"
    )
    conn.execute(
        "INSERT INTO results (peak_label, ms_file_label, peak_area) VALUES ('Peak1', 'S1', 123.0)"
    )


def test_delete_target_logic_success(monkeypatch):
    conn = duckdb.connect(":memory:")
    _create_tables(conn)
    _seed_target_rows(conn)

    monkeypatch.setattr(topt, "duckdb_connection", lambda _wdir: _conn_context(conn))
    monkeypatch.setattr(topt.fac, "AntdNotification", lambda **kwargs: kwargs)

    notice, refreshed, open_modal, open_confirm = topt._delete_target_logic("Peak1", "/tmp")

    assert refreshed is True
    assert open_modal is False
    assert open_confirm is False
    assert notice.get("type") == "success"

    assert conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chromatograms").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0


def test_delete_target_logic_missing_db(monkeypatch):
    monkeypatch.setattr(topt, "duckdb_connection", lambda _wdir: _conn_context(None))
    monkeypatch.setattr(topt.fac, "AntdNotification", lambda **kwargs: kwargs)

    notice, refreshed, open_modal, open_confirm = topt._delete_target_logic("Peak1", "/tmp")

    assert notice.get("type") == "error"
    assert refreshed is topt.dash.no_update
    assert open_modal is False
    assert open_confirm is False


def test_bookmark_target_logic_updates(monkeypatch):
    conn = duckdb.connect(":memory:")
    _create_tables(conn)
    _seed_target_rows(conn)

    monkeypatch.setattr(topt, "duckdb_connection", lambda _wdir: _conn_context(conn))
    monkeypatch.setattr(topt.fac, "AntdNotification", lambda **kwargs: kwargs)

    notice = topt._bookmark_target_logic([True], ["Peak1"], 0, "/tmp")

    assert notice.get("type") == "success"
    assert conn.execute("SELECT bookmark FROM targets WHERE peak_label='Peak1'").fetchone()[0] is True


def test_toggle_bookmark_logic(monkeypatch):
    conn = duckdb.connect(":memory:")
    _create_tables(conn)
    _seed_target_rows(conn)

    monkeypatch.setattr(topt, "duckdb_connection", lambda _wdir: _conn_context(conn))
    monkeypatch.setattr(topt.fac, "AntdIcon", lambda **kwargs: kwargs)

    icon = topt._toggle_bookmark_logic("Peak1", "/tmp")

    assert icon.get("style", {}).get("color") == "gold"
    assert conn.execute("SELECT bookmark FROM targets WHERE peak_label='Peak1'").fetchone()[0] is True


def test_cpu_ram_help_texts(monkeypatch):
    monkeypatch.setattr(topt, "cpu_count", lambda: 8)

    class _VMem:
        available = 16 * 1024 ** 3

    monkeypatch.setattr(topt.psutil, "virtual_memory", lambda: _VMem())

    assert topt._get_cpu_help_text(2) == "Selected 2 / 8 cpus"
    assert topt._get_ram_help_text(4) == "Selected 4GB / 16.0GB available RAM"


def test_bookmark_target_logic_missing_db(monkeypatch):
    monkeypatch.setattr(topt, "duckdb_connection", lambda _wdir: _conn_context(None))
    monkeypatch.setattr(topt.fac, "AntdNotification", lambda **kwargs: kwargs)

    notice = topt._bookmark_target_logic([True], ["Peak1"], 0, "/tmp")

    assert notice.get("type") == "error"


def test_compute_chromatograms_logic_success(monkeypatch):
    conn = object()
    calls = {"compute": None, "optimize": None, "progress": []}

    def fake_compute(wdir, use_for_optimization, batch_size, set_progress, recompute_ms1, recompute_ms2, n_cpus, ram):
        calls["compute"] = {
            "wdir": wdir,
            "use_for_optimization": use_for_optimization,
            "batch_size": batch_size,
            "recompute_ms1": recompute_ms1,
            "recompute_ms2": recompute_ms2,
            "n_cpus": n_cpus,
            "ram": ram,
        }

    def fake_optimize(conn_arg):
        calls["optimize"] = conn_arg
        return 3

    def fake_progress(payload):
        calls["progress"].append(payload)

    monkeypatch.setattr(topt, "duckdb_connection", lambda *args, **kwargs: _conn_context(conn))
    monkeypatch.setattr(topt, "compute_chromatograms_in_batches", fake_compute)
    monkeypatch.setattr(topt, "optimize_rt_spans_batch", fake_optimize)
    monkeypatch.setattr(topt, "activate_workspace_logging", lambda _wdir: None)

    result = topt._compute_chromatograms_logic(
        set_progress=fake_progress,
        recompute_ms1=True,
        recompute_ms2=False,
        n_cpus=2,
        ram=4,
        batch_size=10,
        wdir="/tmp",
    )

    assert result == (True, False)
    assert calls["compute"]["use_for_optimization"] is True
    assert calls["compute"]["batch_size"] == 10
    assert calls["compute"]["recompute_ms1"] is True
    assert calls["compute"]["recompute_ms2"] is False
    assert calls["optimize"] is conn
    percents = [payload[0] for payload in calls["progress"]]
    assert 0 in percents
    assert 95 in percents


def test_compute_chromatograms_logic_missing_db(monkeypatch):
    monkeypatch.setattr(topt, "duckdb_connection", lambda *args, **kwargs: _conn_context(None))
    monkeypatch.setattr(topt, "activate_workspace_logging", lambda _wdir: None)

    result = topt._compute_chromatograms_logic(
        set_progress=None,
        recompute_ms1=False,
        recompute_ms2=False,
        n_cpus=1,
        ram=1,
        batch_size=1,
        wdir="/tmp",
    )

    assert result == "Could not connect to database."
