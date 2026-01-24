from pathlib import Path

import duckdb

from ms_mint_app.duckdb_manager import (
    _create_tables,
    duckdb_connection,
    compute_and_insert_chromatograms_from_ms_data,
    compute_chromatograms_in_batches,
    compute_results_in_batches,
    populate_full_range_downsampled_chromatograms,
    populate_full_range_downsampled_chromatograms_for_target,
)


def _make_workspace(tmp_path):
    base = tmp_path / "mint"
    wdir = base / "workspaces" / "1"
    wdir.mkdir(parents=True)
    return wdir


def test_compute_and_insert_chromatograms_no_targets():
    conn = duckdb.connect(":memory:")
    _create_tables(conn)

    compute_and_insert_chromatograms_from_ms_data(conn)

    assert conn.execute("SELECT COUNT(*) FROM chromatograms").fetchone()[0] == 0


def test_compute_chromatograms_in_batches_no_pairs(tmp_path):
    wdir = _make_workspace(tmp_path)

    with duckdb_connection(wdir, register_activity=False) as conn:
        conn.execute(
            "INSERT INTO ms1_data (ms_file_label, scan_id, mz, intensity, scan_time) "
            "VALUES ('S1', 1, 100.0, 10.0, 1.0)"
        )

    result = compute_chromatograms_in_batches(
        str(wdir),
        use_for_optimization=True,
        batch_size=100,
    )

    assert result == {"total_pairs": 0, "processed": 0, "failed": 0, "batches": 0}


def test_compute_results_in_batches_no_pairs(tmp_path):
    wdir = _make_workspace(tmp_path)

    result = compute_results_in_batches(
        str(wdir),
        batch_size=100,
    )

    assert result == {"total_pairs": 0, "processed": 0, "failed": 0, "batches": 0}


def test_populate_full_range_downsampled_chromatograms_no_lttb(monkeypatch, tmp_path):
    wdir = _make_workspace(tmp_path)
    calls = []

    monkeypatch.setattr("ms_mint_app.duckdb_manager._lttbc", None)
    monkeypatch.setattr(
        "ms_mint_app.duckdb_manager._send_progress",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    populate_full_range_downsampled_chromatograms(str(wdir), set_progress=lambda *_a, **_k: None)

    assert any(call[1].get("detail") == "lttbc not available" for call in calls)


def test_populate_full_range_downsampled_chromatograms_for_target_no_lttb(monkeypatch, tmp_path):
    wdir = _make_workspace(tmp_path)

    monkeypatch.setattr("ms_mint_app.duckdb_manager._lttbc", None)

    assert populate_full_range_downsampled_chromatograms_for_target(str(wdir), "Peak1") is False


def test_compute_results_in_batches_basic(tmp_path):
    wdir = _make_workspace(tmp_path)

    with duckdb_connection(wdir, register_activity=False) as conn:
        conn.execute(
            "INSERT INTO targets (peak_label, rt_min, rt_max, rt_unit) VALUES ('Peak1', 0.0, 2.0, 's')"
        )
        conn.execute(
            "INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, mz_arr, ms_type) "
            "VALUES ('Peak1', 'S1', [0.0,1.0,2.0], [1.0,2.0,3.0], [100.0,101.0,102.0], 'ms1')"
        )

    result = compute_results_in_batches(str(wdir), batch_size=10, recompute=True)

    assert result["processed"] == 1

    with duckdb_connection(wdir, register_activity=False) as conn:
        row = conn.execute(
            "SELECT peak_area, peak_max, peak_mz_of_max, peak_n_datapoints FROM results WHERE peak_label='Peak1'"
        ).fetchone()

    assert row == (4.0, 3.0, 102.0, 3)


def test_compute_chromatograms_in_batches_minimal(tmp_path):
    wdir = _make_workspace(tmp_path)

    with duckdb_connection(wdir, register_activity=False) as conn:
        conn.execute(
            "INSERT INTO samples (ms_file_label, ms_type, use_for_optimization) VALUES ('S1', 'ms1', TRUE)"
        )
        conn.execute(
            "INSERT INTO targets (peak_label, ms_type, mz_mean, mz_width, rt_min, rt_max, rt_unit, peak_selection) "
            "VALUES ('Peak1', 'ms1', 100.0, 10.0, 0.0, 2.0, 's', TRUE)"
        )
        conn.execute(
            "INSERT INTO ms1_data (ms_file_label, scan_id, mz, intensity, scan_time) "
            "VALUES ('S1', 1, 100.0, 10.0, 1.0)"
        )

    compute_chromatograms_in_batches(
        str(wdir),
        use_for_optimization=True,
        batch_size=10,
    )

    with duckdb_connection(wdir, register_activity=False) as conn:
        count = conn.execute("SELECT COUNT(*) FROM chromatograms").fetchone()[0]
    assert count >= 1
