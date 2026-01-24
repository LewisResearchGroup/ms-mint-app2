import numpy as np

import duckdb

from ms_mint_app.plugins.target_optimization import (
    downsample_for_preview,
    _calc_y_range_numpy,
    get_chromatogram_dataframe,
)
from ms_mint_app.duckdb_manager import _create_tables


def test_downsample_for_preview_no_change():
    scan_time = np.arange(50)
    intensity = np.arange(50) * 2

    out_time, out_intensity = downsample_for_preview(scan_time, intensity, max_points=100)

    assert np.array_equal(out_time, scan_time)
    assert np.array_equal(out_intensity, intensity)


def test_downsample_for_preview_reduces():
    scan_time = np.arange(1000)
    intensity = np.arange(1000) * 2

    out_time, out_intensity = downsample_for_preview(scan_time, intensity, max_points=100)

    assert len(out_time) == 100
    assert len(out_intensity) == 100
    assert out_time[0] == 0
    assert out_time[-1] == 999


def test_calc_y_range_numpy_linear():
    data = [{"x": [0, 1, 2], "y": [10, 5, 15]}]

    y_range = _calc_y_range_numpy(data, 0, 2, is_log=False)

    assert y_range == [0, 15 * 1.05]


def test_calc_y_range_numpy_log_fallback():
    data = [{"x": [0, 1, 2], "y": [0.5, 0.8, 1.0]}]

    y_range = _calc_y_range_numpy(data, 0, 2, is_log=True)

    assert y_range[0] < y_range[1]


def test_get_chromatogram_dataframe_cached():
    con = duckdb.connect(":memory:")
    _create_tables(con)

    con.execute(
        "INSERT INTO samples (ms_file_label, color, label, sample_type, use_for_optimization) VALUES ('S1', '#ff0000', 'Sample1', 'TypeA', TRUE)"
    )
    con.execute(
        "INSERT INTO targets (peak_label, intensity_threshold, ms_type) VALUES ('Peak1', 6.0, 'ms1')"
    )
    con.execute(
        "INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type) VALUES ('Peak1', 'S1', [1.0,2.0,3.0], [5.0,8.0,10.0], 'ms1')"
    )

    df = get_chromatogram_dataframe(con, "Peak1", full_range=False)

    assert df is not None
    row = df.to_dicts()[0]
    assert row["ms_file_label"] == "S1"
    assert row["scan_time_sliced"] == [2.0, 3.0]
    assert row["intensity_sliced"] == [8.0, 10.0]


def test_get_chromatogram_dataframe_full_range_from_cache():
    con = duckdb.connect(":memory:")
    _create_tables(con)

    con.execute(
        "INSERT INTO samples (ms_file_label, color, label, sample_type, use_for_optimization, ms_type) "
        "VALUES ('S1', '#ff0000', 'Sample1', 'TypeA', TRUE, 'ms1')"
    )
    con.execute(
        "INSERT INTO targets (peak_label, ms_type, mz_mean, mz_width) VALUES ('Peak1', 'ms1', 100.0, 10.0)"
    )
    con.execute(
        "INSERT INTO chromatograms (peak_label, ms_file_label, ms_type, scan_time_full_ds, intensity_full_ds) "
        "VALUES ('Peak1', 'S1', 'ms1', [1.0,2.0], [10.0,20.0])"
    )

    df = get_chromatogram_dataframe(con, "Peak1", full_range=True)

    assert df is not None
    row = df.to_dicts()[0]
    assert row["scan_time_sliced"] == [1.0, 2.0]
    assert row["intensity_sliced"] == [10.0, 20.0]


def test_get_chromatogram_dataframe_full_range_raw():
    con = duckdb.connect(":memory:")
    _create_tables(con)

    con.execute(
        "INSERT INTO samples (ms_file_label, color, label, sample_type, use_for_optimization) "
        "VALUES ('S1', '#ff0000', 'Sample1', 'TypeA', TRUE)"
    )
    con.execute(
        "INSERT INTO targets (peak_label, ms_type, mz_mean, mz_width, rt_min, rt_max, rt_unit) "
        "VALUES ('Peak1', 'ms1', 100.0, 10.0, 0.0, 2.0, 's')"
    )
    con.execute(
        "CREATE TABLE ms_file_scans (ms_file_label VARCHAR, scan_id INTEGER, scan_time DOUBLE, ms_type VARCHAR)"
    )
    con.execute(
        "INSERT INTO ms_file_scans VALUES ('S1', 1, 1.0, 'ms1')"
    )
    con.execute(
        "INSERT INTO ms1_data (ms_file_label, scan_id, mz, intensity, scan_time) VALUES ('S1', 1, 100.0, 10.0, 1.0)"
    )

    df = get_chromatogram_dataframe(con, "Peak1", full_range=True)

    assert df is not None
    row = df.to_dicts()[0]
    assert row["scan_time_sliced"] == [1.0]
    assert row["intensity_sliced"] == [10.0]


def test_get_chromatogram_dataframe_full_range_missing_target():
    con = duckdb.connect(":memory:")
    _create_tables(con)

    df = get_chromatogram_dataframe(con, "Missing", full_range=True)

    assert df is None


def test_get_chromatogram_dataframe_full_range_ms2_raw():
    con = duckdb.connect(":memory:")
    _create_tables(con)

    con.execute(
        "INSERT INTO samples (ms_file_label, color, label, sample_type, use_for_optimization) "
        "VALUES ('S1', '#ff0000', 'Sample1', 'TypeA', TRUE)"
    )
    con.execute(
        "INSERT INTO targets (peak_label, ms_type, mz_mean, mz_width, rt_min, rt_max, rt_unit, filterLine) "
        "VALUES ('Peak1', 'ms2', 100.0, 10.0, 0.0, 2.0, 's', 'FTMS')"
    )
    con.execute(
        "CREATE TABLE ms_file_scans (ms_file_label VARCHAR, scan_id INTEGER, scan_time DOUBLE, ms_type VARCHAR)"
    )
    con.execute(
        "INSERT INTO ms_file_scans VALUES ('S1', 1, 1.0, 'ms2')"
    )
    con.execute(
        "INSERT INTO ms2_data (ms_file_label, scan_id, mz, intensity, scan_time, filterLine) "
        "VALUES ('S1', 1, 100.0, 10.0, 1.0, 'FTMS')"
    )

    df = get_chromatogram_dataframe(con, "Peak1", full_range=True)

    assert df is not None
    row = df.to_dicts()[0]
    assert row["scan_time_sliced"] == [1.0]
    assert row["intensity_sliced"] == [10.0]


def test_get_chromatogram_dataframe_full_range_populates(monkeypatch):
    con = duckdb.connect(":memory:")
    _create_tables(con)

    con.execute(
        "INSERT INTO samples (ms_file_label, color, label, sample_type, use_for_optimization, ms_type) "
        "VALUES ('S1', '#ff0000', 'Sample1', 'TypeA', TRUE, 'ms1')"
    )
    con.execute(
        "INSERT INTO targets (peak_label, ms_type, mz_mean, mz_width) VALUES ('Peak1', 'ms1', 100.0, 10.0)"
    )
    con.execute(
        "INSERT INTO chromatograms (peak_label, ms_file_label, ms_type, scan_time_full_ds, intensity_full_ds) "
        "VALUES ('Peak1', 'S1', 'ms1', NULL, NULL)"
    )

    def _populate(_wdir, _target_label, n_out, conn):
        conn.execute(
            "UPDATE chromatograms SET scan_time_full_ds = [1.0,2.0], intensity_full_ds = [10.0,20.0] "
            "WHERE peak_label = 'Peak1'"
        )

    monkeypatch.setattr(
        "ms_mint_app.plugins.target_optimization.populate_full_range_downsampled_chromatograms_for_target",
        _populate,
    )

    df = get_chromatogram_dataframe(con, "Peak1", full_range=True, wdir="/tmp")

    assert df is not None
    row = df.to_dicts()[0]
    assert row["scan_time_sliced"] == [1.0, 2.0]
    assert row["intensity_sliced"] == [10.0, 20.0]
