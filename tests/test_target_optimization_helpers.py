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
