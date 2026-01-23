import numpy as np
import polars as pl

from ms_mint_app.plugins.analysis_tools import trace_helper


def test_apply_savgol_smoothing_window_adjustments():
    intensity = np.linspace(1, 10, 11)

    smoothed = trace_helper.apply_savgol_smoothing(intensity, window_length=6, polyorder=10)

    assert len(smoothed) == len(intensity)
    assert np.all(smoothed >= 1.0)


def test_apply_lttb_downsampling_without_lttbc(monkeypatch):
    scan_time = np.linspace(0, 10, 200)
    intensity = np.linspace(1, 5, 200)

    monkeypatch.setattr(trace_helper, "_lttbc", None)
    trace_helper._LTTBC_MISSING_WARNED = False

    out_x, out_y = trace_helper.apply_lttb_downsampling(scan_time, intensity, n_out=50)

    assert np.allclose(out_x, scan_time)
    assert np.allclose(out_y, intensity)


def test_calculate_rt_alignment_and_shifts():
    df = pl.DataFrame(
        {
            "scan_time_sliced": [
                [4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0],
            ],
            "intensity_sliced": [
                [10.0, 50.0, 20.0],
                [5.0, 20.0, 60.0],
            ],
            "ms_file_label": ["File1", "File2"],
            "sample_type": ["Sample", "Sample"],
        }
    )

    shifts = trace_helper.calculate_rt_alignment(df, 4.0, 6.0)
    assert shifts["File1"] == 0.5
    assert shifts["File2"] == -0.5

    grouped = trace_helper.calculate_shifts_per_sample_type(df, shifts)
    assert grouped["Sample"] == 0.0


def test_generate_chromatogram_traces_alignment(monkeypatch):
    monkeypatch.setattr(trace_helper, "sparsify_chrom", lambda x, y, **_: (x, y))

    df = pl.DataFrame(
        {
            "scan_time_sliced": [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ],
            "intensity_sliced": [
                [10.0, 20.0, 15.0],
                [5.0, 15.0, 25.0],
            ],
            "label": ["SampleA", "SampleB"],
            "ms_file_label": ["FileA", "FileB"],
            "sample_type": ["Type1", "Type1"],
            "color": ["#ff0000", "#00ff00"],
            "scan_time_min_in_range": [1.0, 1.0],
            "scan_time_max_in_range": [3.0, 3.0],
            "intensity_min_in_range": [10.0, 5.0],
            "intensity_max_in_range": [20.0, 25.0],
        }
    )

    traces, x_min, x_max, y_min, y_max = trace_helper.generate_chromatogram_traces(
        df,
        use_megatrace=False,
        rt_alignment_shifts={"FileA": 1.0},
        ms_type="ms1",
        smoothing_params={"enabled": False},
        downsample_params={"enabled": False},
    )

    assert len(traces) == 2
    assert traces[0]["x"][0] == 2.0
    assert x_min == 1.0
    assert x_max == 3.0
    assert y_min == 5.0
    assert y_max == 25.0
