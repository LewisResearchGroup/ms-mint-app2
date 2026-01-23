import numpy as np

from ms_mint_app.peak_fitting import emg_gaussian, fit_emg_peak, fit_peaks_batch


def test_fit_emg_peak_success():
    scan_time = np.linspace(0, 10, 200)
    intensity = emg_gaussian(scan_time, amplitude=1000, center=5.0, sigma=0.5, gamma=1.5)

    result = fit_emg_peak(scan_time, intensity, expected_rt=5.0)

    assert result.success is True
    assert result.fit_r_squared > 0.9
    assert abs(result.peak_rt_fitted - 5.0) < 0.2
    assert result.peak_area_fitted > 0


def test_fit_emg_peak_insufficient_data():
    scan_time = np.array([1.0, 2.0, 3.0])
    intensity = np.array([10.0, 20.0, 15.0])

    result = fit_emg_peak(scan_time, intensity)

    assert result.success is False
    assert result.error_message == "Insufficient data points"


def test_fit_emg_peak_no_positive_intensity():
    scan_time = np.linspace(0, 10, 10)
    intensity = np.zeros_like(scan_time)

    result = fit_emg_peak(scan_time, intensity)

    assert result.success is False
    assert result.error_message == "No positive intensity"


def test_fit_peaks_batch_small_dataset():
    scan_time = np.linspace(0, 10, 100)
    intensity = emg_gaussian(scan_time, amplitude=500, center=4.0, sigma=0.7, gamma=1.2)
    peaks_data = [
        ("Peak1", "File1", scan_time, intensity, 4.0),
        ("Peak2", "File2", scan_time, intensity * 0.5, 4.0),
    ]

    results = fit_peaks_batch(peaks_data, n_workers=8)

    assert len(results) == 2
    for res in results:
        assert res["peak_label"] in {"Peak1", "Peak2"}
        assert "peak_area_fitted" in res
        assert "fit_success" in res
