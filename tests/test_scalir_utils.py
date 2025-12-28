import numpy as np
import pandas as pd
import pytest
from ms_mint_app.plugins.scalir_utils import (
    classic_lstsqr,
    classic_lstsqr_variable_slope,
    classic_lstsqr_variable_slope_interval,
    find_linear_range,
    find_linear_range_variable_slope,
    ConcentrationEstimator,
    to_conc
)

def test_classic_lstsqr_fixed_slope():
    # y = x + 2 (log scale)
    x = np.array([1, 2, 3])
    y = np.array([3, 4, 5])
    intercept, res, r_ini, r_last = classic_lstsqr(x, y)
    assert intercept == pytest.approx(2.0)
    assert res == pytest.approx(0.0)
    assert r_ini == pytest.approx(0.0)
    assert r_last == pytest.approx(0.0)

def test_classic_lstsqr_variable_slope():
    # y = 2x + 1 (log scale)
    x = np.array([1, 2, 3])
    y = np.array([3, 5, 7])
    intercept, slope, res, r_ini, r_last = classic_lstsqr_variable_slope(x, y)
    assert intercept == pytest.approx(1.0)
    assert slope == pytest.approx(2.0)
    assert res == pytest.approx(0.0)

def test_classic_lstsqr_variable_slope_interval():
    # y = 3x + 1, interval [0.5, 2.0] -> slope should be clamped to 2.0
    x = np.array([1, 2, 3])
    y = np.array([4, 7, 10])
    intercept, slope, res, r_ini, r_last = classic_lstsqr_variable_slope_interval(x, y, (0.5, 2.0))
    assert slope == pytest.approx(2.0)
    # y_hat = 2x + intercept
    # y_avg = 7, x_avg = 2
    # 7 = 2*2 + intercept -> intercept = 3
    assert intercept == pytest.approx(3.0)

def test_find_linear_range():
    # Linear part: [2, 3, 4] with y = x + 1
    # Noise at ends: [1, 5]
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([10, 3, 4, 5, 20])
    intercept, x_c, y_c, res = find_linear_range(x, y, 0.01)
    assert len(x_c) == 3
    assert np.all(x_c == [2, 3, 4])
    assert intercept == pytest.approx(1.0)

def test_concentration_estimator_fixed():
    x_train = pd.DataFrame({
        'peak_label': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [10, 100, 1000, 20, 200, 2000]
    })
    # y = x * 0.1 (fixed slope = 1 in log scale)
    y_train = np.array([1, 10, 100, 2, 20, 200])
    
    estimator = ConcentrationEstimator()
    estimator.fit(x_train, y_train, v_slope='fixed')
    
    # Check params
    assert 'A' in estimator.params_.peak_label.values
    params_A = estimator.params_[estimator.params_.peak_label == 'A'].iloc[0]
    assert params_A.slope == 1.0
    # log(y) = 1.0 * log(x) + intercept
    # log(1) = 1.0 * log(10) + intercept -> 0 = 2.3025 + intercept -> intercept = -2.3025
    # e^intercept = 0.1
    assert np.exp(params_A.intercept) == pytest.approx(0.1)

    # Predict
    X_test = pd.DataFrame({
        'peak_label': ['A', 'B'],
        'value': [500, 500]
    })
    pred = estimator.predict(X_test)
    assert pred.iloc[0].pred_conc == pytest.approx(50.0)
    assert pred.iloc[1].pred_conc == pytest.approx(50.0)

def test_to_conc():
    # y = e^(2 * log(x) + 1) = x^2 * e^1
    val = to_conc(2.0, 1.0, 10.0)
    assert val == pytest.approx(100.0 * np.exp(1.0))

def test_transform_empty():
    X = pd.DataFrame({'peak_label': ['A'], 'value': [10.0]})
    cal = pd.DataFrame(columns=["peak_label", "slope", "intercept", "lin_range_min", "lin_range_max"])
    from ms_mint_app.plugins.scalir_utils import transform
    res = transform(X, cal)
    assert len(res) == 1
    assert res.iloc[0].peak_label is np.nan # or whatever it returns for index alignment
    assert "pred_conc" in res.columns
