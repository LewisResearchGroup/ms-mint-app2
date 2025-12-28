"""Standalone SCALiR-style calibration utilities with no ms_conc dependency."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


def classic_lstsqr(x_list, y_list):
    """Least squares with fixed slope=1 on log scale."""
    x_arr = np.asarray(x_list, dtype=float)
    y_arr = np.asarray(y_list, dtype=float)
    N = float(x_arr.size)
    x_avg = np.mean(x_arr)
    y_avg = np.mean(y_arr)
    delta_x = x_arr - x_avg
    delta_y = y_arr - y_avg
    # slope fixed to 1, so only intercept is estimated
    y_interc = y_avg - x_avg
    y_hat = y_interc + x_arr
    residual = float(np.sum((y_arr - y_hat) ** 2) / (N**2))
    r_ini = float((y_arr[0] - y_hat[0]) ** 2)
    r_last = float((y_arr[-1] - y_hat[-1]) ** 2)
    return y_interc, residual, r_ini, r_last


def classic_lstsqr_variable_slope(x_list, y_list):
    """Least squares with free slope on log scale."""
    x_arr = np.asarray(x_list, dtype=float)
    y_arr = np.asarray(y_list, dtype=float)
    N = float(x_arr.size)
    x_avg = np.mean(x_arr)
    y_avg = np.mean(y_arr)
    delta_x = x_arr - x_avg
    delta_y = y_arr - y_avg
    var_x = float(np.dot(delta_x, delta_x))
    cov_xy = float(np.dot(delta_x, delta_y))
    if var_x == 0:
        slope = 1.0
    else:
        slope = cov_xy / var_x
    y_interc = y_avg - slope * x_avg
    y_hat = y_interc + slope * x_arr
    residual = float(np.sum((y_arr - y_hat) ** 2) / N)
    r_ini = float((y_arr[0] - y_hat[0]) ** 2)
    r_last = float((y_arr[-1] - y_hat[-1]) ** 2)
    return y_interc, slope, residual, r_ini, r_last


def classic_lstsqr_variable_slope_interval(x_list, y_list, slope_interval):
    """Least squares with slope clamped to interval on log scale."""
    x_arr = np.asarray(x_list, dtype=float)
    y_arr = np.asarray(y_list, dtype=float)
    N = float(x_arr.size)
    x_avg = np.mean(x_arr)
    y_avg = np.mean(y_arr)
    delta_x = x_arr - x_avg
    delta_y = y_arr - y_avg
    var_x = float(np.dot(delta_x, delta_x))
    cov_xy = float(np.dot(delta_x, delta_y))
    if var_x == 0:
        slope = 1.0
    else:
        slope = cov_xy / var_x
    slope = max(min(slope, slope_interval[1]), slope_interval[0])
    y_interc = y_avg - slope * x_avg
    y_hat = y_interc + slope * x_arr
    residual = float(np.sum((y_arr - y_hat) ** 2) / (N**2))
    r_ini = float((y_arr[0] - y_hat[0]) ** 2)
    r_last = float((y_arr[-1] - y_hat[-1]) ** 2)
    return y_interc, slope, residual, r_ini, r_last


def find_linear_range(x, y, th):
    """Search linear range assuming slope=1 (log scale)."""
    x_c = np.asarray(x, dtype=float)
    y_c = np.asarray(y, dtype=float)
    order = np.lexsort((x_c, y_c))
    x_c = x_c[order]
    y_c = y_c[order]
    y_intercept, res, r_ini, r_last = classic_lstsqr(x_c, y_c)
    while ((res > th) or (r_ini > 0.2) or (r_last > 0.2)) and len(x_c) > 3:
        if r_ini > r_last:
            x_c = x_c[1:]
            y_c = y_c[1:]
        else:
            x_c = x_c[:-1]
            y_c = y_c[:-1]
        y_intercept, res, r_ini, r_last = classic_lstsqr(x_c, y_c)
    return y_intercept, x_c, y_c, res


def find_linear_range_variable_slope(x, y, th):
    """Search linear range allowing free slope (log scale)."""
    x_c = np.asarray(x, dtype=float)
    y_c = np.asarray(y, dtype=float)
    order = np.lexsort((x_c, y_c))
    x_c = x_c[order]
    y_c = y_c[order]
    y_intercept, slope, res, r_ini, r_last = classic_lstsqr_variable_slope(x_c, y_c)
    while ((res > th) or (r_ini > 0.2) or (r_last > 0.2)) and len(x_c) > 3:
        if r_ini > r_last:
            x_c = x_c[1:]
            y_c = y_c[1:]
        else:
            x_c = x_c[:-1]
            y_c = y_c[:-1]
        y_intercept, slope, res, r_ini, r_last = classic_lstsqr_variable_slope(x_c, y_c)
    return y_intercept, slope, x_c, y_c, res


def find_linear_range_variable_slope_interval(x, y, th, interval):
    """Search linear range with slope constrained to interval (log scale)."""
    interval = (min(interval), max(interval))
    x_c = np.asarray(x, dtype=float)
    y_c = np.asarray(y, dtype=float)
    order = np.lexsort((x_c, y_c))
    x_c = x_c[order]
    y_c = y_c[order]
    y_intercept, slope, res, r_ini, r_last = classic_lstsqr_variable_slope_interval(x_c, y_c, interval)
    while ((res > th) or (r_ini > 0.2) or (r_last > 0.2)) and len(x_c) > 3:
        if r_ini > r_last:
            x_c = x_c[1:]
            y_c = y_c[1:]
        else:
            x_c = x_c[:-1]
            y_c = y_c[:-1]
        y_intercept, slope, res, r_ini, r_last = classic_lstsqr_variable_slope_interval(x_c, y_c, interval)
    return y_intercept, slope, x_c, y_c, res


def calibration_curves(x_train, y_train):
    peaks = np.unique(x_train.peak_label)
    rows = []
    for col in peaks:
        x = np.asarray(x_train.value[x_train.peak_label == col], dtype=float)
        y = np.asarray(y_train[x_train.peak_label == col], dtype=float)
        mask = (x > 1e-11) & (y > 1e-11)
        x = x[mask]
        y = y[mask]
        x_log = np.log(x)
        y_log = np.log(y)
        if x_log.size > 2:
            y_inter, x_c, y_c, res = find_linear_range(x_log, y_log, 0.01)
        else:
            y_inter, x_c, y_c, res = 0.0, x_log, y_log, 0.0
        rows.append(
            {
                "peak_label": col,
                "slope": 1.0,
                "intercept": y_inter,
                "lin_range_min": float(min(y_c)) if len(y_c) else np.nan,
                "lin_range_max": float(max(y_c)) if len(y_c) else np.nan,
                "N_points": float(len(x_c)),
                "Residual": float(res),
            }
        )
    calibration = pd.DataFrame(rows, columns=["peak_label", "slope", "intercept", "lin_range_min", "lin_range_max", "N_points", "Residual"])
    calibration["LLOQ"] = calibration.lin_range_min.apply(lambda x: np.exp(x) if pd.notna(x) else np.nan)
    calibration["ULOQ"] = calibration.lin_range_max.apply(lambda x: np.exp(x) if pd.notna(x) else np.nan)
    return calibration


def calibration_curves_variable_slope(x_train, y_train):
    peaks = np.unique(x_train.peak_label)
    rows = []
    for col in peaks:
        x = np.asarray(x_train.value[x_train.peak_label == col], dtype=float)
        y = np.asarray(y_train[x_train.peak_label == col], dtype=float)
        mask = (x > 1e-11) & (y > 1e-11)
        x = x[mask]
        y = y[mask]
        x_log = np.log(x)
        y_log = np.log(y)
        if x_log.size > 2:
            y_inter, slope, x_c, y_c, res = find_linear_range_variable_slope(x_log, y_log, 0.01)
        else:
            y_inter, slope, x_c, y_c, res = 0.0, 1.0, x_log, y_log, 0.0
        rows.append(
            {
                "peak_label": col,
                "slope": float(slope),
                "intercept": float(y_inter),
                "lin_range_min": float(min(y_c)) if len(y_c) else np.nan,
                "lin_range_max": float(max(y_c)) if len(y_c) else np.nan,
                "N_points": float(len(x_c)),
                "Residual": float(res),
            }
        )
    calibration = pd.DataFrame(
        rows, columns=["peak_label", "slope", "intercept", "lin_range_min", "lin_range_max", "N_points", "Residual"]
    )
    calibration["LLOQ"] = calibration.lin_range_min.apply(lambda x: np.exp(x) if pd.notna(x) else np.nan)
    calibration["ULOQ"] = calibration.lin_range_max.apply(lambda x: np.exp(x) if pd.notna(x) else np.nan)
    return calibration


def calibration_curves_variable_slope_interval(x_train, y_train, interval):
    peaks = np.unique(x_train.peak_label)
    rows = []
    for col in peaks:
        x = np.asarray(x_train.value[x_train.peak_label == col], dtype=float)
        y = np.asarray(y_train[x_train.peak_label == col], dtype=float)
        mask = (x > 1e-11) & (y > 1e-11)
        x = x[mask]
        y = y[mask]
        x_log = np.log(x)
        y_log = np.log(y)
        if x_log.size > 2:
            y_inter, slope, x_c, y_c, res = find_linear_range_variable_slope_interval(x_log, y_log, 0.01, interval)
        else:
            y_inter, slope, x_c, y_c, res = 0.0, 1.0, x_log, y_log, 0.0
        rows.append(
            {
                "peak_label": col,
                "slope": float(slope),
                "intercept": float(y_inter),
                "lin_range_min": float(min(y_c)) if len(y_c) else np.nan,
                "lin_range_max": float(max(y_c)) if len(y_c) else np.nan,
                "N_points": float(len(x_c)),
                "Residual": float(res),
            }
        )
    calibration = pd.DataFrame(
        rows, columns=["peak_label", "slope", "intercept", "lin_range_min", "lin_range_max", "N_points", "Residual"]
    )
    calibration["LLOQ"] = calibration.lin_range_min.apply(lambda x: np.exp(x) if pd.notna(x) else np.nan)
    calibration["ULOQ"] = calibration.lin_range_max.apply(lambda x: np.exp(x) if pd.notna(x) else np.nan)
    return calibration


def info_from_Mint(mint_, by):
    """Keep only columns needed for downstream calculations."""
    return mint_[["ms_file_label", "peak_label", by]].rename(columns={"ms_file_label": "ms_file"})


def setting_from_stdinfo(std_info, results_):
    """Attach standard concentrations to results table."""
    output = results_.copy()
    try:
        output.ms_file = output.ms_file.apply(lambda x: os.path.basename(x).replace(".mzXML", ""))
    except Exception:
        pass
    long_std = std_info.melt(id_vars="peak_label", var_name="ms_file", value_name="STD_CONC")
    merged = output.merge(long_std, on=["peak_label", "ms_file"], how="left")
    return merged[merged["STD_CONC"].notna()]


def training_from_standard_results(std_results, by="peak_max"):
    x_train = std_results[["peak_label", by]].copy()
    x_train.rename(columns={by: "value"}, inplace=True)
    y_train = np.array(std_results.STD_CONC)
    return x_train, y_train


def to_conc(slope, intercept, point_vector):
    return np.exp(slope * np.log(point_vector + 1e-12) + intercept)


def transform(X, calibration_curves_df):
    calibration_curve = calibration_curves_df[["peak_label", "slope", "intercept", "lin_range_min", "lin_range_max"]]
    X0 = X.copy().fillna(0)
    results = []
    for _, (peak_label, slope, intercept, lin_range_min, lin_range_max) in calibration_curve.iterrows():
        value = X0.loc[X0.peak_label == peak_label, "value"]
        conc = to_conc(slope, intercept, value)
        inrange = np.ones(len(conc))
        df = pd.DataFrame({"value": value, "pred_conc": conc, "in_range": inrange})
        df.loc[df.pred_conc < np.exp(lin_range_min), "in_range"] = 0
        df.loc[df.pred_conc > np.exp(lin_range_max), "in_range"] = 0
        df["peak_label"] = peak_label
        results.append(df)
    if not results:
        return pd.DataFrame(columns=["peak_label", "pred_conc", "in_range"], index=X0.index)
    df_conc = pd.concat(results).loc[X0.index, ["peak_label", "pred_conc", "in_range"]]
    return df_conc


def train_to_validation(X, Y, curves):
    X0 = X.copy()
    X0["true_conc"] = Y
    curves0 = curves.copy()
    numeric_cols = curves0.select_dtypes(include=["number"]).columns
    curves0.loc[:, numeric_cols] = curves0.loc[:, numeric_cols].fillna(1e-12)
    curves0["lin_range_min"] = pd.to_numeric(curves0["lin_range_min"], errors="coerce")
    curves0["lin_range_max"] = pd.to_numeric(curves0["lin_range_max"], errors="coerce")
    curves0["Y_min"] = np.exp(curves0["lin_range_min"].to_numpy(dtype=float) - 1e-8)
    curves0["Y_max"] = np.exp(curves0["lin_range_max"].to_numpy(dtype=float) + 1e-8)

    y_min_map = curves0.set_index("peak_label")["Y_min"]
    y_max_map = curves0.set_index("peak_label")["Y_max"]

    X0["Y_min"] = X0.peak_label.map(y_min_map)
    X0["Y_max"] = X0.peak_label.map(y_max_map)

    X0.loc[X0.true_conc < X0.Y_min, "true_conc"] = None
    X0.loc[X0.true_conc > X0.Y_max, "true_conc"] = None
    return np.array(X0.true_conc)


class ConcentrationEstimator:
    """Small wrapper to fit and predict concentrations."""

    def __init__(self):
        self.params_ = pd.DataFrame()
        self.interval = (0.5, 2.0)

    def set_interval(self, interval):
        self.interval = interval

    def fit(self, X, y, v_slope="fixed"):
        if v_slope == "interval":
            self.params_ = calibration_curves_variable_slope_interval(X, y, self.interval)
        elif v_slope == "wide":
            self.params_ = calibration_curves_variable_slope(X, y)
        else:
            self.params_ = calibration_curves(X, y)

    def predict(self, X):
        return transform(X, self.params_)
