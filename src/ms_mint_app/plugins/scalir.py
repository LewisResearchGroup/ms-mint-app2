#!/usr/bin/env python3
"""Quick analysis script for SCALiR sample data.

The script reads a Mint full results table and a standards table, fits the
calibration curves using the existing ms_conc logic, outputs a concentration
table, and saves per-compound standard-curve plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from scalir_utils import (  # noqa: E402
    ConcentrationEstimator,
    info_from_Mint,
    setting_from_stdinfo,
    train_to_validation,
    training_from_standard_results,
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def prepare_mint_results(path: Path, intensity: str) -> pd.DataFrame:
    mint_df = read_table(path)
    if intensity not in mint_df.columns:
        available = ", ".join(mint_df.columns)
        raise ValueError(
            f"Column '{intensity}' not found in Mint results. Available: {available}"
        )
    return info_from_Mint(mint_df, intensity)


def prepare_standards(path: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    standards_df = read_table(path)
    units = None
    if "unit" in standards_df.columns:
        units = standards_df[["peak_label", "unit"]].copy()
        standards_df = standards_df.drop(columns=["unit"])
    return standards_df, units


def intersect_peaks(
    mint_results: pd.DataFrame,
    standards: pd.DataFrame,
    units: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Iterable[str]]:
    common = sorted(
        set(mint_results.peak_label.unique()) & set(standards.peak_label.unique())
    )
    if not common:
        raise ValueError("No overlapping peak_label values between inputs.")

    mint_filtered = mint_results[mint_results.peak_label.isin(common)].copy()
    standards_filtered = standards[standards.peak_label.isin(common)].copy()
    units_filtered = None
    if units is not None:
        units_filtered = units[units.peak_label.isin(common)].copy()
    return mint_filtered, standards_filtered, units_filtered, common


def fit_estimator(
    mint_results: pd.DataFrame,
    standards: pd.DataFrame,
    intensity: str,
    slope_mode: str,
    interval: Tuple[float, float],
) -> Tuple[
    ConcentrationEstimator,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
]:
    std_results = setting_from_stdinfo(standards, mint_results)
    x_train, y_train = training_from_standard_results(std_results, by=intensity)

    estimator = ConcentrationEstimator()
    if slope_mode == "interval":
        estimator.set_interval(np.array(interval, dtype=float))
    estimator.fit(x_train, y_train, v_slope=slope_mode)
    return estimator, std_results, x_train, y_train, estimator.params_


def build_concentration_table(
    estimator: ConcentrationEstimator,
    mint_results: pd.DataFrame,
    intensity: str,
    units: Optional[pd.DataFrame],
) -> pd.DataFrame:
    quant_input = mint_results[["ms_file", "peak_label", intensity]].rename(
        columns={intensity: "value"}
    )
    pred = estimator.predict(quant_input)
    quant_input["pred_conc"] = pred.pred_conc
    quant_input["in_range"] = pred.in_range
    if units is not None:
        quant_input = quant_input.merge(units, on="peak_label", how="left")
    return quant_input


def training_plot_frame(
    estimator: ConcentrationEstimator,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    params: pd.DataFrame,
) -> pd.DataFrame:
    frame = x_train.copy()
    frame["true_conc"] = y_train
    frame["corrected_conc"] = train_to_validation(x_train, y_train, params)
    frame["pred_conc"] = estimator.predict(frame[["peak_label", "value"]]).pred_conc
    frame["in_range"] = frame["corrected_conc"].notna().astype(int)
    frame = frame[frame.value > 0]
    frame["LLOQ"] = frame.peak_label.map(params.set_index("peak_label")["LLOQ"])
    frame["ULOQ"] = frame.peak_label.map(params.set_index("peak_label")["ULOQ"])
    return frame


def slugify_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    safe = safe.strip("_")
    return safe or "compound"


def plot_standard_curve(
    frame: pd.DataFrame,
    peak_label: str,
    units: Optional[pd.DataFrame],
    output_dir: Path,
) -> Path:
    subset = frame[frame.peak_label == peak_label]
    if subset.empty:
        return output_dir / f"{slugify_label(peak_label)}_curve.png"

    unit = None
    if units is not None:
        unit = units[units.peak_label == peak_label].unit.iloc[0]

    fig, ax = plt.subplots(figsize=(5, 4))
    in_range = subset[subset.in_range == 1]
    out_range = subset[subset.in_range != 1]

    if not out_range.empty:
        ax.scatter(
            out_range.true_conc,
            out_range.value,
            color="gray",
            label="Outside range",
            zorder=1,
        )
    if not in_range.empty:
        ax.scatter(
            in_range.true_conc,
            in_range.value,
            color="black",
            label="In range",
            zorder=2,
        )
        line_data = in_range.sort_values("pred_conc")
        ax.plot(
            line_data.pred_conc,
            line_data.value,
            color="black",
            linewidth=1,
            label="Fit",
            zorder=3,
        )

    xlabel = f"{peak_label} concentration"
    if unit:
        xlabel = f"{xlabel} ({unit})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{peak_label} intensity (AU)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{slugify_label(peak_label)}_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit SCALiR calibration curves and compute concentrations from a "
            "Mint full results table."
        )
    )
    parser.add_argument(
        "--mint-results",
        type=Path,
        default=Path("sample_files/SCALiR_MINT_Peaklist_Full_Results.csv"),
        help="Path to the Mint full results CSV.",
    )
    parser.add_argument(
        "--standards",
        type=Path,
        default=Path("sample_files/SCALiR_Standards_Concentrations File.csv"),
        help="Path to the standards concentration table.",
    )
    parser.add_argument(
        "--intensity",
        choices=["peak_max", "peak_area"],
        default="peak_area",
        help="Mint column to use for intensity.",
    )
    parser.add_argument(
        "--slope-mode",
        choices=["fixed", "interval", "wide"],
        default="fixed",
        help="Slope handling mode passed to ConcentrationEstimator.",
    )
    parser.add_argument(
        "--interval",
        nargs=2,
        type=float,
        default=(0.85, 1.15),
        metavar=("LOW", "HIGH"),
        help="Slope interval when using --slope-mode interval.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/output"),
        help="Directory to write concentration tables and plots.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate per-compound plots (disabled by default for speed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mint_results = prepare_mint_results(args.mint_results, args.intensity)
    standards, units = prepare_standards(args.standards)
    mint_results, standards, units, common = intersect_peaks(
        mint_results, standards, units
    )

    estimator, std_results, x_train, y_train, params = fit_estimator(
        mint_results, standards, args.intensity, args.slope_mode, args.interval
    )

    concentrations = build_concentration_table(
        estimator, mint_results, args.intensity, units
    )
    concentrations_path = args.output_dir / "concentrations.csv"
    concentrations.to_csv(concentrations_path, index=False)

    params_path = args.output_dir / "standard_curve_parameters.csv"
    params.to_csv(params_path, index=False)

    plot_paths = []
    if args.plots:
        train_frame = training_plot_frame(estimator, x_train, y_train, params)
        plots_dir = args.output_dir / "plots"
        for label in common:
            plot_paths.append(plot_standard_curve(train_frame, label, units, plots_dir))

    print("Analysis complete.")
    print(f"Saved concentrations to: {concentrations_path}")
    print(f"Saved curve parameters to: {params_path}")
    if args.plots:
        print("Plots:")
        for path in plot_paths:
            print(f" - {path}")
    else:
        print("Plots: skipped (use --plots to generate)")


if __name__ == "__main__":
    main()
