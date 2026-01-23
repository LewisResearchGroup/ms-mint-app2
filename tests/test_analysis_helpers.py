import base64
import pandas as pd
import numpy as np
import pytest

from ms_mint_app.plugins.analysis import (
    _parse_uploaded_standards,
    _analysis_tour_steps,
    _build_color_map,
    _clean_numeric,
    _calc_y_range_numpy,
    _create_pivot_custom,
    load_persisted_scalir_results,
    _plot_curve_fig,
)
from ms_mint_app.duckdb_manager import _create_tables
import duckdb


def _encode_csv(content: str) -> str:
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    return f"data:text/csv;base64,{encoded}"


def test_parse_uploaded_standards_csv():
    contents = _encode_csv("peak_label,true_conc\nPeak1,10\nPeak2,20\n")
    df = _parse_uploaded_standards(contents, "standards.csv")

    assert list(df.columns) == ["peak_label", "true_conc"]
    assert len(df) == 2


def test_parse_uploaded_standards_rejects_missing():
    with pytest.raises(ValueError):
        _parse_uploaded_standards(None, None)


def test_parse_uploaded_standards_rejects_large():
    contents = "data:text/csv;base64," + ("A" * 15_000_001)
    with pytest.raises(ValueError, match="too large"):
        _parse_uploaded_standards(contents, "standards.csv")


def test_parse_uploaded_standards_rejects_type():
    contents = "data:application/octet-stream;base64,AAAA"
    with pytest.raises(ValueError, match="Unsupported"):
        _parse_uploaded_standards(contents, "standards.bin")


def test_analysis_tour_steps_variants():
    steps_pca = _analysis_tour_steps("pca")
    steps_tsne = _analysis_tour_steps("tsne")
    steps_violin = _analysis_tour_steps("raincloud")
    steps_cluster = _analysis_tour_steps("clustermap")
    steps_other = _analysis_tour_steps("other")

    assert len(steps_pca) > len(steps_other)
    assert len(steps_tsne) > len(steps_other)
    assert len(steps_violin) > len(steps_other)
    assert len(steps_cluster) > len(steps_other)


def test_build_color_map_respects_existing_and_fills_missing():
    df = pd.DataFrame(
        {
            "sample_type": ["A", "B", "C"],
            "color": ["#ff0000", None, "#bbbbbb"],
        }
    )

    color_map = _build_color_map(df, "sample_type")

    assert color_map["A"] == "#ff0000"
    assert "B" in color_map
    assert "C" in color_map


def test_build_color_map_missing_group():
    df = pd.DataFrame({"other": ["A"], "color": ["#ff0000"]})
    assert _build_color_map(df, "sample_type") == {}


def test_clean_numeric_drops_inf_and_fills():
    df = pd.DataFrame(
        {
            "a": [1.0, np.inf, np.nan],
            "b": [np.nan, np.nan, np.nan],
            "c": [2.0, 3.0, -np.inf],
        }
    )

    cleaned = _clean_numeric(df)

    assert cleaned.isna().sum().sum() == 0
    assert "b" not in cleaned.columns


def test_calc_y_range_numpy_log():
    data = [
        {"x": [0, 1, 2], "y": [0.5, 2.0, 4.0]},
        {"x": [0, 1, 2], "y": [1.1, 3.0, 5.0]},
    ]

    y_range = _calc_y_range_numpy(data, 0, 2, is_log=True)

    assert y_range is not None
    assert y_range[0] < y_range[1]


def test_create_pivot_custom_basic():
    con = duckdb.connect(":memory:")
    _create_tables(con)

    con.execute(
        "INSERT INTO samples (ms_file_label, sample_type, ms_type, use_for_analysis) VALUES ('S1', 'Sample', 'ms1', TRUE)"
    )
    con.execute(
        "INSERT INTO targets (peak_label, ms_type) VALUES ('Peak1', 'ms1'), ('Peak2', 'ms1')"
    )
    con.execute(
        "INSERT INTO results (ms_file_label, peak_label, peak_area) VALUES ('S1', 'Peak1', 100.0), ('S1', 'Peak2', 200.0)"
    )

    df = _create_pivot_custom(con, value="peak_area", table="results")

    assert "ms_type" in df.columns
    assert "sample_type" in df.columns
    assert "Peak1" in df.columns
    assert "Peak2" in df.columns
    assert df.loc[0, "Peak1"] == 100.0
    assert df.loc[0, "Peak2"] == 200.0


def test_load_persisted_scalir_results(tmp_path):
    results_dir = tmp_path / "results" / "scalir"
    results_dir.mkdir(parents=True)
    (results_dir / "train_frame.csv").write_text("peak_label,value\nPeak1,1\n")
    (results_dir / "standard_curve_parameters.csv").write_text("peak_label,param\nPeak1,0.5\n")
    (results_dir / "units.csv").write_text("peak_label,unit\nPeak1,uM\n")

    data = load_persisted_scalir_results(str(tmp_path))

    assert data is not None
    assert "train_frame" in data
    assert "params" in data
    assert data["common"] == ["Peak1"]


def test_plot_curve_fig_with_data():
    frame = pd.DataFrame(
        {
            "peak_label": ["Peak1", "Peak1"],
            "true_conc": [1.0, 2.0],
            "value": [10.0, 20.0],
            "in_range": [1, 0],
            "pred_conc": [1.0, 2.0],
        }
    )

    fig = _plot_curve_fig(frame, "Peak1")

    assert len(fig.data) >= 2
