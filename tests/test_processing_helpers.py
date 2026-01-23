import base64
import pandas as pd
import pytest

from ms_mint_app.plugins.processing import _parse_uploaded_standards, _plot_curve_fig, _generate_csv_from_db
from ms_mint_app.duckdb_manager import duckdb_connection


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


def test_parse_uploaded_standards_rejects_type():
    contents = "data:application/octet-stream;base64,AAAA"
    with pytest.raises(ValueError, match="Unsupported"):
        _parse_uploaded_standards(contents, "standards.bin")


def test_plot_curve_fig_empty():
    frame = pd.DataFrame(columns=["peak_label", "true_conc", "value", "in_range", "pred_conc"])
    fig = _plot_curve_fig(frame, "MissingPeak")

    assert fig.data == ()


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


def test_generate_csv_from_db(tmp_path):
    wdir = tmp_path / "workspace"
    wdir.mkdir()

    with duckdb_connection(wdir) as conn:
        conn.execute(
            "INSERT INTO samples (ms_file_label, ms_type) VALUES ('S1', 'ms1')"
        )
        conn.execute(
            "INSERT INTO results (peak_label, ms_file_label, peak_area) VALUES ('Peak1', 'S1', 123.0)"
        )

    csv_path = _generate_csv_from_db(str(wdir), "TestWS", ["peak_area"])

    assert csv_path is not None
    assert pd.read_csv(csv_path).shape[0] == 1
