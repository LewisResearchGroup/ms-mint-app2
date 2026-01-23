import pandas as pd
import pytest

from ms_mint_app.tools import get_targets_v2


def _run_get_targets(tmp_path, df):
    file_path = tmp_path / "targets.csv"
    df.to_csv(file_path, index=False)
    return get_targets_v2([str(file_path)])


@pytest.mark.parametrize(
    "polarity_input, expected",
    [
        ("+", "Positive"),
        ("positive", "Positive"),
        ("-", "Negative"),
        ("negative", "Negative"),
    ],
)
def test_polarity_normalization(tmp_path, polarity_input, expected):
    df = pd.DataFrame(
        [
            {"peak_label": "Glucose", "rt": 120.5, "polarity": polarity_input},
        ]
    )

    targets_df, failed_files, failed_targets, _ = _run_get_targets(tmp_path, df)

    assert failed_files == {}
    assert failed_targets == []
    assert targets_df.iloc[0]["polarity"] == expected


def test_polarity_missing_is_null(tmp_path):
    df = pd.DataFrame(
        [
            {"peak_label": "Glucose", "rt": 120.5},
        ]
    )

    targets_df, failed_files, failed_targets, _ = _run_get_targets(tmp_path, df)

    assert failed_files == {}
    assert failed_targets == []
    assert pd.isna(targets_df.iloc[0]["polarity"])


@pytest.mark.parametrize(
    "row, expected_ms_type",
    [
        ({"peak_label": "MS2WithFilter", "rt": 120.5, "filterLine": "FTMS + p ESI Full ms2 163.06@hcd25.00"}, "ms2"),
        ({"peak_label": "MS1NoFilter", "rt": 120.5}, "ms1"),
        ({"peak_label": "Contradiction1", "rt": 120.5, "ms_type": "ms2"}, "ms1"),
        ({"peak_label": "Contradiction2", "rt": 120.5, "ms_type": "ms1", "filterLine": "FTMS + p ESI Full ms2 163.06@hcd25.00"}, "ms2"),
    ],
)
def test_ms_type_derived_from_filterline(tmp_path, row, expected_ms_type):
    df = pd.DataFrame([row])

    targets_df, failed_files, failed_targets, _ = _run_get_targets(tmp_path, df)

    assert failed_files == {}
    assert failed_targets == []
    assert targets_df.iloc[0]["ms_type"] == expected_ms_type
