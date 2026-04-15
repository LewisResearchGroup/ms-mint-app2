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


def test_maven_ms2_peak_list_maps_isotope_label_and_medrt(tmp_path):
    df = pd.DataFrame(
        [
            {
                "compound": "355.200134@21.942015",
                "medRt": 21.942,
                "isotopeLabel": "+ c ESI SRM ms2 426.320 [355.199-355.201]",
                "parent": 355.200134,
            },
        ]
    )

    targets_df, failed_files, failed_targets, _ = _run_get_targets(tmp_path, df)

    assert failed_files == {}
    assert failed_targets == []

    row = targets_df.iloc[0]
    assert row["peak_label"] == "355.200134@21.942015"
    assert row["filterLine"] == "+ c ESI SRM ms2 426.320 [355.199-355.201]"
    assert row["ms_type"] == "ms2"
    assert row["rt"] == pytest.approx(21.942 * 60)
    assert row["rt_min"] == pytest.approx((21.942 * 60) - 5.0)
    assert row["rt_max"] == pytest.approx((21.942 * 60) + 5.0)
    assert row["mz_mean"] == pytest.approx(355.200134)


def test_category_and_notes_columns_are_imported(tmp_path):
    df = pd.DataFrame(
        [
            {
                "Compound": "Glucose",
                "meanRt": 2.0,
                "meanMz": 180.0634,
                "Category": "Sugar",
                "Notes": "Imported from El-Maven",
            },
        ]
    )

    targets_df, failed_files, failed_targets, _ = _run_get_targets(tmp_path, df)

    assert failed_files == {}
    assert failed_targets == []
    row = targets_df.iloc[0]
    assert row["peak_label"] == "Glucose"
    assert row["category"] == "Sugar"
    assert row["notes"] == "Imported from El-Maven"


def test_unknown_target_columns_are_reported_but_do_not_fail_import(tmp_path):
    df = pd.DataFrame(
        [
            {
                "Compound": "Lactate",
                "meanRt": 1.5,
                "meanMz": 89.0244,
                "Category": "Organic acid",
                "Notes": "Keep",
                "Unexpected Column": "ignored",
            },
        ]
    )

    targets_df, failed_files, failed_targets, stats = _run_get_targets(tmp_path, df)

    assert failed_files == {}
    assert failed_targets == []
    assert len(targets_df) == 1
    assert stats["ignored_columns"] == ["Unexpected Column"]
    assert stats["ignored_columns_by_file"]["targets.csv"] == ["Unexpected Column"]
