import pandas as pd
import pytest

from ms_mint_app.tools import get_targets_v2


DEFAULT_RT_WINDOW = 5.0


def _run_get_targets(tmp_path, df):
    file_path = tmp_path / "targets.csv"
    df.to_csv(file_path, index=False)
    return get_targets_v2([str(file_path)])


def _failed_error(failed_targets, label):
    for entry in failed_targets:
        if entry.get("peak_label") == label:
            return entry.get("error", "")
    return ""


def test_rt_only_derives_bounds(tmp_path):
    df = pd.DataFrame(
        [
            {"peak_label": "Test1", "rt": 120.5},
        ]
    )

    targets_df, failed_files, failed_targets, _ = _run_get_targets(tmp_path, df)

    assert failed_files == {}
    assert failed_targets == []

    row = targets_df.iloc[0]
    assert row["rt"] == pytest.approx(120.5)
    assert row["rt_min"] == pytest.approx(120.5 - DEFAULT_RT_WINDOW)
    assert row["rt_max"] == pytest.approx(120.5 + DEFAULT_RT_WINDOW)
    assert bool(row["rt_auto_adjusted"]) is True


def test_rt_min_max_derives_center(tmp_path):
    df = pd.DataFrame(
        [
            {"peak_label": "Test2", "rt_min": 119.5, "rt_max": 121.5},
        ]
    )

    targets_df, failed_files, failed_targets, _ = _run_get_targets(tmp_path, df)

    assert failed_files == {}
    assert failed_targets == []

    row = targets_df.iloc[0]
    assert row["rt"] == pytest.approx(120.5)
    assert row["rt_min"] == pytest.approx(119.5)
    assert row["rt_max"] == pytest.approx(121.5)
    assert bool(row["rt_auto_adjusted"]) is False


def test_rt_and_rt_min_derives_rt_max(tmp_path):
    df = pd.DataFrame(
        [
            {"peak_label": "Test3", "rt": 120.0, "rt_min": 118.0},
        ]
    )

    targets_df, failed_files, failed_targets, _ = _run_get_targets(tmp_path, df)

    assert failed_files == {}
    assert failed_targets == []

    row = targets_df.iloc[0]
    assert row["rt"] == pytest.approx(120.0)
    assert row["rt_min"] == pytest.approx(118.0)
    assert row["rt_max"] == pytest.approx(122.0)
    assert bool(row["rt_auto_adjusted"]) is True


def test_rt_and_rt_max_derives_rt_min(tmp_path):
    df = pd.DataFrame(
        [
            {"peak_label": "Test4", "rt": 120.0, "rt_max": 123.0},
        ]
    )

    targets_df, failed_files, failed_targets, _ = _run_get_targets(tmp_path, df)

    assert failed_files == {}
    assert failed_targets == []

    row = targets_df.iloc[0]
    assert row["rt"] == pytest.approx(120.0)
    assert row["rt_min"] == pytest.approx(117.0)
    assert row["rt_max"] == pytest.approx(123.0)
    assert bool(row["rt_auto_adjusted"]) is True


@pytest.mark.parametrize(
    "invalid_row, expected_fragments",
    [
        ({"peak_label": "BadNoRt"}, ["must have at least one of: rt, rt_min, or rt_max"]),
        ({"peak_label": "BadOnlyMin", "rt_min": 119.0}, ["Cannot derive RT values", "Valid combinations"]),
        ({"peak_label": "BadOnlyMax", "rt_max": 121.5}, ["Cannot derive RT values", "Valid combinations"]),
        ({"peak_label": "BadNegativeMin", "rt": 0.5, "rt_min": -0.5}, ["rt_min cannot be negative"]),
        ({"peak_label": "BadInverted", "rt_min": 121.0, "rt_max": 119.0}, ["rt_min (121.00)"]),
    ],
)
def test_rt_validation_errors_are_reported(tmp_path, invalid_row, expected_fragments):
    df = pd.DataFrame(
        [
            {"peak_label": "ValidTarget", "rt": 60.0},
            invalid_row,
        ]
    )

    targets_df, failed_files, failed_targets, _ = _run_get_targets(tmp_path, df)

    assert failed_files == {}
    assert len(targets_df) == 1
    assert len(failed_targets) == 1

    error_msg = _failed_error(failed_targets, invalid_row["peak_label"])
    for fragment in expected_fragments:
        assert fragment in error_msg


def test_mz_width_defaults_to_10_ppm(tmp_path):
    df = pd.DataFrame(
        [
            {"peak_label": "Glucose", "rt": 120.5},
        ]
    )

    targets_df, failed_files, failed_targets, _ = _run_get_targets(tmp_path, df)

    assert failed_files == {}
    assert failed_targets == []

    row = targets_df.iloc[0]
    assert row["mz_width"] == pytest.approx(10.0)
