import types

import pytest

import ms_mint_app.duckdb_manager as dm


def test_calculate_optimal_params_with_user_values(monkeypatch):
    cpus, ram_gb, batch_size = dm.calculate_optimal_params(user_cpus=2, user_ram=8)

    assert cpus == 2
    assert ram_gb == 4  # capped at 2x CPU
    assert batch_size == 1000  # minimum enforced


def test_calculate_optimal_params_auto(monkeypatch):
    monkeypatch.setattr(dm, "get_physical_cores", lambda: 4)
    monkeypatch.setattr("os.cpu_count", lambda: 8)

    class _VMem:
        available = 16 * 1024 ** 3

    monkeypatch.setattr("psutil.virtual_memory", lambda: _VMem())

    cpus, ram_gb, batch_size = dm.calculate_optimal_params()

    assert cpus == 4
    assert ram_gb == 8
    assert batch_size == 1600


def test_calculate_optimal_batch_size_limits(monkeypatch):
    monkeypatch.setattr(dm, "calculate_optimal_params", lambda user_cpus=None, user_ram=None: (2, 4, 1200))

    batch_size = dm.calculate_optimal_batch_size(total_pairs=2000)
    assert batch_size == 500


def test_get_effective_cpus():
    assert dm.get_effective_cpus(8, 4) == 4
    assert dm.get_effective_cpus(None, 4) == 4
    assert dm.get_effective_cpus(2, None) == 2


def test_build_where_and_params():
    filter_ = {"name": ["foo"], "status": ["A", "B"]}
    filter_options = {"name": {"filterMode": "keyword"}, "status": {}}

    where_sql, params = dm.build_where_and_params(filter_, filter_options)

    assert where_sql == 'WHERE "name" ILIKE ? AND "status" IN (?,?)'
    assert params == ["%foo%", "A", "B"]


def test_build_order_by_with_tie():
    sorter = {"columns": ["name", "value"], "orders": ["ascend", "descend"]}
    column_types = {"name": "VARCHAR", "value": "DOUBLE", "id": "INTEGER"}

    order = dm.build_order_by(sorter, column_types, tie=("id", "ASC"))

    assert order == 'ORDER BY "name" COLLATE NOCASE ASC NULLS LAST, "value" DESC NULLS FIRST, "id" ASC'


def test_build_order_by_ignores_invalid_columns():
    sorter = {"columns": ["missing"], "orders": ["ascend"]}
    column_types = {"name": "VARCHAR"}

    order = dm.build_order_by(sorter, column_types)
    assert order == ""


def test_corruption_flag_helpers(tmp_path):
    ws_path = tmp_path / "workspace"

    assert dm.is_workspace_corrupted(ws_path) is False

    dm._mark_corrupted(ws_path)
    assert dm.is_workspace_corrupted(ws_path) is True

    dm.clear_corruption_flag(ws_path)
    assert dm.is_workspace_corrupted(ws_path) is False


def test_corruption_notification_shape():
    note = dm.get_corruption_notification()
    assert note["type"] == "error"
    assert "corrupted" in note["description"].lower()


def test_apply_savgol_smoothing_basic():
    intensity = [1.0, 2.0, 3.0, 4.0, 5.0]
    smoothed = dm._apply_savgol_smoothing(intensity, window_length=3, polyorder=1)
    assert len(smoothed) == len(intensity)


def test_apply_lttb_downsampling_no_library(monkeypatch):
    monkeypatch.setattr(dm, "_lttbc", None)
    scan_time = [0, 1, 2, 3]
    intensity = [1, 2, 3, 4]
    out_x, out_y = dm._apply_lttb_downsampling(scan_time, intensity, n_out=2)
    assert out_x == scan_time
    assert out_y == intensity
