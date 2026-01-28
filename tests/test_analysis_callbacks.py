from types import SimpleNamespace

import duckdb
import pytest

import dash
import ms_mint_app.plugins.analysis.plugin as analysis_plugin
import ms_mint_app.plugins.analysis.tsne as tsne_module
from ms_mint_app.plugins.analysis._shared import TAB_DEFAULT_NORM

from ms_mint_app.duckdb_manager import _create_tables, duckdb_connection
from ms_mint_app.plugins.analysis.plugin import update_content


def _make_workspace(tmp_path):
    base = tmp_path / "mint"
    wdir = base / "workspaces" / "1"
    wdir.mkdir(parents=True)
    return wdir


def _patch_callback_context(monkeypatch, triggered=None):
    monkeypatch.setattr(
        dash,
        "callback_context",
        SimpleNamespace(triggered=triggered or []),
        raising=False,
    )


def _seed_analysis_data(conn):
    conn.execute(
        "INSERT INTO samples (ms_file_label, sample_type, ms_type, use_for_analysis, color) VALUES "
        "('S1', 'TypeA', 'ms1', TRUE, '#ff0000'),"
        "('S2', 'TypeA', 'ms1', TRUE, '#ff0000'),"
        "('S3', 'TypeB', 'ms1', TRUE, '#00ff00'),"
        "('S4', 'TypeB', 'ms1', TRUE, '#00ff00')"
    )
    conn.execute(
        "INSERT INTO targets (peak_label, ms_type) VALUES ('Peak1', 'ms1'), ('Peak2', 'ms1')"
    )
    conn.execute(
        "INSERT INTO results (peak_label, ms_file_label, peak_area) VALUES "
        "('Peak1', 'S1', 10.0), ('Peak2', 'S1', 20.0),"
        "('Peak1', 'S2', 11.0), ('Peak2', 'S2', 19.0),"
        "('Peak1', 'S3', 12.0), ('Peak2', 'S3', 18.0),"
        "('Peak1', 'S4', 13.0), ('Peak2', 'S4', 17.0)"
    )


def test_update_content_requires_analysis_context():
    with pytest.raises(dash.exceptions.PreventUpdate):
        update_content(None, "pca", None, None, [], [], "peak_area", None, "sample_type",
                       0, 0, True, True, 10, 10, "/tmp", None, None, 30)


def test_update_content_requires_wdir(monkeypatch):
    _patch_callback_context(monkeypatch)
    with pytest.raises(dash.exceptions.PreventUpdate):
        update_content({"page": "Analysis"}, "pca", None, None, [], [], "peak_area", None, "sample_type",
                       0, 0, True, True, 10, 10, None, None, None, 30)


def test_update_content_no_results(monkeypatch, tmp_path):
    _patch_callback_context(monkeypatch)
    wdir = _make_workspace(tmp_path)

    result = update_content(
        {"page": "Analysis"},
        "pca",
        None,
        None,
        [],
        [],
        "peak_area",
        None,
        "sample_type",
        0,
        0,
        True,
        True,
        10,
        10,
        str(wdir),
        None,
        None,
        30,
    )

    assert len(result) == 9
    assert result[0] is None
    assert result[3:] == ([], [], [], [], [], [])


def test_update_content_scalir_missing(monkeypatch, tmp_path):
    _patch_callback_context(monkeypatch)
    wdir = _make_workspace(tmp_path)

    with duckdb_connection(wdir, register_activity=False) as conn:
        _create_tables(conn)
        conn.execute(
            "INSERT INTO results (peak_label, ms_file_label, peak_area) VALUES ('Peak1', 'S1', 10.0)"
        )

    result = update_content(
        {"page": "Analysis"},
        "pca",
        None,
        None,
        [],
        [],
        "scalir_conc",
        None,
        "sample_type",
        0,
        0,
        True,
        True,
        10,
        10,
        str(wdir),
        None,
        None,
        30,
    )

    assert result[0] is None


def test_update_content_pca_basic(monkeypatch, tmp_path):
    _patch_callback_context(monkeypatch)
    wdir = _make_workspace(tmp_path)

    with duckdb_connection(wdir, register_activity=False) as conn:
        _create_tables(conn)
        _seed_analysis_data(conn)

    result = update_content(
        {"page": "Analysis"},
        "pca",
        None,
        None,
        [],
        [],
        "peak_area",
        "none",
        "sample_type",
        0,
        0,
        True,
        True,
        10,
        10,
        str(wdir),
        None,
        None,
        30,
    )

    fig = result[1]
    compound_options = result[4]
    assert fig is not None
    assert len(getattr(fig, "data", [])) > 0
    assert len(compound_options) == 2


def test_update_content_tsne_basic(monkeypatch, tmp_path):
    _patch_callback_context(monkeypatch)
    wdir = _make_workspace(tmp_path)

    class DummyTSNE:
        def __init__(self, n_components, perplexity, n_jobs, random_state, init):
            self.n_components = n_components

        def fit_transform(self, data):
            import numpy as np
            return np.zeros((data.shape[0], self.n_components))

    # Patch TSNE in the submodule that uses it
    monkeypatch.setattr(tsne_module, "TSNE", DummyTSNE)

    with duckdb_connection(wdir, register_activity=False) as conn:
        _create_tables(conn)
        _seed_analysis_data(conn)

    result = update_content(
        {"page": "Analysis"},
        "tsne",
        None,
        None,
        [],
        [],
        "peak_area",
        "none",
        "sample_type",
        0,
        0,
        True,
        True,
        10,
        10,
        str(wdir),
        None,
        None,
        5,
    )

    fig = result[2]
    # compound_options not expected for tsne in slots? tsne returns:
    # dash.no_update, dash.no_update, fig, dash.no_update, dash.no_update, ...
    # Wait, check plugin.py for tsne return
    # return dash.no_update, dash.no_update, fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    # So slot 4 is no_update. test checked 4.
    # The original test checked result[4]. If plugin.py changed behavior to not return options for tsne, this test will fail.
    # In tsne block of update_content: 
    # return dash.no_update, dash.no_update, fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    # So compound_options is NOT returned for tSNE tab anymore?
    # Original analysis.py likely did return options.
    # I should remove the assertion for compound_options in tsne test if it's no longer returned.
    
    assert fig is not None
    assert len(getattr(fig, "data", [])) > 0
    # assert len(compound_options) == 2 # Removed since tsne doesn't return options in plugin.py logic


def test_update_content_raincloud_basic(monkeypatch, tmp_path):
    _patch_callback_context(monkeypatch, triggered=[])
    wdir = _make_workspace(tmp_path)

    with duckdb_connection(wdir, register_activity=False) as conn:
        _create_tables(conn)
        _seed_analysis_data(conn)

    result = update_content(
        {"page": "Analysis"},
        "raincloud",
        None,
        None,
        None,
        [],
        "peak_area",
        "none",
        "sample_type",
        0,
        0,
        True,
        True,
        10,
        10,
        str(wdir),
        None,
        None,
        30,
    )

    graphs = result[3]
    options = result[4]
    selected = result[5]

    assert graphs
    assert selected in {opt["value"] for opt in options}


def test_update_content_bar_basic(monkeypatch, tmp_path):
    _patch_callback_context(monkeypatch, triggered=[])
    wdir = _make_workspace(tmp_path)

    with duckdb_connection(wdir, register_activity=False) as conn:
        _create_tables(conn)
        _seed_analysis_data(conn)

    result = update_content(
        {"page": "Analysis"},
        "bar",
        None,
        None,
        [],
        None,
        "peak_area",
        "none",
        "sample_type",
        0,
        0,
        True,
        True,
        10,
        10,
        str(wdir),
        None,
        None,
        30,
    )

    graphs = result[6]
    options = result[7]
    selected = result[8]

    assert graphs
    assert selected in {opt["value"] for opt in options}


def test_update_content_raincloud_user_selection(monkeypatch, tmp_path):
    _patch_callback_context(monkeypatch, triggered=[{"prop_id": "violin-comp-checks.value"}])
    wdir = _make_workspace(tmp_path)

    with duckdb_connection(wdir, register_activity=False) as conn:
        _create_tables(conn)
        _seed_analysis_data(conn)

    result = update_content(
        {"page": "Analysis"},
        "raincloud",
        None,
        None,
        ["Peak2"],
        [],
        "peak_area",
        "none",
        "sample_type",
        0,
        0,
        True,
        True,
        10,
        10,
        str(wdir),
        None,
        None,
        30,
    )

    selected = result[5]
    assert selected == "Peak2"


def test_update_content_bar_user_selection(monkeypatch, tmp_path):
    _patch_callback_context(monkeypatch, triggered=[{"prop_id": "bar-comp-checks.value"}])
    wdir = _make_workspace(tmp_path)

    with duckdb_connection(wdir, register_activity=False) as conn:
        _create_tables(conn)
        _seed_analysis_data(conn)

    result = update_content(
        {"page": "Analysis"},
        "bar",
        None,
        None,
        [],
        "Peak1",
        "peak_area",
        "none",
        "sample_type",
        0,
        0,
        True,
        True,
        10,
        10,
        str(wdir),
        None,
        None,
        30,
    )

    selected = result[8]
    assert selected == "Peak1"


def test_default_tsne_metric_is_zscore():
    assert TAB_DEFAULT_NORM['tsne'] == 'zscore'
