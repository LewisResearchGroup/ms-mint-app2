import duckdb

from ms_mint_app.plugins.analysis._shared import create_invisible_figure
from ms_mint_app.plugins.analysis.data_pipeline import _prepare_matrix_data


class _ConnCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        return False


def test_prepare_matrix_data_uses_injected_conn_factory():
    called = {}
    invisible_fig = create_invisible_figure()

    def _factory(wdir, n_cpus=None, ram=None, read_only=False):
        called["wdir"] = wdir
        called["n_cpus"] = n_cpus
        called["ram"] = ram
        called["read_only"] = read_only
        return _ConnCtx(None)

    matrix_data, outputs = _prepare_matrix_data(
        wdir="/tmp/wdir",
        metric="peak_area",
        selected_group="sample_type",
        grouping_fields=["sample_type"],
        norm_value="none",
        invisible_fig=invisible_fig,
        conn_factory=_factory,
    )

    assert matrix_data is None
    assert called["wdir"] == "/tmp/wdir"
    assert called["read_only"] is True
    assert outputs[1] is invisible_fig


def test_prepare_matrix_data_returns_empty_outputs_on_duckdb_error():
    invisible_fig = create_invisible_figure()

    class _BrokenConn:
        def execute(self, *_args, **_kwargs):
            raise duckdb.IOException("simulated I/O failure")

    def _factory(*_args, **_kwargs):
        return _ConnCtx(_BrokenConn())

    matrix_data, outputs = _prepare_matrix_data(
        wdir="/tmp/wdir",
        metric="peak_area",
        selected_group="sample_type",
        grouping_fields=["sample_type"],
        norm_value="none",
        invisible_fig=invisible_fig,
        conn_factory=_factory,
    )

    assert matrix_data is None
    assert outputs[0] is None
    assert outputs[1] is invisible_fig
