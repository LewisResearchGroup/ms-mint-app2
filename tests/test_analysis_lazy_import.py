import importlib
import sys
import types


def test_analysis_shared_import_does_not_eager_import_plugin(monkeypatch):
    if importlib.util.find_spec("dash") is None:
        dash_stub = types.ModuleType("dash")
        dash_stub.html = object()
        dash_stub.dcc = object()
        monkeypatch.setitem(sys.modules, "dash", dash_stub)

    if importlib.util.find_spec("dash.dependencies") is None:
        deps_stub = types.ModuleType("dash.dependencies")

        class _DummyDependency:
            def __init__(self, *args, **kwargs):
                pass

        deps_stub.Input = _DummyDependency
        deps_stub.Output = _DummyDependency
        deps_stub.State = _DummyDependency
        deps_stub.ALL = object()
        deps_stub.MATCH = object()
        monkeypatch.setitem(sys.modules, "dash.dependencies", deps_stub)

    if importlib.util.find_spec("dash.exceptions") is None:
        exc_stub = types.ModuleType("dash.exceptions")

        class _PreventUpdate(Exception):
            pass

        exc_stub.PreventUpdate = _PreventUpdate
        monkeypatch.setitem(sys.modules, "dash.exceptions", exc_stub)

    if importlib.util.find_spec("feffery_antd_components") is None:
        monkeypatch.setitem(
            sys.modules,
            "feffery_antd_components",
            types.ModuleType("feffery_antd_components"),
        )

    if importlib.util.find_spec("plotly") is None:
        plotly_stub = types.ModuleType("plotly")
        graph_objects_stub = types.ModuleType("plotly.graph_objects")
        subplots_stub = types.ModuleType("plotly.subplots")
        express_stub = types.ModuleType("plotly.express")
        colors_stub = types.ModuleType("plotly.colors")

        class _DummyFigure:
            pass

        graph_objects_stub.Figure = _DummyFigure
        subplots_stub.make_subplots = lambda *args, **kwargs: _DummyFigure()
        colors_stub.qualitative = types.SimpleNamespace(Plotly=["#1f77b4"])

        plotly_stub.graph_objects = graph_objects_stub
        plotly_stub.subplots = subplots_stub
        plotly_stub.express = express_stub
        plotly_stub.colors = colors_stub

        monkeypatch.setitem(sys.modules, "plotly", plotly_stub)
        monkeypatch.setitem(sys.modules, "plotly.graph_objects", graph_objects_stub)
        monkeypatch.setitem(sys.modules, "plotly.subplots", subplots_stub)
        monkeypatch.setitem(sys.modules, "plotly.express", express_stub)
        monkeypatch.setitem(sys.modules, "plotly.colors", colors_stub)

    if importlib.util.find_spec("sklearn") is None:
        sklearn_stub = types.ModuleType("sklearn")
        preprocessing_stub = types.ModuleType("sklearn.preprocessing")

        class _StandardScaler:
            def fit_transform(self, values):
                return values

        preprocessing_stub.StandardScaler = _StandardScaler
        sklearn_stub.preprocessing = preprocessing_stub
        monkeypatch.setitem(sys.modules, "sklearn", sklearn_stub)
        monkeypatch.setitem(sys.modules, "sklearn.preprocessing", preprocessing_stub)

    for module_name in (
        "ms_mint_app.plugins.analysis.plugin",
        "ms_mint_app.plugins.analysis._shared",
        "ms_mint_app.plugins.analysis",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    shared_module = importlib.import_module("ms_mint_app.plugins.analysis._shared")

    assert shared_module is not None
    assert "ms_mint_app.plugins.analysis.plugin" not in sys.modules
