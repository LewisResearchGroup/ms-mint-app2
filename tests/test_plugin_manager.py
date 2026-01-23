import sys
import types

from ms_mint_app.plugin_manager import LazyPlugin, PluginManager


def test_lazy_plugin_loads_once():
    module_name = "tests.dummy_plugin_module"
    module = types.ModuleType(module_name)

    class DummyPlugin:
        init_count = 0

        def __init__(self):
            DummyPlugin.init_count += 1

        def layout(self):
            return "layout"

        def callbacks(self, app, fsc, cache, *args, **kwargs):
            return "callbacks"

    module.DummyPlugin = DummyPlugin
    sys.modules[module_name] = module

    plugin = LazyPlugin(label="Dummy", order=1, module_path=module_name, class_name="DummyPlugin")

    assert plugin.layout() == "layout"
    assert plugin.layout() == "layout"
    assert DummyPlugin.init_count == 1


def test_plugin_manager_builtin_order(monkeypatch):
    monkeypatch.setattr("ms_mint_app.plugin_manager.metadata.entry_points", lambda **_: [])

    mgr = PluginManager()
    labels = list(mgr.get_plugins().keys())

    assert labels[:3] == ["Workspaces", "MS-Files", "Targets"]
    assert labels[-1] == "Analysis"
