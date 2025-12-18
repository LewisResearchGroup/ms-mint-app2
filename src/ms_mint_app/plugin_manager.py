from collections import OrderedDict
from importlib import metadata
import importlib

import sys


class LazyPlugin:
    def __init__(self, *, label, order, module_path, class_name):
        self._label = label
        self._order = order
        self._module_path = module_path
        self._class_name = class_name
        self._plugin_instance = None
        self._callbacks_registered = False

    @property
    def label(self):
        return self._label

    @property
    def order(self):
        return self._order

    def _load(self):
        if self._plugin_instance is None:
            module = importlib.import_module(self._module_path)
            cls = getattr(module, self._class_name)
            self._plugin_instance = cls()
        return self._plugin_instance

    def ensure_loaded(self):
        return self._load()

    def layout(self):
        return self._load().layout()

    def callbacks(self, app, fsc, cache, *args, **kwargs):
        return self._load().callbacks(app, fsc, cache, *args, **kwargs)

    @property
    def callbacks_registered(self):
        return self._callbacks_registered

    def mark_callbacks_registered(self):
        self._callbacks_registered = True


class PluginManager:
    def __init__(self):
        self.plugins = OrderedDict()
        self.discover_plugins()

    def discover_plugins(self):
        # Register built-in plugins lazily (avoid importing heavy modules at startup)
        builtins = [
            {
                "label": "Workspaces",
                "order": 0,
                "module": "ms_mint_app.plugins.workspaces",
                "class": "WorkspacesPlugin",
            },
            {
                "label": "MS-Files",
                "order": 1,
                "module": "ms_mint_app.plugins.ms_files",
                "class": "MsFilesPlugin",
            },
            {
                "label": "Targets",
                "order": 2,
                "module": "ms_mint_app.plugins.targets",
                "class": "TargetsPlugin",
            },
            {
                "label": "Optimization",
                "order": 6,
                "module": "ms_mint_app.plugins.target_optimization",
                "class": "TargetOptimizationPlugin",
            },
            {
                "label": "Processing",
                "order": 7,
                "module": "ms_mint_app.plugins.processing",
                "class": "ProcessingPlugin",
            },
            # {
            #     "label": "Quality Control",
            #     "order": 8,
            #     "module": "ms_mint_app.plugins.quality_control",
            #     "class": "QualityControlPlugin",
            # },
            {
                "label": "Analysis",
                "order": 9,
                "module": "ms_mint_app.plugins.analysis",
                "class": "AnalysisPlugin",
            },
        ]

        for spec in builtins:
            self.register_plugin(
                spec["label"],
                LazyPlugin(
                    label=spec["label"],
                    order=spec["order"],
                    module_path=spec["module"],
                    class_name=spec["class"],
                ),
            )

        # Discover and register external plugins
        if sys.version_info >= (3, 10):
            entry_points = metadata.entry_points(group="ms_mint_app.plugins")
        else:
            entry_points = metadata.entry_points().get("ms_mint_app.plugins", [])

        for entry_point in entry_points:
            plugin = entry_point.load()
            plugin_name = entry_point.name
            plugin_instance = plugin()
            self.register_plugin(plugin_name, plugin_instance)

    def register_plugin(self, plugin_name, plugin_instance):
        self.plugins[plugin_name] = plugin_instance

    def get_plugins(self):
        plugins = sorted(self.plugins.values(), key=lambda x: x.order)
        return {plugin.label: plugin for plugin in plugins}
