from collections import OrderedDict
from importlib import metadata

import sys

from .plugins.workspaces import WorkspacesPlugin
from .plugins.ms_files import MsFilesPlugin
from .plugins.targets import TargetsPlugin
from .plugins.target_optimization import TargetOptimizationPlugin
from .plugins.processing import ProcessingPlugin
from .plugins.quality_control import QualityControlPlugin
from .plugins.analysis import AnalysisPlugin
from .plugins.ms2_browser import MS2BrowserPlugin


class PluginManager:
    def __init__(self):
        self.plugins = OrderedDict()
        self.discover_plugins()

    def discover_plugins(self):
        # Register built-in plugins with order attribute
        self.register_plugin("Workspaces", WorkspacesPlugin())
        self.register_plugin("MS-Files", MsFilesPlugin())
        self.register_plugin("Targets", TargetsPlugin())
        #self.register_plugin("Add Metabolites", AddMetabolitesPlugin())
        self.register_plugin("Optimization", TargetOptimizationPlugin())
        # add the new MS2 browser plugin
        self.register_plugin("MS2 Browser", MS2BrowserPlugin())
        self.register_plugin("Processing", ProcessingPlugin())
        self.register_plugin("Quality Control", QualityControlPlugin())
        self.register_plugin("Analysis", AnalysisPlugin())

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
