"""Analysis package exports with lazy plugin import.

Avoid importing Dash-heavy UI modules when consumers only need lightweight
submodules (e.g., analysis._shared in tests).
"""

__all__ = ["AnalysisPlugin", "layout", "callbacks"]


def __getattr__(name):
    if name in __all__:
        from .plugin import AnalysisPlugin, layout, callbacks
        return {
            "AnalysisPlugin": AnalysisPlugin,
            "layout": layout,
            "callbacks": callbacks,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
