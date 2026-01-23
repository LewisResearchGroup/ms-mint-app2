from dash import html

from ms_mint_app.app import _build_layout


class _DummyExplorer:
    def layout(self):
        return html.Div("explorer", id="dummy-explorer")


def test_build_layout_basic():
    plugins = {"Workspaces": object()}
    layout = _build_layout(plugins=plugins, file_explorer=_DummyExplorer())

    # Basic sanity checks on layout structure
    assert hasattr(layout, "children")
    child_ids = {getattr(c, "id", None) for c in layout.children}
    assert "tmpdir" in child_ids
    assert "wdir" in child_ids
