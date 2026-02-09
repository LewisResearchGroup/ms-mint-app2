import pytest

pytest.importorskip("dash.dependencies")
pytest.importorskip("feffery_antd_components")

from ms_mint_app.plugins.analysis.plugin import layout


def _find_component_by_id(node, target_id):
    if getattr(node, "id", None) == target_id:
        return node

    children = getattr(node, "children", None)
    if children is None:
        return None

    if isinstance(children, (list, tuple)):
        for child in children:
            found = _find_component_by_id(child, target_id)
            if found is not None:
                return found
        return None

    return _find_component_by_id(children, target_id)


def test_analysis_layout_defaults_to_pca_visibility():
    tree = layout()
    qc_container = _find_component_by_id(tree, "analysis-qc-container")
    pca_container = _find_component_by_id(tree, "analysis-pca-container")

    assert qc_container is not None
    assert pca_container is not None
    assert qc_container.style.get("display") == "none"
    assert pca_container.style.get("display") == "block"
