import pandas as pd
import pytest

from ms_mint_app.plugins import processing as proc


def test_show_standards_filename():
    assert proc.show_standards_filename("file.csv") == "file.csv"
    assert proc.show_standards_filename(None) == "No standards file selected."


def test_reset_scalir_requires_click():
    with pytest.raises(proc.PreventUpdate):
        proc.reset_scalir(0)


def test_reset_scalir_payload():
    result = proc.reset_scalir(1)
    assert result[0] == ""
    assert result[-1] == "No standards file selected."


def test_update_scalir_plot_requires_selection():
    plots, style, path = proc.update_scalir_plot(None, {"train_frame": "x"})
    assert plots == []
    assert style["display"] == "none"
    assert path == ""


def test_update_scalir_plot_missing_store():
    with pytest.raises(proc.PreventUpdate):
        proc.update_scalir_plot("Peak1", None)


def test_update_scalir_plot_missing_frame():
    with pytest.raises(proc.PreventUpdate):
        proc.update_scalir_plot("Peak1", {"train_frame": None})


def test_update_scalir_plot_with_saved_plot(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "peak_label": ["Peak1"],
            "true_conc": [1.0],
            "value": [10.0],
            "in_range": [1],
            "pred_conc": [1.0],
        }
    )
    store = {
        "train_frame": df.to_json(orient="split"),
        "units": None,
        "params": None,
        "plot_dir": str(tmp_path),
        "generated_all_plots": True,
    }

    plot_path = tmp_path / "peak1_curve.png"
    plot_path.write_text("plot")

    monkeypatch.setattr(proc, "_plot_curve_fig", lambda *_a, **_k: "FIG")
    monkeypatch.setattr(proc, "slugify_label", lambda label: "peak1")
    monkeypatch.setattr(proc.dcc, "Graph", lambda **kwargs: {"figure": kwargs.get("figure")})

    plots, style, path_text = proc.update_scalir_plot("Peak1", store)

    assert plots[0]["figure"] == "FIG"
    assert style["display"] == "block"
    assert "peak1_curve.png" in path_text


def test_open_scalir_modal_requires_click():
    with pytest.raises(proc.PreventUpdate):
        proc.open_scalir_modal(0, "/tmp")


def test_open_scalir_modal_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(proc.dcc, "Graph", lambda **kwargs: {"figure": kwargs.get("figure")})
    monkeypatch.setattr(proc, "_plot_curve_fig", lambda *_a, **_k: "FIG")

    visible, status, conc, options, selected, plots, style, store = proc.open_scalir_modal(1, str(tmp_path))

    assert visible is True
    assert options == []
    assert selected is None
    assert plots == []
    assert style["display"] == "none"


def test_open_scalir_modal_with_results(monkeypatch, tmp_path):
    output_dir = tmp_path / "results" / "scalir"
    output_dir.mkdir(parents=True)
    (output_dir / "train_frame.csv").write_text(
        "peak_label,true_conc,value,in_range,pred_conc\nPeak1,1,10,1,1\n"
    )
    (output_dir / "standard_curve_parameters.csv").write_text("peak_label,param\nPeak1,0.5\n")
    (output_dir / "units.csv").write_text("peak_label,unit\nPeak1,uM\n")

    monkeypatch.setattr(proc.dcc, "Graph", lambda **kwargs: {"figure": kwargs.get("figure")})
    monkeypatch.setattr(proc, "_plot_curve_fig", lambda *_a, **_k: "FIG")

    visible, status, conc, options, selected, plots, style, store = proc.open_scalir_modal(1, str(tmp_path))

    assert visible is True
    assert options[0]["value"] == "Peak1"
    assert selected == "Peak1"
    assert plots[0]["figure"] == "FIG"
    assert style["display"] == "block"
    assert store["common"] == ["Peak1"]
