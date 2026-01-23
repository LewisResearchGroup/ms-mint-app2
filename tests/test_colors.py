from ms_mint_app.colors import hex_to_rgb, create_palette, make_palette_hsv


def test_hex_to_rgb():
    r, g, b = hex_to_rgb("#ff0000")
    assert r == 1.0
    assert g == 0.0
    assert b == 0.0


def test_create_palette_length_and_format():
    palette = create_palette(5, seed=123)
    assert len(palette) == 5
    assert all(c.startswith("#") and len(c) == 7 for c in palette)


def test_make_palette_hsv_respects_existing():
    labels = ["A", "B", "C"]
    existing = {"A": "#112233"}
    palette = make_palette_hsv(labels, existing_map=existing, seed=1)
    assert palette["A"] == "#112233"
    assert "B" in palette
    assert "C" in palette
