from distinctipy import distinctipy


def hex_to_rgb(hex_color):
    return int(hex_color[1:3], 16) / 255, int(hex_color[3:5], 16) / 255, int(hex_color[5:7], 16) / 255


def make_palette(labels, existing_map=None):
    """
    Genera un dict {label: "#rrggbb"} con colores distintos.
    Respeta los colores predefinidos (no los cambia).
    """
    predef = existing_map or {}
    missing = [l for l in labels if l not in existing_map]

    exclude = [hex_to_rgb(c) for c in existing_map.values()]
    new_colors = distinctipy.get_colors(len(missing), exclude_colors=exclude + [(0, 0, 0), (1, 1, 1)])

    palette = predef.copy()
    palette.update({lab: distinctipy.get_hex(rgb) for lab, rgb in zip(missing, new_colors)})
    return palette


if __name__ == '__main__':
    labels = [f'label_{i}' for i in range(100)]
    defined = {'label_0': '#fcba03', 'label_10': '#e703fc', 'label_20': '#2803fc'}
    palette = make_palette(labels, existing_map=defined)
    distinctipy.color_swatch(palette.values())
    print(palette)
