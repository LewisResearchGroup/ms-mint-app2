import colorsys
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional

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
    new_colors = distinctipy.get_colors(len(missing), exclude_colors=exclude + [(0, 0, 0), (1, 1, 1)],
                                        pastel_factor=0.7)
    palette = predef.copy()
    palette.update({lab: distinctipy.get_hex(rgb) for lab, rgb in zip(missing, new_colors)})
    return palette


@dataclass
class PaletteParams:
    min_distance: float = 0.15
    s_range: Tuple[float, float] = (0.7, 0.95)
    v_range: Tuple[float, float] = (0.75, 0.95)
    seed: Optional[int] = None
    max_attempts: int = 400
    hue_weight: float = 2.0


class HSVPalette:
    """Compact HSV palette generator respecting existing colors."""
    _GOLDEN_ANGLE = (1 + 5 ** 0.5) / 2  # φ

    def __init__(self, params: PaletteParams | None = None):
        self.p = params or PaletteParams()
        self._rng = random.Random(self.p.seed)

    @staticmethod
    def hex_to_hsv(hex_color: str) -> Tuple[float, float, float]:
        """#RRGGBB -> HSV floats in [0,1].
        Raises ValueError on invalid input.
        """
        if not isinstance(hex_color, str) or not hex_color.startswith('#') or len(hex_color) != 7:
            raise ValueError(f"Invalid hex color: {hex_color!r}")
        try:
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Invalid hex color: {hex_color!r}") from e
        return colorsys.rgb_to_hsv(r, g, b)

    @staticmethod
    def hsv_to_hex(h: float, s: float, v: float) -> str:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)

        # precise rounding with clamp
        def _c(x: float) -> int:
            x = 0 if x < 0 else (1 if x > 1 else x)
            return int(round(x * 255))

        return f"#{_c(r):02x}{_c(g):02x}{_c(b):02x}"

    @staticmethod
    def hsv_distance(a: Tuple[float, float, float], b: Tuple[float, float, float], hue_weight: float = 2.0) -> float:
        """Euclidean distance in HSV with circular hue and hue_weight.
        Values in [0, ~3].
        """

        # circular hue delta
        d = abs(a[0] - b[0]) % 1.0
        chd = d if d <= 0.5 else 1.0 - d
        dh = chd * hue_weight
        ds = a[1] - b[1]
        dv = a[2] - b[2]
        return math.sqrt(dh * dh + ds * ds + dv * dv)

    # ---- helpers ----
    def _rand_in(self, lo: float, hi: float, jitter: float = 0.0) -> float:
        x = lo + (hi - lo) * self._rng.random()
        if jitter:
            x += (self._rng.random() * 2 - 1) * jitter
        return max(lo, min(hi, x))

    def _candidate(self, idx: int, n: int) -> Tuple[float, float, float]:
        # Golden‑angle hue sequence with slight deterministic jitter
        h = (idx * self._GOLDEN_ANGLE) % 1.0
        h = (h + self._rng.uniform(-0.015, 0.015)) % 1.0
        s = self._rand_in(*self.p.s_range, jitter=0.02 if n > 20 else 0.0)
        v = self._rand_in(*self.p.v_range, jitter=0.02 if n > 20 else 0.0)
        return (h, s, v)

    # ---- core ----
    def generate(self, n: int, *, existing: Iterable[str] | None = None) -> List[str]:
        existing_hsv: List[Tuple[float, float, float]] = []
        if existing:
            for c in existing:
                try:
                    existing_hsv.append(self.hex_to_hsv(c))
                except Exception:
                    continue

        # Slightly relax distance when many colors are requested
        min_d = self.p.min_distance if n <= 50 else max(0.08, self.p.min_distance * 0.6)

        out: List[str] = []
        out_hsv: List[Tuple[float, float, float]] = []

        for i in range(n):
            ok = False
            for attempt in range(self.p.max_attempts):
                cand = self._candidate(i + attempt, n)
                if all(self.hsv_distance(cand, e, self.p.hue_weight) >= min_d for e in existing_hsv + out_hsv):
                    out.append(self.hsv_to_hex(*cand))
                    out_hsv.append(cand)
                    ok = True
                    break
            if not ok:
                # fall back: take hue only, mid s/v
                h = (i * self._GOLDEN_ANGLE) % 1.0
                s = sum(self.p.s_range) / 2
                v = sum(self.p.v_range) / 2
                out.append(self.hsv_to_hex(h, s, v))
                out_hsv.append((h, s, v))

        return out


# -----------------------
# Convenience API (stable names)
# -----------------------

def create_palette(
        n: int,
        *,
        existing: Optional[Iterable[str]] = None,
        min_distance: float = 0.15,
        s_range: Tuple[float, float] = (0.7, 0.95),
        v_range: Tuple[float, float] = (0.75, 0.95),
        seed: Optional[int] = 42,
        max_attempts: int = 400,
        hue_weight: float = 2.0,
) -> List[str]:
    params = PaletteParams(
        min_distance=min_distance,
        s_range=s_range,
        v_range=v_range,
        seed=seed,
        max_attempts=max_attempts,
        hue_weight=hue_weight,
    )
    return HSVPalette(params).generate(n, existing=existing)


def make_palette_hsv(
        labels: Iterable[str],
        existing_map: Optional[Dict[str, str]] = None,
        **kwargs,
) -> Dict[str, str]:
    """Assign colors per unique element, preserving existing_map.

    kwargs are forwarded to create_palette.
    """
    predef = existing_map or {}
    missing = [l for l in labels if l not in existing_map]
    if not missing:
        return existing_map
    exclude = list(existing_map.values())
    new_colors = create_palette(len(missing), existing=exclude, **kwargs)
    palette = predef.copy()
    palette.update({lab: col for lab, col in zip(missing, new_colors)})
    return palette


if __name__ == '__main__':
    labels = [f'label_{i}' for i in range(100)]
    defined = {'label_0': '#fcba03', 'label_10': '#e703fc', 'label_20': '#2803fc'}
    palette = make_palette(labels, existing_map=defined)
    palette_hsv = make_palette_hsv(labels, existing_map=defined, s_range=(0.90, 0.95), v_range=(0.90, 0.95))
    distinctipy.color_swatch(palette_hsv.values())
    print(palette)
