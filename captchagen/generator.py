# Our own captcha generator: random text or math expressions, rendered with
# per-character distortion and a difficulty knob (1 = clean, 5 = nasty).
# Every sample is fully determined by (seed, index), so datasets are reproducible.

import math
import os
import random
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Readable-ish fonts that ship with Windows. Anything in ./fonts is used as well.
DEFAULT_FONT_NAMES = [
    "arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf", "cour.ttf", "courbd.ttf",
    "verdana.ttf", "verdanab.ttf", "georgia.ttf", "georgiab.ttf", "comic.ttf", "comicbd.ttf",
    "impact.ttf", "tahoma.ttf", "tahomabd.ttf", "trebuc.ttf", "trebucbd.ttf", "segoeui.ttf",
    "segoeuib.ttf", "consola.ttf", "consolab.ttf", "calibri.ttf", "calibrib.ttf",
    "bahnschrift.ttf", "Candara.ttf", "corbel.ttf", "constan.ttf", "pala.ttf", "lucon.ttf",
]

CHARSETS = {
    "digits": "0123456789",
    "upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "alnum": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    # Without look-alikes (0/O/o, 1/l/I, 5/S, 2/Z, ...) - what real captchas usually use
    "safe": "23456789ABCDEFGHJKLMNPQRTUVWXYabdefghjkmnpqrty",
}

MATH_OPERATORS = "+-*"
# How a label character is drawn; the label itself stays ASCII
DISPLAY_MAP = {"*": "\u00d7"}


@dataclass
class GeneratorConfig:
    mode: str = "text"               # "text" or "math"
    charset: str = "safe"            # name from CHARSETS or a literal string of characters
    min_length: int = 4              # text mode only
    max_length: int = 7
    max_operand_digits: int = 3      # math mode only
    operators: str = MATH_OPERATORS
    width: int = 256
    height: int = 64
    font_dirs: list = field(default_factory=lambda: ["fonts", "C:/Windows/Fonts"])
    font_names: list = field(default_factory=lambda: list(DEFAULT_FONT_NAMES))

    @property
    def chars(self) -> str:
        return CHARSETS.get(self.charset, self.charset)

    @property
    def vocab(self) -> str:
        """Every character a label can contain, in a stable order (for the model's class map)."""
        if self.mode == "math":
            return "0123456789" + self.operators
        return "".join(sorted(set(self.chars)))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vocab"] = self.vocab
        return d


@dataclass
class Sample:
    image: Image.Image
    text: str
    difficulty: int
    answer: int | None = None        # math mode: the evaluated result


def find_fonts(config: GeneratorConfig) -> list[str]:
    found = []
    for d in config.font_dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        if d == "fonts":
            # Project font folder: take everything
            found += sorted(str(f) for f in p.iterdir() if f.suffix.lower() in (".ttf", ".otf"))
        else:
            found += [str(p / n) for n in config.font_names if (p / n).is_file()]
    if not found:
        raise RuntimeError(f"No fonts found in {config.font_dirs}; put some .ttf files in ./fonts")
    return found


@lru_cache(maxsize=512)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


class CaptchaGenerator:
    def __init__(self, config: GeneratorConfig | None = None, seed: int = 0):
        self.config = config or GeneratorConfig()
        self.seed = seed
        self.fonts = find_fonts(self.config)

    # ---------- public API ----------

    def generate(self, index: int, difficulty: int, text: str | None = None) -> Sample:
        """Deterministic: the same (seed, index, difficulty) always gives the same image."""
        rng = random.Random(self.seed * 1_000_003 + index)
        np_rng = np.random.default_rng(self.seed * 1_000_003 + index)
        difficulty = max(1, min(5, difficulty))
        t = (difficulty - 1) / 4  # 0..1 strength used by every effect

        answer = None
        if text is None:
            text, answer = self._random_label(rng, difficulty)

        bg = self._background(rng, t)
        text_color = self._text_color(rng, bg)
        self._draw_noise(bg, rng, t, under_text=True)
        text_layer = self._render_text(text, rng, t, text_color)
        self._paste_centered(bg, text_layer, rng, t)
        self._draw_noise(bg, rng, t, under_text=False, text_color=text_color)

        img = self._warp(bg, np_rng, rng, t)
        img = self._speckle(img, np_rng, t)
        if t > 0 and rng.random() < 0.3 + 0.5 * t:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.4 + 0.9 * t)))
        return Sample(img, text, difficulty, answer)

    # ---------- labels ----------

    def _random_label(self, rng: random.Random, difficulty: int) -> tuple[str, int | None]:
        cfg = self.config
        if cfg.mode == "math":
            op = rng.choice(cfg.operators)
            # Harder captchas allow longer operands; multiplication stays small
            digits = 1 + round((cfg.max_operand_digits - 1) * (difficulty - 1) / 4)
            if op == "*":
                digits = min(digits, 2)
            a = rng.randint(0, 10 ** rng.randint(1, digits) - 1)
            b = rng.randint(0, 10 ** rng.randint(1, digits) - 1)
            answer = a + b if op == "+" else a - b if op == "-" else a * b
            return f"{a}{op}{b}", answer
        length = rng.randint(cfg.min_length, cfg.max_length)
        return "".join(rng.choice(cfg.chars) for _ in range(length)), None

    # ---------- background ----------

    def _background(self, rng: random.Random, t: float) -> Image.Image:
        w, h = self.config.width, self.config.height
        base = tuple(rng.randint(200, 255) if rng.random() < 0.7 else rng.randint(0, 255) for _ in range(3))
        if rng.random() > 0.2 + 0.6 * t:
            return Image.new("RGB", (w, h), base)
        # Linear gradient between two colors, at a random angle
        other = tuple(max(0, min(255, c + rng.randint(-120, 120))) for c in base)
        angle = rng.uniform(0, math.pi)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        proj = xs * math.cos(angle) + ys * math.sin(angle)
        proj = (proj - proj.min()) / max(1e-6, proj.max() - proj.min())
        arr = np.array(base, np.float32) * (1 - proj[..., None]) + np.array(other, np.float32) * proj[..., None]
        return Image.fromarray(arr.astype(np.uint8), "RGB")

    @staticmethod
    def _luminance(c) -> float:
        return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]

    def _text_color(self, rng: random.Random, bg: Image.Image) -> tuple:
        # Keep the text legible against the average background
        bg_lum = self._luminance(np.asarray(bg).reshape(-1, 3).mean(0))
        for _ in range(50):
            c = tuple(rng.randint(0, 255) for _ in range(3))
            if abs(self._luminance(c) - bg_lum) > 90:
                return c
        return (0, 0, 0) if bg_lum > 127 else (255, 255, 255)

    # ---------- text ----------

    def _render_text(self, text: str, rng: random.Random, t: float, color: tuple) -> Image.Image:
        h = self.config.height
        font_path = rng.choice(self.fonts)
        mixed_fonts = t > 0.5 and rng.random() < 0.3   # hardest levels: font changes per char
        base_size = int(h * rng.uniform(0.75, 0.95))
        max_rot = 5 + 30 * t
        spacing = rng.uniform(2 - 8 * t, 6 - 5 * t)    # negative = overlapping glyphs

        glyphs = []
        for ch in text:
            size = max(10, int(base_size * rng.uniform(1 - 0.25 * t, 1 + 0.15 * t)))
            font = load_font(rng.choice(self.fonts) if mixed_fonts else font_path, size)
            glyph = DISPLAY_MAP.get(ch, ch)
            l, tp, r, b = font.getbbox(glyph)
            tile = Image.new("L", (r - l + 8, b - tp + 8), 0)
            ImageDraw.Draw(tile).text((4 - l, 4 - tp), glyph, font=font, fill=255)
            tile = tile.rotate(rng.uniform(-max_rot, max_rot), resample=Image.BICUBIC, expand=True)
            if t > 0 and rng.random() < 0.5 * t:      # random shear
                shear = rng.uniform(-0.35, 0.35) * t
                tile = tile.transform(tile.size, Image.AFFINE, (1, shear, -shear * tile.height / 2, 0, 1, 0),
                                      resample=Image.BICUBIC)
            glyphs.append(tile.crop(tile.getbbox()) if tile.getbbox() else tile)

        total_w = sum(g.width for g in glyphs) + int(spacing) * (len(glyphs) - 1) + 8
        max_h = max(g.height for g in glyphs)
        canvas_h = int(max_h * (1 + 0.4 * t)) + 8
        mask = Image.new("L", (max(total_w, 8), canvas_h), 0)
        x = 4
        for g in glyphs:
            y = (canvas_h - g.height) // 2 + int(rng.uniform(-0.2, 0.2) * t * max_h)
            y = max(0, min(canvas_h - g.height, y))
            # Max-blend so overlapping glyphs merge instead of erasing each other
            region = mask.crop((x, y, x + g.width, y + g.height))
            mask.paste(Image.fromarray(np.maximum(np.asarray(region), np.asarray(g))), (x, y))
            x += g.width + int(spacing)

        mask = mask.crop(mask.getbbox())
        layer = Image.new("RGBA", mask.size, color + (0,))
        layer.putalpha(mask)
        return layer

    def _paste_centered(self, bg: Image.Image, layer: Image.Image, rng: random.Random, t: float):
        w, h = bg.size
        # Fit inside the canvas with a margin, keeping aspect ratio
        scale = min((w - 8) / layer.width, (h - 6) / layer.height, 1.0)
        scale *= rng.uniform(0.85, 1.0)
        layer = layer.resize((max(1, int(layer.width * scale)), max(1, int(layer.height * scale))), Image.LANCZOS)
        free_x, free_y = w - layer.width, h - layer.height
        cx = free_x // 2 + int(rng.uniform(-0.5, 0.5) * free_x * (0.3 + 0.7 * t))
        cy = free_y // 2 + int(rng.uniform(-0.5, 0.5) * free_y)
        bg.paste(layer, (max(0, min(free_x, cx)), max(0, min(free_y, cy))), layer)

    # ---------- noise ----------

    def _rand_color(self, rng: random.Random) -> tuple:
        return tuple(rng.randint(0, 255) for _ in range(3))

    def _draw_noise(self, img: Image.Image, rng: random.Random, t: float, under_text: bool, text_color=None):
        w, h = img.size
        d = ImageDraw.Draw(img)
        if under_text:
            # Background clutter: dots and faint arcs
            for _ in range(int(rng.uniform(0, 1) * (20 + 300 * t))):
                x, y, r = rng.randint(0, w), rng.randint(0, h), rng.randint(0, 1 + int(2 * t))
                d.ellipse((x - r, y - r, x + r, y + r), fill=self._rand_color(rng))
            for _ in range(rng.randint(0, 1 + int(4 * t))):
                x0, y0 = rng.randint(-w // 2, w), rng.randint(-h, h)
                d.arc((x0, y0, x0 + rng.randint(w // 4, w), y0 + rng.randint(h // 2, 2 * h)),
                      rng.uniform(0, 360), rng.uniform(0, 360), fill=self._rand_color(rng), width=rng.randint(1, 2))
            return
        # Lines crossing the text; at higher levels some use the text color so they can't be filtered by color
        for _ in range(rng.randint(0, int(1 + 4 * t))):
            color = text_color if (text_color and rng.random() < 0.6 * t) else self._rand_color(rng)
            pts = [(rng.randint(0, w // 4), rng.randint(0, h))]
            for _ in range(rng.randint(1, 3)):
                pts.append((min(w, pts[-1][0] + rng.randint(w // 6, w // 2)), rng.randint(0, h)))
            d.line(pts, fill=color, width=rng.randint(1, 1 + int(2 * t)))

    def _speckle(self, img: Image.Image, np_rng: np.random.Generator, t: float) -> Image.Image:
        if t == 0:
            return img
        arr = np.asarray(img).astype(np.int16)
        arr += np_rng.normal(0, 4 + 14 * t, arr.shape).astype(np.int16)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    def _warp(self, img: Image.Image, np_rng: np.random.Generator, rng: random.Random, t: float) -> Image.Image:
        """Sine-wave displacement in both directions."""
        if t == 0:
            return img
        w, h = img.size
        ax, ay = rng.uniform(0, 1.5 + 3 * t), rng.uniform(0, 1 + 2.5 * t)
        fx, fy = rng.uniform(0.02, 0.08), rng.uniform(0.05, 0.2)
        px, py = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
        ys, xs = np.mgrid[0:h, 0:w]
        src_x = np.clip(xs + ax * np.sin(ys * fy + px), 0, w - 1).astype(np.int32)
        src_y = np.clip(ys + ay * np.sin(xs * fx + py), 0, h - 1).astype(np.int32)
        arr = np.asarray(img)
        return Image.fromarray(arr[src_y, src_x], "RGB")
