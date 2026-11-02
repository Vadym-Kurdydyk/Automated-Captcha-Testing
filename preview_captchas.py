# Render a grid of samples (one row per difficulty) to eyeball the generator.
#
#   python preview_captchas.py                 -> preview_text.png
#   python preview_captchas.py --mode math     -> preview_math.png

import argparse

from PIL import Image, ImageDraw

from captchagen import CaptchaGenerator, GeneratorConfig


def main():
    p = argparse.ArgumentParser(description="Preview generated captchas per difficulty")
    p.add_argument("--mode", choices=["text", "math"], default="text")
    p.add_argument("--charset", default="safe")
    p.add_argument("--per-level", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    gen = CaptchaGenerator(GeneratorConfig(mode=args.mode, charset=args.charset), seed=args.seed)
    w, h, label_h, pad = gen.config.width, gen.config.height, 16, 6
    grid = Image.new("RGB", (pad + args.per_level * (w + pad), pad + 5 * (h + label_h + pad)), "white")
    d = ImageDraw.Draw(grid)
    for level in range(1, 6):
        for col in range(args.per_level):
            s = gen.generate(level * 1000 + col, level)
            x, y = pad + col * (w + pad), pad + (level - 1) * (h + label_h + pad)
            grid.paste(s.image, (x, y))
            caption = f"L{level}: {s.text}" + (f" = {s.answer}" if s.answer is not None else "")
            d.text((x, y + h + 2), caption, fill="black")

    out = args.out or f"preview_{args.mode}.png"
    grid.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
