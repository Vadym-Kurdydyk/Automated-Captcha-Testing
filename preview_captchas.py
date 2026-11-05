# Render a grid of samples (one row per difficulty) to eyeball the generator.
#
#   python preview_captchas.py                                  -> preview_text.png
#   python preview_captchas.py --mode math                      -> preview_math.png
#   python preview_captchas.py --random-size --style-variety    -> preview_text_varied.png

import argparse

from PIL import Image, ImageDraw

from captchagen import CaptchaGenerator, GeneratorConfig


def main():
    p = argparse.ArgumentParser(description="Preview generated captchas per difficulty")
    p.add_argument("--mode", choices=["text", "math"], default="text")
    p.add_argument("--charset", default="safe")
    p.add_argument("--random-size", action="store_true")
    p.add_argument("--style-variety", action="store_true")
    p.add_argument("--per-level", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    config = GeneratorConfig(mode=args.mode, charset=args.charset,
                             random_size=args.random_size, style_variety=args.style_variety)
    gen = CaptchaGenerator(config, seed=args.seed)
    samples = [[gen.generate(level * 1000 + col, level) for col in range(args.per_level)] for level in range(1, 6)]

    # Cells sized to the largest sample so random-size images show at their real size
    cell_w = max(s.image.width for row in samples for s in row)
    cell_h = max(s.image.height for row in samples for s in row)
    label_h, pad = 16, 6
    grid = Image.new("RGB", (pad + args.per_level * (cell_w + pad), pad + 5 * (cell_h + label_h + pad)), "white")
    d = ImageDraw.Draw(grid)
    for level, row in enumerate(samples, 1):
        for col, s in enumerate(row):
            x, y = pad + col * (cell_w + pad), pad + (level - 1) * (cell_h + label_h + pad)
            grid.paste(s.image, (x, y))
            caption = f"L{level}: {s.text}" + (f" = {s.answer}" if s.answer is not None else "")
            caption += f"  ({s.image.width}x{s.image.height})" if args.random_size else ""
            d.text((x, y + cell_h + 2), caption, fill="black")

    suffix = "_varied" if (args.random_size or args.style_variety) else ""
    out = args.out or f"preview_{args.mode}{suffix}.png"
    grid.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
