# Read captcha images with a trained checkpoint.
#
#   python predict.py checkpoints/text_v1/best.pth some.png other.png
#   python predict.py checkpoints/math_v1/best.pth folder_of_pngs/

import argparse
import re
from pathlib import Path

import torch
from PIL import Image

from captchagen.datasets import Vocab, default_transform
from captchagen.model import CRNN, greedy_decode

MATH_RE = re.compile(r"^(\d+)([+\-*])(\d+)$")


def solve_math(text: str) -> int | None:
    """Evaluate 'a<op>b' without eval(); None if the prediction isn't a valid expression."""
    m = MATH_RE.match(text)
    if not m:
        return None
    a, op, b = int(m[1]), m[2], int(m[3])
    return a + b if op == "+" else a - b if op == "-" else a * b


class CaptchaReader:
    def __init__(self, checkpoint: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.vocab = Vocab(ckpt["vocab"])
        self.height, self.width = ckpt["image_size"]
        self.is_math = ckpt["generator_config"]["mode"] == "math"
        self.model = CRNN(self.vocab.num_classes).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.transform = default_transform()

    @torch.no_grad()
    def read(self, images: list[Image.Image]) -> list[str]:
        batch = torch.stack([self.transform(img.convert("RGB").resize((self.width, self.height), Image.BILINEAR))
                             for img in images]).to(self.device)
        return [self.vocab.decode(seq) for seq in greedy_decode(self.model(batch))]


def main():
    p = argparse.ArgumentParser(description="Predict captcha text with a trained CRNN")
    p.add_argument("checkpoint")
    p.add_argument("inputs", nargs="+", help="image files or folders")
    args = p.parse_args()

    paths = []
    for inp in map(Path, args.inputs):
        paths += sorted(inp.glob("*.png")) if inp.is_dir() else [inp]

    reader = CaptchaReader(args.checkpoint)
    for i in range(0, len(paths), 64):
        chunk = paths[i:i + 64]
        for path, text in zip(chunk, reader.read([Image.open(pth) for pth in chunk])):
            extra = f"  = {solve_math(text)}" if reader.is_math else ""
            print(f"{path.name}\t{text}{extra}")


if __name__ == "__main__":
    main()
