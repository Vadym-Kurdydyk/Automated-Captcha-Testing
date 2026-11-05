# Read captcha images with a trained checkpoint. Any image size works: images are fitted
# into the model's input box keeping their aspect ratio (see captchagen/preprocess.py).
#
#   python predict.py checkpoints/text_v1_ft/best.pth some.png other.jpg
#   python predict.py checkpoints/math_v1/best.pth folder_of_images/

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from captchagen.datasets import Vocab
from captchagen.model import CRNN, greedy_decode, path_confidence
from captchagen.preprocess import model_transform

MATH_RE = re.compile(r"^(\d+)([+\-*])(\d+)$")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def default_checkpoint() -> str:
    """Newest trained model that exists."""
    for c in ("checkpoints/text_v2/best.pth", "checkpoints/text_v1_ft/best.pth", "checkpoints/text_v1/best.pth"):
        if Path(c).exists():
            return c
    raise SystemExit("No checkpoint found; train one or pass a checkpoint path")


def solve_math(text: str) -> int | None:
    """Evaluate 'a<op>b' without eval(); None if the prediction isn't a valid expression."""
    m = MATH_RE.match(text)
    if not m:
        return None
    a, op, b = int(m[1]), m[2], int(m[3])
    return a + b if op == "+" else a - b if op == "-" else a * b


@dataclass
class Prediction:
    text: str
    confidence: float
    answer: int | None = None


class CaptchaReader:
    def __init__(self, checkpoint: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.vocab = Vocab(ckpt["vocab"])
        self.height, self.width = ckpt["image_size"]
        self.generator_config = ckpt["generator_config"]
        self.is_math = self.generator_config["mode"] == "math"
        self.model = CRNN(self.vocab.num_classes).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.transform = model_transform(self.width, self.height)

    @torch.no_grad()
    def read(self, images: list[Image.Image]) -> list[Prediction]:
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        log_probs = self.model(batch)
        out = []
        for seq, conf in zip(greedy_decode(log_probs), path_confidence(log_probs)):
            text = self.vocab.decode(seq)
            out.append(Prediction(text, conf, solve_math(text) if self.is_math else None))
        return out


def main():
    p = argparse.ArgumentParser(description="Predict captcha text with a trained CRNN")
    p.add_argument("checkpoint")
    p.add_argument("inputs", nargs="+", help="image files or folders")
    args = p.parse_args()

    paths = []
    for inp in map(Path, args.inputs):
        paths += sorted(f for f in inp.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES) if inp.is_dir() else [inp]

    reader = CaptchaReader(args.checkpoint)
    for i in range(0, len(paths), 64):
        chunk = paths[i:i + 64]
        for path, pred in zip(chunk, reader.read([Image.open(pth) for pth in chunk])):
            extra = f"  = {pred.answer}" if reader.is_math else ""
            print(f"{path.name}\t{pred.text}{extra}\t(confidence {pred.confidence:.2f})")


if __name__ == "__main__":
    main()
