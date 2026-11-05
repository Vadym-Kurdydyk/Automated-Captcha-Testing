# Score a checkpoint on a dataset split, per difficulty level.
#
#   python evaluate.py checkpoints/text_v1_ft/best.pth data/text_v2
#   python evaluate.py checkpoints/text_v1_ft/best.pth data/text_v2 --stretch   # old resize, for comparison

import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from captchagen.datasets import ManifestCaptchaDataset, ctc_collate
from captchagen.metrics import SequenceMetrics
from predict import CaptchaReader
from captchagen.model import greedy_decode


def main():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on a dataset split")
    p.add_argument("checkpoint")
    p.add_argument("data")
    p.add_argument("--split", default="test")
    p.add_argument("--stretch", action="store_true", help="plain resize to the model size (no aspect ratio)")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    reader = CaptchaReader(args.checkpoint)
    transform = reader.transform
    if args.stretch:
        transform = transforms.Compose([transforms.Resize((reader.height, reader.width)), transforms.Grayscale(),
                                        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    overall = SequenceMetrics()
    print(f"{args.checkpoint} on {args.data} [{args.split}]{' (stretch)' if args.stretch else ''}")
    for level in range(1, 6):
        ds = ManifestCaptchaDataset(args.data, args.split, transform, difficulties=[level])
        if ds.vocab.chars != reader.vocab.chars:
            raise SystemExit("dataset vocab doesn't match the checkpoint")
        m = SequenceMetrics()
        with torch.no_grad():
            for images, _, _, texts in DataLoader(ds, 128, num_workers=args.workers, collate_fn=ctc_collate):
                preds = [reader.vocab.decode(s) for s in greedy_decode(reader.model(images.to(reader.device)))]
                m.update(preds, texts)
                overall.update(preds, texts)
        print(f"  L{level}: acc {m.accuracy * 100:5.1f}%  CER {m.cer * 100:5.2f}%  ({m.n} images)")
    print(f"  ALL: acc {overall.accuracy * 100:5.2f}%  CER {overall.cer * 100:5.2f}%")


if __name__ == "__main__":
    main()
