# PyTorch datasets for our generator: one reads a dataset built by generate_dataset.py,
# the other renders fresh captchas on the fly (unlimited data, nothing on disk).

import json
import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from captchagen.generator import CaptchaGenerator, GeneratorConfig
from captchagen.splits import split_for_label


class Vocab:
    """Character <-> class index. Index 0 is reserved for the CTC blank."""

    def __init__(self, chars: str):
        self.chars = chars
        self.char_to_int = {c: i + 1 for i, c in enumerate(chars)}

    @property
    def num_classes(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char_to_int[c] for c in text], dtype=torch.long)

    def decode(self, indices) -> str:
        return "".join(self.chars[i - 1] for i in indices if i > 0)


def default_transform():
    return transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def ctc_collate(batch):
    """Stack images, concatenate variable-length labels the way nn.CTCLoss wants them."""
    images, labels, texts = zip(*batch)
    return (torch.stack(images),
            torch.cat(labels),
            torch.tensor([len(l) for l in labels], dtype=torch.long),
            list(texts))


class ManifestCaptchaDataset(Dataset):
    """Reads <root>/labels.csv + images written by generate_dataset.py."""

    def __init__(self, root: str, split: str = "train", transform=None, difficulties=None):
        self.root = root
        with open(os.path.join(root, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.vocab = Vocab(self.meta["config"]["vocab"])
        df = pd.read_csv(os.path.join(root, "labels.csv"), dtype={"text": str})
        df = df[df.split == split]
        if difficulties is not None:
            df = df[df.difficulty.isin(difficulties)]
        self.frame = df.reset_index(drop=True)
        self.transform = transform or default_transform()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(os.path.join(self.root, "images", row.filename)).convert("RGB")
        return self.transform(image), self.vocab.encode(row.text), row.text


class OnTheFlyCaptchaDataset(Dataset):
    """Renders a new captcha per index. Labels that hash into val/test are re-rolled,
    so training never sees text that a ManifestCaptchaDataset val/test split could contain
    (as long as both use the same split percentages)."""

    def __init__(self, config: GeneratorConfig | None = None, length: int = 100_000, seed: int = 0,
                 difficulties=(1, 2, 3, 4, 5), transform=None, split: str = "train",
                 val_pct: int = 5, test_pct: int = 5, weights=None):
        self.config = config or GeneratorConfig()
        self.generator = CaptchaGenerator(self.config, seed=seed)
        self.vocab = Vocab(self.config.vocab)
        self.length = length
        self.difficulties = list(difficulties)  # mutable: change it between epochs for a curriculum
        self.weights = weights                  # relative sampling weight per difficulty (None = uniform)
        self.transform = transform or default_transform()
        self.split = split
        self.val_pct, self.test_pct = val_pct, test_pct
        self.epoch = 0  # bump each epoch to get new images for the same indices

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = random.Random((self.epoch * self.length + idx) * 7919)
        difficulty = rng.choices(self.difficulties, self.weights)[0]
        # Pick the label first (cheap) and only render once it falls in our split
        while True:
            text, _ = self.generator._random_label(rng, difficulty)
            if split_for_label(text, self.val_pct, self.test_pct) == self.split:
                break
        sample = self.generator.generate(self.epoch * self.length + idx, difficulty, text=text)
        return self.transform(sample.image), self.vocab.encode(sample.text), sample.text
