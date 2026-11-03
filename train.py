# Train the CRNN on captchas from our generator.
#
#   python train.py --data data/text_v1 --epochs 30            # dataset built by generate_dataset.py
#   python train.py --on-the-fly --mode math --epochs 30       # fresh images every epoch, nothing on disk
#
# Writes checkpoints/<name>/best.pth, last.pth and history.json.

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from captchagen import GeneratorConfig
from captchagen.datasets import ManifestCaptchaDataset, OnTheFlyCaptchaDataset, ctc_collate, default_transform
from captchagen.metrics import SequenceMetrics
from captchagen.model import CRNN, greedy_decode

LEVELS = [1, 2, 3, 4, 5]


def train_transform():
    # Light extra augmentation on top of what the generator already does (stops it memorizing the disk set)
    return transforms.Compose([
        transforms.RandomAffine(degrees=3, translate=(0.03, 0.06), scale=(0.92, 1.05), shear=5, fill=255),
        transforms.ColorJitter(0.3, 0.3),
        default_transform(),
    ])


def curriculum_levels(epoch: int, epochs: int, warmup_frac: float) -> list[int]:
    """Unlock one difficulty level at a time over the first warmup_frac of training."""
    if warmup_frac <= 0:
        return LEVELS
    max_level = 1 + int(4 * epoch / max(1, warmup_frac * epochs))
    return LEVELS[:min(5, max_level)]


class Data:
    """Hides the difference between a disk dataset and on-the-fly generation.
    With both --data and --on-the-fly, training images are generated (same generator config as the
    disk dataset) while val/test come from disk, so results stay comparable with earlier runs."""

    def __init__(self, args):
        self.args = args
        if args.data:
            self.meta_config = json.loads((Path(args.data) / "meta.json").read_text())["config"]
            self.gen_config = GeneratorConfig(**{k: v for k, v in self.meta_config.items() if k != "vocab"})
        else:
            self.gen_config = GeneratorConfig(mode=args.mode, charset=args.charset)
            self.meta_config = self.gen_config.to_dict()

    def train_set(self, levels):
        a = self.args
        if not a.on_the_fly:
            return ManifestCaptchaDataset(a.data, "train", train_transform(), difficulties=levels)
        weights = a.level_weights[:len(levels)] if a.level_weights else None
        # seed + 1: don't re-render the exact images of a disk dataset built with the same seed
        return OnTheFlyCaptchaDataset(self.gen_config, length=a.samples_per_epoch, seed=a.seed + 1,
                                      difficulties=levels, weights=weights)

    def eval_set(self, split, level):
        a = self.args
        if a.data:
            return ManifestCaptchaDataset(a.data, split, difficulties=[level])
        # Different seed per split so val and test are distinct fixed sets
        seed = a.seed + (1000 if split == "val" else 2000)
        return OnTheFlyCaptchaDataset(self.gen_config, length=a.eval_samples // 5, seed=seed,
                                      difficulties=[level], split=split)


def loader(ds, args, shuffle):
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                      collate_fn=ctc_collate, pin_memory=True, drop_last=shuffle)


def build_eval_cache(data, split, args):
    """Load an eval split into memory once (fp16), so evaluating each epoch needs no workers."""
    images, texts, levels = [], [], []
    for level in LEVELS:
        for imgs, _, _, t in loader(data.eval_set(split, level), args, shuffle=False):
            images.append(imgs.half())
            texts += t
            levels += [level] * len(t)
    return torch.cat(images), texts, levels


@torch.no_grad()
def evaluate(model, cache, args, device, vocab):
    """Returns {"overall": summary, "levels": {level: summary}}."""
    model.eval()
    images, texts, levels = cache
    preds = []
    for i in range(0, len(texts), args.batch_size):
        with torch.autocast(device.type, enabled=args.amp):
            log_probs = model(images[i:i + args.batch_size].to(device).float())
        preds += [vocab.decode(seq) for seq in greedy_decode(log_probs.float())]
    overall, per_level = SequenceMetrics(), {}
    overall.update(preds, texts)
    for level in LEVELS:
        m = SequenceMetrics()
        pairs = [(p, t) for p, t, l in zip(preds, texts, levels) if l == level]
        if pairs:
            m.update(*zip(*pairs))
        per_level[level] = m.summary()
    return {"overall": overall.summary(), "levels": per_level}


def fmt_levels(result):
    return "  ".join(f"L{l}:{s['accuracy'] * 100:5.1f}%" for l, s in result["levels"].items())


def main():
    p = argparse.ArgumentParser(description="Train a CRNN+CTC captcha reader")
    p.add_argument("--data", help="dataset folder from generate_dataset.py")
    p.add_argument("--on-the-fly", action="store_true",
                   help="generate training images during training (with --data: same config, val/test from disk)")
    p.add_argument("--init", help="checkpoint to start from (fine-tuning)")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/<name>/last.pth")
    p.add_argument("--level-weights", type=lambda s: [float(x) for x in s.split(",")], default=None,
                   help="on-the-fly only: sampling weight per level, e.g. 1,1,2,3,4")
    p.add_argument("--mode", choices=["text", "math"], default="text", help="on-the-fly without --data only")
    p.add_argument("--charset", default="safe", help="on-the-fly without --data only")
    p.add_argument("--samples-per-epoch", type=int, default=50000, help="on-the-fly only")
    p.add_argument("--eval-samples", type=int, default=5000, help="on-the-fly only")
    p.add_argument("--name", default=None, help="checkpoint folder name")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--curriculum", type=float, default=0.3,
                   help="fraction of epochs over which levels 2..5 are unlocked (0 = all levels from the start)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    args = p.parse_args()
    if not (args.data or args.on_the_fly):
        p.error("give --data, --on-the-fly, or both")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = args.amp and device.type == "cuda"
    data = Data(args)
    vocab = data.eval_set("val", 1).vocab
    name = args.name or (Path(args.data).name if args.data else f"onthefly_{args.mode}")
    out_dir = Path("checkpoints") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  classes: {vocab.num_classes}  vocab: {vocab.chars!r}  -> {out_dir}")
    val_cache = build_eval_cache(data, "val", args)
    print(f"Validation set: {len(val_cache[1])} images")

    model = CRNN(vocab.num_classes).to(device)
    if args.init:
        init = torch.load(args.init, map_location=device)
        if init["vocab"] != vocab.chars:
            raise SystemExit(f"--init checkpoint vocab {init['vocab']!r} doesn't match dataset vocab {vocab.chars!r}")
        model.load_state_dict(init["model"])
        print(f"Initialized from {args.init} (epoch {init['epoch']}, val acc {init['val']['accuracy'] * 100:.2f}%)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # The training set size changes with the curriculum, so the LR schedule is per epoch-fraction
    steps_per_epoch = len(loader(data.train_set(LEVELS), args, shuffle=True))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=steps_per_epoch, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    checkpoint_meta = {"generator_config": data.meta_config, "vocab": vocab.chars,
                       "image_size": [data.meta_config["height"], data.meta_config["width"]]}
    history, best_acc, start_epoch = [], -1.0, 0
    if args.resume:
        # Continue an interrupted run in this checkpoint folder (same command line, plus --resume)
        last = torch.load(out_dir / "last.pth", map_location=device)
        model.load_state_dict(last["model"])
        start_epoch = last["epoch"]
        history = json.loads((out_dir / "history.json").read_text())[:start_epoch]
        best_acc = max((h["val"]["overall"]["accuracy"] for h in history if len(h["levels"]) == 5), default=-1.0)
        if "optimizer" in last:
            optimizer.load_state_dict(last["optimizer"])
            scheduler.load_state_dict(last["scheduler"])
            scaler.load_state_dict(last["scaler"])
        else:  # older checkpoint without training state: at least put the LR schedule back where it was
            for _ in range(start_epoch * steps_per_epoch):
                scheduler.step()
        print(f"Resumed {out_dir} at epoch {start_epoch} (best val acc so far {best_acc * 100:.2f}%)")

    for epoch in range(start_epoch, args.epochs):
        levels = curriculum_levels(epoch, args.epochs, args.curriculum)
        train_ds = data.train_set(levels)
        if isinstance(train_ds, OnTheFlyCaptchaDataset):
            train_ds.epoch = epoch
        model.train()
        start, running, steps = time.time(), 0.0, 0

        for images, targets, target_lengths, _ in loader(train_ds, args, shuffle=True):
            if steps >= steps_per_epoch:  # keep the LR schedule in step even if this epoch has more batches
                break
            images = images.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=args.amp):
                log_probs = model(images)
            # CTC is numerically touchy in fp16, so compute it in fp32
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
            loss = criterion(log_probs.float(), targets, input_lengths, target_lengths)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            steps += 1

        # Steps skipped when the curriculum makes an epoch shorter: advance the schedule anyway
        for _ in range(steps_per_epoch - steps):
            scheduler.step()

        val = evaluate(model, val_cache, args, device, vocab)
        acc = val["overall"]["accuracy"]
        print(f"Epoch {epoch + 1:3d}/{args.epochs}  levels {levels[0]}-{levels[-1]}  "
              f"loss {running / max(1, steps):.4f}  val acc {acc * 100:5.2f}%  CER {val['overall']['cer'] * 100:5.2f}%  "
              f"[{fmt_levels(val)}]  {time.time() - start:.0f}s")

        history.append({"epoch": epoch + 1, "levels": levels, "loss": running / max(1, steps), "val": val})
        state = {"model": model.state_dict(), "epoch": epoch + 1, "val": val["overall"], **checkpoint_meta}
        torch.save({**state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict()}, out_dir / "last.pth")
        # Only count epochs that have seen every level, otherwise early easy-only epochs could "win"
        if len(levels) == 5 and acc > best_acc:
            best_acc = acc
            torch.save(state, out_dir / "best.pth")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    if not (out_dir / "best.pth").exists():
        torch.save(state, out_dir / "best.pth")
    model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device)["model"])
    test = evaluate(model, build_eval_cache(data, "test", args), args, device, vocab)
    print(f"\nTEST (best checkpoint)  acc {test['overall']['accuracy'] * 100:.2f}%  "
          f"CER {test['overall']['cer'] * 100:.2f}%  [{fmt_levels(test)}]")
    print("By length: " + "  ".join(f"{k}:{v * 100:.1f}%" for k, v in test["overall"]["by_length"].items()))
    (out_dir / "test.json").write_text(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
