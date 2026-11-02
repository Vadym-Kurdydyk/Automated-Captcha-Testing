# Build a captcha dataset on disk with our own generator.
#
#   python generate_dataset.py --out data/text_v1 --count 60000
#   python generate_dataset.py --out data/math_v1 --mode math --count 60000 --difficulty 1-5
#
# Output:
#   <out>/images/000123.png
#   <out>/labels.csv     filename,text,difficulty,split[,answer]
#   <out>/meta.json      generator config (incl. vocab) + seed, for reproducing / training

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from captchagen import CaptchaGenerator, GeneratorConfig
from captchagen.splits import split_for_label

_gen = None  # one generator per worker process


def _init_worker(config_dict, seed):
    global _gen
    _gen = CaptchaGenerator(GeneratorConfig(**config_dict), seed=seed)


def _make_one(job):
    index, difficulty, out_dir, val_pct, test_pct = job
    sample = _gen.generate(index, difficulty)
    filename = f"{index:07d}.png"
    sample.image.save(os.path.join(out_dir, "images", filename))
    return {
        "filename": filename,
        "text": sample.text,
        "difficulty": sample.difficulty,
        "split": split_for_label(sample.text, val_pct, test_pct),
        "answer": sample.answer,
    }


def parse_difficulty(s: str) -> tuple[int, int]:
    lo, _, hi = s.partition("-")
    return int(lo), int(hi or lo)


def main():
    p = argparse.ArgumentParser(description="Build a captcha dataset with our own generator")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=60000)
    p.add_argument("--mode", choices=["text", "math"], default="text")
    p.add_argument("--charset", default="safe", help="digits | upper | alnum | safe | literal characters")
    p.add_argument("--min-length", type=int, default=4)
    p.add_argument("--max-length", type=int, default=7)
    p.add_argument("--difficulty", default="1-5", help="single level '3' or range '1-5' (spread evenly)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-pct", type=int, default=5)
    p.add_argument("--test-pct", type=int, default=5)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()

    config = GeneratorConfig(mode=args.mode, charset=args.charset, min_length=args.min_length,
                             max_length=args.max_length, width=args.width, height=args.height)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    lo, hi = parse_difficulty(args.difficulty)
    levels = list(range(lo, hi + 1))
    # Cycle through levels so every difficulty gets the same share
    jobs = [(i, levels[i % len(levels)], str(out), args.val_pct, args.test_pct) for i in range(args.count)]

    config_dict = {k: v for k, v in config.to_dict().items() if k != "vocab"}
    start = time.time()
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(config_dict, args.seed)) as pool:
        for n, row in enumerate(pool.map(_make_one, jobs, chunksize=256), 1):
            rows.append(row)
            if n % 5000 == 0 or n == args.count:
                print(f"{n}/{args.count} images  ({n / (time.time() - start):.0f} img/s)")

    fields = ["filename", "text", "difficulty", "split"] + (["answer"] if args.mode == "math" else [])
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "count": args.count, "difficulty": levels,
                   "config": config.to_dict()}, f, indent=2)

    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    print(f"Done in {time.time() - start:.0f}s -> {out}  splits: {counts}")


if __name__ == "__main__":
    main()
