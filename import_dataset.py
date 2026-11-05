# Turn a folder of your own labeled captcha images into a dataset train.py can fine-tune on.
# Only use captchas you own or are authorized to test.
#
# Labels come from either:
#   - the file name: "AB12C.png", "AB12C_7.png" (text = everything before the first "_"), or
#   - a CSV with columns filename,text (or filename,label):  --labels my_labels.csv
#
#   python import_dataset.py --src my_captchas/ --out data/mine
#   python import_dataset.py --src scans/ --labels scans/labels.csv --out data/mine --case upper
#
# Output has the same layout as generate_dataset.py (images/, labels.csv, meta.json);
# every image gets difficulty 0 ("real"). Splits are by label, like our generated data.

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from captchagen import GeneratorConfig
from captchagen.splits import split_for_label

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def labels_from_csv(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    key = "text" if rows and "text" in rows[0] else "label"
    return {r["filename"]: r[key] for r in rows}


def main():
    p = argparse.ArgumentParser(description="Import your own labeled captchas as a training dataset")
    p.add_argument("--src", required=True, help="folder with captcha images")
    p.add_argument("--labels", help="CSV with filename,text; default: label from the file name")
    p.add_argument("--out", required=True)
    p.add_argument("--case", choices=["keep", "upper", "lower"], default="keep",
                   help="normalize labels if your captcha isn't case-sensitive")
    p.add_argument("--val-pct", type=int, default=10)
    p.add_argument("--test-pct", type=int, default=10)
    p.add_argument("--width", type=int, default=256, help="model input width (keep the default for fine-tuning)")
    p.add_argument("--height", type=int, default=64)
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    csv_labels = labels_from_csv(Path(args.labels)) if args.labels else None
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows, skipped = [], Counter()
    for f in sorted(src.iterdir()):
        if f.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        text = csv_labels.get(f.name) if csv_labels is not None else f.stem.split("_")[0]
        text = (text or "").strip().replace(" ", "")
        text = text.upper() if args.case == "upper" else text.lower() if args.case == "lower" else text
        if not text:
            skipped["no label"] += 1
            continue
        try:
            with Image.open(f) as img:
                img.verify()
        except (UnidentifiedImageError, OSError):
            skipped["unreadable image"] += 1
            continue
        filename = f"{len(rows):07d}{f.suffix.lower()}"
        shutil.copy2(f, out / "images" / filename)
        rows.append({"filename": filename, "text": text, "difficulty": 0,
                     "split": split_for_label(text, args.val_pct, args.test_pct), "source": f.name})

    if not rows:
        raise SystemExit(f"No labeled images found in {src}")

    chars = "".join(sorted(set("".join(r["text"] for r in rows))))
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "text", "difficulty", "split", "source"])
        w.writeheader()
        w.writerows(rows)
    # The config describes the labels (charset/vocab) and model input size; the generator
    # can also render synthetic captchas with this charset (used by server.py's challenges)
    config = GeneratorConfig(charset=chars, width=args.width, height=args.height)
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump({"imported_from": str(src), "count": len(rows), "difficulty": [0], "config": config.to_dict()},
                  fh, indent=2)

    splits = Counter(r["split"] for r in rows)
    lengths = [len(r["text"]) for r in rows]
    print(f"Imported {len(rows)} images -> {out}  splits: {dict(splits)}")
    print(f"Characters ({len(chars)}): {chars!r}   label length {min(lengths)}-{max(lengths)}")
    if skipped:
        print(f"Skipped: {dict(skipped)}")
    if len(rows) < 500:
        print("Tip: a few hundred images is a minimum; 1000+ works much better. "
              "Use --mix-synthetic when training so the model doesn't just memorize them.")


if __name__ == "__main__":
    main()
