# Train/val/test assignment by *label*, not by image: every rendering of the same
# text lands in the same split, so validation measures generalization to unseen strings.

import hashlib


def split_for_label(text: str, val_pct: int = 5, test_pct: int = 5) -> str:
    bucket = int(hashlib.md5(text.encode()).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"
