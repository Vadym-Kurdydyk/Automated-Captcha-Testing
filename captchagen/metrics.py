from collections import defaultdict


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class SequenceMetrics:
    """Exact-match accuracy and character error rate, overall and grouped by label length."""

    def __init__(self):
        self.n = self.correct = self.edits = self.chars = 0
        self.by_length = defaultdict(lambda: [0, 0])  # length -> [correct, total]

    def update(self, preds: list[str], targets: list[str]):
        for p, t in zip(preds, targets):
            ok = p == t
            self.n += 1
            self.correct += ok
            self.edits += edit_distance(p, t)
            self.chars += len(t)
            self.by_length[len(t)][0] += ok
            self.by_length[len(t)][1] += 1

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.n)

    @property
    def cer(self) -> float:
        return self.edits / max(1, self.chars)

    def summary(self) -> dict:
        return {"accuracy": self.accuracy, "cer": self.cer, "count": self.n,
                "by_length": {k: c / t for k, (c, t) in sorted(self.by_length.items())}}
