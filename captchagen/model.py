# CRNN: CNN feature extractor -> BiLSTM over image columns -> per-column class scores for CTC.
# Input [B, 1, 64, 256] -> output [T=64, B, num_classes] log-probabilities.

import torch
import torch.nn as nn


def conv_block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            conv_block(in_channels, 64), nn.MaxPool2d(2, 2),          # 64x256 -> 32x128
            conv_block(64, 128), nn.MaxPool2d(2, 2),                  # -> 16x64
            conv_block(128, 256), conv_block(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),                             # -> 8x64  (keep width = sequence length)
            conv_block(256, 512), conv_block(512, 512),
            nn.MaxPool2d((2, 1), (2, 1)),                             # -> 4x64
            conv_block(512, 512),
            nn.AdaptiveAvgPool2d((1, None)),                          # -> 1x64
            nn.Dropout2d(dropout),
        )
        self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True, dropout=dropout)
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x).squeeze(2)        # [B, 512, T]
        f = f.permute(2, 0, 1)            # [T, B, 512]
        f, _ = self.rnn(f)
        return self.classifier(f).log_softmax(2)


def greedy_decode(log_probs: torch.Tensor) -> list[list[int]]:
    """[T, B, C] -> class indices per sample, with repeats collapsed and blanks (0) removed."""
    best = log_probs.argmax(2).transpose(0, 1).tolist()
    out = []
    for row in best:
        seq, prev = [], 0
        for idx in row:
            if idx != 0 and idx != prev:
                seq.append(idx)
            prev = idx
        out.append(seq)
    return out
