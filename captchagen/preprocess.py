# One preprocessing path for training, predict.py and the API, so the model always
# sees images prepared the same way: any size in, fixed model input out.

import numpy as np
from PIL import Image
from torchvision import transforms


def border_color(img: Image.Image) -> tuple:
    """Median color of the outermost pixels - a good guess for the background."""
    a = np.asarray(img)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(c) for c in np.median(edge, axis=0))


class FitToBox:
    """Scale to fit width x height keeping aspect ratio, pad the rest with the border color.
    A class (not a lambda) so DataLoader workers on Windows can pickle it."""

    def __init__(self, width: int = 256, height: int = 64):
        self.width, self.height = width, height

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        if img.size == (self.width, self.height):
            return img
        scale = min(self.width / img.width, self.height / img.height)
        new_w, new_h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
        resized = img.resize((new_w, new_h), Image.LANCZOS if scale < 1 else Image.BICUBIC)
        out = Image.new("RGB", (self.width, self.height), border_color(img))
        out.paste(resized, ((self.width - new_w) // 2, (self.height - new_h) // 2))
        return out


def model_transform(width: int = 256, height: int = 64):
    return transforms.Compose([FitToBox(width, height), transforms.Grayscale(), transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])
