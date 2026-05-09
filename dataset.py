# File for transforming our image data into dataset ready to train our model on


import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image

def create_dataframe(df_path: str):
    img_dir = Path("diverse_math_dataset/")
    data = []

    # Using os.scandir is significantly faster than os.listdir for 60k+ files
    for entry in os.scandir(img_dir):
        if entry.is_file() and entry.name.endswith(".png"):
            try:
                parts = entry.name.split("_")
                # Safety check: ensure the filename follows the res_X_eq_Y_Z format
                if len(parts) >= 4:
                    result = parts[1]
                    equation = parts[3]

                    data.append({
                        "filename": str(entry.name),
                        "equation": equation,
                        "result": result,
                        "eq_len": len(equation)  # Helpful for padding
                    })
            except IndexError:
                print(f"Skipping malformed filename: {entry.name}")

    df = pd.DataFrame(data)
    df.to_csv(df_path, index=False)
    print(f"Success! Saved {len(df)} entries to {df_path}")


class CaptchaDataset(Dataset):
    """Captcha Dataset"""
    def __init__(self,csv_file, root_dir, transform = None):
        self.captcha_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.char_to_int = {char: i + 1 for i, char in enumerate("0123456789+-")}

    def __len__(self):
        return len(self.captcha_frame)

    def __getitem__(self, idx):
        # Join the path
        img_name = os.path.join(self.root_dir, self.captcha_frame.iloc[idx, 0])

        #  Load as PIL Image (standard for PyTorch)
        image = Image.open(img_name).convert('RGB')

        # Get the equation string and convert to list of integers
        equation_str = str(self.captcha_frame.iloc[idx, 1])
        label = [self.char_to_int[char] for char in equation_str]
        label = torch.tensor(label, dtype=torch.long)

        # Handle transforms
        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    #Create a csv file from our images
    create_dataframe(df_path="image_data.csv") #Run as a one time thing
    pass