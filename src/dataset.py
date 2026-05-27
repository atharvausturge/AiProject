import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(8),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class ThumbnailDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])


def load_manifest(manifest="data/manifest.csv"):
    df = pd.read_csv(manifest)
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    return df


def make_splits(df, val_frac=0.15, test_frac=0.15, seed=42):
    """Shuffle and split into train/val/test, stratified by label."""
    parts = {"train": [], "val": [], "test": []}
    for label, grp in df.groupby("label"):
        grp = grp.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(grp)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        parts["test"].append(grp.iloc[:n_test])
        parts["val"].append(grp.iloc[n_test:n_test + n_val])
        parts["train"].append(grp.iloc[n_test + n_val:])
    return {k: pd.concat(v).reset_index(drop=True) for k, v in parts.items()}


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
