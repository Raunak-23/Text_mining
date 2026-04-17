"""
data_loader.py
==============
Handles loading of MMSD 2.0 dataset and the sample dataset.
Provides PyTorch Dataset and DataLoader wrappers for all three models.

Author: Lab Project – Meme & Sarcasm Understanding
"""

import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMG_SIZE        = 224          # Standard size for all vision models
MAX_TEXT_LEN    = 64           # Max token length for text
MEAN            = [0.485, 0.456, 0.406]   # ImageNet mean
STD             = [0.229, 0.224, 0.225]   # ImageNet std

LABEL_MAP = {0: "not_sarcastic", 1: "sarcastic"}


# ──────────────────────────────────────────────
# Image transforms
# ──────────────────────────────────────────────
def get_transform(split: str = "train"):
    """
    Returns image transform pipeline.

    Args:
        split (str): 'train' applies augmentation; 'val'/'test' does not.

    Returns:
        torchvision.transforms.Compose
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])


# ──────────────────────────────────────────────
# Core Dataset Class
# ──────────────────────────────────────────────
class MemeDataset(Dataset):
    """
    Generic Dataset for Meme Sarcasm Detection.

    Expects a list of dicts with keys:
        - 'image_path' : str   – path to the meme image
        - 'text'       : str   – OCR / caption text of the meme
        - 'label'      : int   – 0 (not sarcastic) or 1 (sarcastic)

    Args:
        samples      (list[dict])  : list of sample dicts
        tokenizer                  : HuggingFace tokenizer (optional)
        split        (str)         : 'train', 'val', or 'test'
        max_text_len (int)         : maximum token sequence length
        return_raw_text (bool)     : if True, also return raw text string
    """

    def __init__(self, samples, tokenizer=None, split="train",
                 max_text_len=MAX_TEXT_LEN, return_raw_text=False):
        self.samples       = samples
        self.tokenizer     = tokenizer
        self.split         = split
        self.max_text_len  = max_text_len
        self.return_raw    = return_raw_text
        self.transform     = get_transform(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample     = self.samples[idx]
        image_path = sample["image_path"]
        text       = sample.get("text", "")
        label      = int(sample["label"])

        # ── Load image ──────────────────────────
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Fallback: create blank image for robustness
            image = Image.fromarray(
                np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 128
            )

        image_tensor = self.transform(image)

        # ── Tokenise text ────────────────────────
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = encoding["input_ids"].squeeze(0)       # (seq_len,)
            attention_mask = encoding["attention_mask"].squeeze(0)   # (seq_len,)
        else:
            # Return dummy tensors when no tokenizer is provided
            input_ids      = torch.zeros(self.max_text_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_len, dtype=torch.long)

        result = {
            "image"         : image_tensor,          # (3, H, W)
            "input_ids"     : input_ids,              # (seq_len,)
            "attention_mask": attention_mask,         # (seq_len,)
            "label"         : torch.tensor(label, dtype=torch.long),
        }

        if self.return_raw:
            result["text"]  = text
            result["path"]  = image_path

        return result


# ──────────────────────────────────────────────
# MMSD 2.0 Loader
# ──────────────────────────────────────────────
class MMSD2Loader:
    """
    Loads the MMSD 2.0 dataset from disk.

    Expected directory layout (after downloading):
        data/raw/MMSD2/
            images/          ← meme images  (*.jpg / *.png)
            train.json       ← list of {id, text, label}
            val.json
            test.json

    If the dataset is not present, it falls back to the sample dataset.
    """

    DATASET_ROOT = Path("data/raw/MMSD2")

    def __init__(self, tokenizer=None, batch_size=32,
                 num_workers=0, max_text_len=MAX_TEXT_LEN):
        self.tokenizer    = tokenizer
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.max_text_len = max_text_len

    def _load_split(self, json_path: Path) -> list:
        """Parse a MMSD-2.0 JSON split into a list of sample dicts."""
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        samples = []
        for rec in records:
            img_path = self.DATASET_ROOT / "images" / f"{rec['id']}.jpg"
            if not img_path.exists():
                img_path = self.DATASET_ROOT / "images" / f"{rec['id']}.png"
            samples.append({
                "image_path": str(img_path),
                "text"      : rec.get("text", ""),
                "label"     : int(rec["label"]),
            })
        return samples

    def get_loaders(self):
        """
        Returns (train_loader, val_loader, test_loader).

        Falls back to SampleDatasetLoader if MMSD 2.0 is absent.
        """
        if not (self.DATASET_ROOT / "train.json").exists():
            print("[MMSD2Loader] MMSD 2.0 not found – using sample dataset.")
            return SampleDatasetLoader(
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                max_text_len=self.max_text_len,
            ).get_loaders()

        train_samples = self._load_split(self.DATASET_ROOT / "train.json")
        val_samples   = self._load_split(self.DATASET_ROOT / "val.json")
        test_samples  = self._load_split(self.DATASET_ROOT / "test.json")

        return self._make_loaders(train_samples, val_samples, test_samples)

    def _make_loaders(self, train_samples, val_samples, test_samples):
        train_ds = MemeDataset(train_samples, self.tokenizer, "train",  self.max_text_len)
        val_ds   = MemeDataset(val_samples,   self.tokenizer, "val",    self.max_text_len)
        test_ds  = MemeDataset(test_samples,  self.tokenizer, "test",   self.max_text_len)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True,  num_workers=self.num_workers,
                                  pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers,
                                  pin_memory=False)
        test_loader  = DataLoader(test_ds,  batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers,
                                  pin_memory=False)

        print(f"[MMSD2Loader] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
        return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# Sample Dataset Loader (Demo / CPU-friendly)
# ──────────────────────────────────────────────
class SampleDatasetLoader:
    """
    Loads the bundled 200-sample demo dataset from data/sample_dataset/.

    This is used for quick testing and is guaranteed to run on CPU.
    """

    SAMPLE_DIR = Path("data/sample_dataset")

    def __init__(self, tokenizer=None, batch_size=16,
                 num_workers=0, max_text_len=MAX_TEXT_LEN):
        self.tokenizer    = tokenizer
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.max_text_len = max_text_len

    def get_loaders(self, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Split sample dataset into train / val / test and return DataLoaders."""
        meta_path = self.SAMPLE_DIR / "metadata.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Sample dataset metadata not found at {meta_path}.\n"
                "Run:  python src/preprocessing.py --generate_sample"
            )

        with open(meta_path, "r") as f:
            samples = json.load(f)

        # Attach full paths
        for s in samples:
            s["image_path"] = str(self.SAMPLE_DIR / "images" / s["image_file"])

        # Reproducible split
        random.seed(seed)
        random.shuffle(samples)
        n       = len(samples)
        n_test  = max(1, int(n * test_ratio))
        n_val   = max(1, int(n * val_ratio))

        test_samples  = samples[:n_test]
        val_samples   = samples[n_test: n_test + n_val]
        train_samples = samples[n_test + n_val:]

        train_ds = MemeDataset(train_samples, self.tokenizer, "train",  self.max_text_len)
        val_ds   = MemeDataset(val_samples,   self.tokenizer, "val",    self.max_text_len)
        test_ds  = MemeDataset(test_samples,  self.tokenizer, "test",   self.max_text_len)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True,  num_workers=self.num_workers)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers)
        test_loader  = DataLoader(test_ds,  batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers)

        print(f"[SampleLoader] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
        return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# Utility: class weights for imbalanced datasets
# ──────────────────────────────────────────────
def compute_class_weights(samples: list, num_classes: int = 2) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for BCELoss / CrossEntropyLoss.

    Args:
        samples     (list[dict]) : list with 'label' keys
        num_classes (int)        : number of classes

    Returns:
        torch.Tensor of shape (num_classes,)
    """
    labels  = [s["label"] for s in samples]
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    loader = SampleDatasetLoader(batch_size=4)
    try:
        train_l, val_l, test_l = loader.get_loaders()
        batch = next(iter(train_l))
        print("Image tensor shape :", batch["image"].shape)
        print("Input IDs shape    :", batch["input_ids"].shape)
        print("Labels             :", batch["label"])
    except FileNotFoundError as e:
        print(e)
