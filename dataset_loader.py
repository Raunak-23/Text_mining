"""
dataset_loader.py
-----------------
Loads and prepares the GoEmotions dataset and the supplementary
crisis-detection dataset.  All heavy downloading is done via the
Hugging Face `datasets` library so the code is reproducible on any
machine with internet access.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROC = ROOT_DIR / "data" / "processed"
CRISIS_DIR = ROOT_DIR / "data" / "crisis"

for p in [DATA_RAW, DATA_PROC, CRISIS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ── GoEmotions label list (28 emotions + neutral) ────────────────────────────
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# ── Hierarchical emotion mapping ─────────────────────────────────────────────
EMOTION_HIERARCHY = {
    "positive": [
        "admiration", "amusement", "approval", "caring", "curiosity",
        "desire", "excitement", "gratitude", "joy", "love", "optimism",
        "pride", "realization", "relief",
    ],
    "negative": [
        "anger", "annoyance", "disappointment", "disapproval", "disgust",
        "embarrassment", "fear", "grief", "nervousness", "remorse",
        "sadness",
    ],
    "ambiguous": ["confusion", "surprise"],
    "neutral": ["neutral"],
}

# Reverse map: emotion → group
EMOTION_TO_GROUP = {
    emotion: group
    for group, emotions in EMOTION_HIERARCHY.items()
    for emotion in emotions
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GoEmotions
# ─────────────────────────────────────────────────────────────────────────────

def load_goemotions(simplified: bool = False) -> dict[str, pd.DataFrame]:
    """
    Download and return GoEmotions as a dict of DataFrames.

    Parameters
    ----------
    simplified : bool
        If True use the 'simplified' config (27 → 6 groups).
        If False use the full 28-label version.

    Returns
    -------
    dict with keys 'train', 'validation', 'test'
    """
    config = "simplified" if simplified else "raw"
    logger.info("Loading GoEmotions (%s config) from HuggingFace …", config)

    hf_dataset = load_dataset("go_emotions", config)

    splits = {}
    for split_name in ["train", "validation", "test"]:
        df = hf_dataset[split_name].to_pandas()

        if config == "raw":
            # labels column is a list of ints → binarise
            label_matrix = np.zeros((len(df), len(GOEMOTIONS_LABELS)), dtype=np.int8)
            for i, label_list in enumerate(df["labels"]):
                for lbl in label_list:
                    label_matrix[i, lbl] = 1
            label_df = pd.DataFrame(label_matrix, columns=GOEMOTIONS_LABELS)
            df = pd.concat([df[["text"]].reset_index(drop=True), label_df], axis=1)
        else:
            # simplified: labels is a list of ints mapping to 6 categories
            simple_labels = ["admiration", "amusement", "anger", "annoyance",
                             "approval", "caring", "confusion", "curiosity",
                             "desire", "disappointment", "disapproval",
                             "disgust", "embarrassment", "excitement",
                             "fear", "gratitude", "grief", "joy", "love",
                             "nervousness", "optimism", "pride", "realization",
                             "relief", "remorse", "sadness", "surprise",
                             "neutral"]
            label_matrix = np.zeros((len(df), len(simple_labels)), dtype=np.int8)
            for i, label_list in enumerate(df["labels"]):
                for lbl in label_list:
                    if lbl < len(simple_labels):
                        label_matrix[i, lbl] = 1
            label_df = pd.DataFrame(label_matrix, columns=simple_labels)
            df = pd.concat([df[["text"]].reset_index(drop=True), label_df], axis=1)

        splits[split_name] = df
        csv_path = DATA_RAW / f"goemotions_{split_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("  %s → %d rows  (%s)", split_name, len(df), csv_path)

    return splits


def add_hierarchical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary columns for each of the four top-level emotion groups:
    positive, negative, ambiguous, neutral.
    """
    for group, emotions in EMOTION_HIERARCHY.items():
        present = [e for e in emotions if e in df.columns]
        df[f"group_{group}"] = (df[present].sum(axis=1) > 0).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Crisis dataset (synthetic + curated)
# ─────────────────────────────────────────────────────────────────────────────

CRISIS_SAMPLES = {
    "crisis": [
        "I can't take this anymore, I want to end my life.",
        "Nobody would miss me if I was gone forever.",
        "I've been thinking about hurting myself every night.",
        "I have a plan to kill myself this weekend.",
        "I don't see the point in living any more.",
        "I want to disappear and never come back.",
        "I've been cutting myself to deal with the pain.",
        "Life is not worth living without them.",
        "I've already written my goodbye note.",
        "Every day I think about how I'd do it.",
        "I feel completely hopeless and trapped.",
        "The pain is so unbearable I just want it to stop permanently.",
        "I've been stockpiling pills for weeks now.",
        "Nobody can help me, I've already decided.",
        "I'm not going to be here much longer.",
        "I keep fantasizing about driving off a bridge.",
        "I finally understand why people choose to end it.",
        "No one would care if I stopped existing tomorrow.",
        "I'm saying goodbye to everyone without them knowing.",
        "I've tried everything and nothing helps, so I'm giving up.",
        "I'm reaching out because I don't know how much longer I can hold on.",
        "I hurt myself again tonight and it's getting worse.",
        "My family would be better off without me around.",
        "I feel like a burden to everyone I know.",
        "I just can't see a future for myself.",
    ],
    "non_crisis": [
        "Today was such a tough day at work, feeling exhausted.",
        "I broke up with my partner and I'm really sad.",
        "I failed my exam and I'm so disappointed in myself.",
        "My dog passed away last week and I miss him so much.",
        "I'm really stressed about this deadline approaching.",
        "Sometimes I feel like nobody understands me.",
        "I'm having a hard time sleeping lately.",
        "I feel lonely since moving to a new city.",
        "I'm worried about my parents' health.",
        "Today was overwhelming but I'll get through it.",
        "I'm struggling to stay motivated at work.",
        "I'm going through a rough patch in my relationship.",
        "I feel really disappointed with how things turned out.",
        "I'm exhausted from dealing with so many problems.",
        "I had the worst anxiety attack during the presentation.",
        "I haven't felt happy in a while but I'll figure it out.",
        "Work stress is really getting to me lately.",
        "I'm feeling really down today for no reason.",
        "I wish things were different but I know they'll improve.",
        "I cried a lot today but I feel a bit better now.",
        "I'm going to therapy to work through my issues.",
        "Talking to friends really helped me feel less alone.",
        "I've been journaling to manage my anxiety.",
        "Things are hard but I'm taking it one day at a time.",
        "I have bad days sometimes but mostly I'm okay.",
        "I called a friend when I was feeling low and it helped.",
        "I need to practice more self-care this week.",
        "Exercise always lifts my mood when I'm feeling down.",
        "I'm grateful I have support from my family.",
        "Even though it's tough, I believe things will get better.",
    ],
}


def load_crisis_dataset() -> pd.DataFrame:
    """
    Build and return the crisis/non-crisis binary dataset.
    Also saves it to data/crisis/crisis_dataset.csv.
    """
    rows = []
    for label, texts in CRISIS_SAMPLES.items():
        for t in texts:
            rows.append({"text": t, "crisis": 1 if label == "crisis" else 0})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    path = CRISIS_DIR / "crisis_dataset.csv"
    df.to_csv(path, index=False)
    logger.info("Crisis dataset: %d samples  (%s)", len(df), path)
    return df


def split_crisis(df: pd.DataFrame, test_size: float = 0.2):
    train, test = train_test_split(
        df, test_size=test_size, stratify=df["crisis"], random_state=42
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def load_all() -> dict:
    """
    Load everything and return a single dict with all splits + crisis data.
    """
    go = load_goemotions(simplified=False)
    for split in go:
        go[split] = add_hierarchical_columns(go[split])

    crisis_df = load_crisis_dataset()
    crisis_train, crisis_test = split_crisis(crisis_df)

    return {
        "goemotions": go,
        "crisis_train": crisis_train,
        "crisis_test": crisis_test,
        "label_names": GOEMOTIONS_LABELS,
        "hierarchy": EMOTION_HIERARCHY,
        "emotion_to_group": EMOTION_TO_GROUP,
    }


if __name__ == "__main__":
    data = load_all()
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            print(f"{k}: {v.shape}")
        elif isinstance(v, dict) and "train" in v:
            for split, df in v.items():
                print(f"  goemotions/{split}: {df.shape}")
