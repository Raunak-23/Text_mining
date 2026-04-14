"""
preprocessing.py
================
Handles all data preprocessing steps:
  1. Image cleaning & normalisation checks
  2. Text cleaning (emoji, HTML, stopword removal)
  3. Sample dataset generation (synthetic meme records for CPU demo)
  4. Saving processed metadata to data/processed/

Author: Lab Project – Meme & Sarcasm Understanding
"""

import os
import re
import json
import random
import argparse
import unicodedata
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SAMPLE_DIR    = Path("data/sample_dataset")
PROCESSED_DIR = Path("data/processed")
IMG_DIR       = SAMPLE_DIR / "images"


# ──────────────────────────────────────────────
# 1. TEXT PREPROCESSING
# ──────────────────────────────────────────────

# Common English stopwords (lightweight, no NLTK needed)
STOPWORDS = {
    "a","an","the","is","it","in","of","on","at","to","for",
    "and","or","but","not","with","this","that","are","was","were",
    "be","been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","can","need","dare",
    "i","me","my","you","your","he","his","she","her","we","our","they",
    "their","its","us","him","who","which","what","when","where","how",
}

HTML_TAG_RE   = re.compile(r"<[^>]+>")
URL_RE        = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE    = re.compile(r"@\w+")
HASHTAG_RE    = re.compile(r"#(\w+)")
MULTI_SPACE   = re.compile(r"\s+")


def remove_html(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text)


def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def remove_mentions(text: str) -> str:
    return MENTION_RE.sub(" ", text)


def expand_hashtags(text: str) -> str:
    """Convert #ThisIsSarcastic → 'ThisIsSarcastic'."""
    return HASHTAG_RE.sub(r"\1", text)


def remove_emojis(text: str) -> str:
    """Strip characters outside the Basic Multilingual Plane (most emojis)."""
    return "".join(c for c in text if ord(c) <= 0xFFFF)


def normalise_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def remove_special_chars(text: str) -> str:
    """Keep only alphanumerics, spaces, basic punctuation."""
    return re.sub(r"[^a-zA-Z0-9\s.,!?'\"()-]", " ", text)


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_stopwords(text: str, keep_negations: bool = True) -> str:
    """
    Remove stopwords.

    Args:
        keep_negations (bool): Keep 'not', 'no', 'never' etc. for sentiment.
    """
    negations = {"not", "no", "never", "neither", "nor", "none"}
    tokens    = text.split()
    if keep_negations:
        return " ".join(t for t in tokens if t not in (STOPWORDS - negations))
    return " ".join(t for t in tokens if t not in STOPWORDS)


def clean_text(text: str, remove_stops: bool = False) -> str:
    """
    Full text cleaning pipeline.

    Args:
        text          (str)  : raw meme text
        remove_stops  (bool) : whether to strip stopwords

    Returns:
        str: cleaned text
    """
    text = normalise_unicode(text)
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = expand_hashtags(text)
    text = remove_emojis(text)
    text = remove_special_chars(text)
    text = to_lowercase(text)
    text = MULTI_SPACE.sub(" ", text).strip()
    if remove_stops:
        text = remove_stopwords(text)
    return text


def compute_text_stats(texts: list) -> dict:
    """Return vocabulary size, avg/max/min token counts."""
    lengths = [len(t.split()) for t in texts]
    vocab   = set(" ".join(texts).split())
    return {
        "vocab_size"  : len(vocab),
        "avg_len"     : float(np.mean(lengths)),
        "max_len"     : int(np.max(lengths)),
        "min_len"     : int(np.min(lengths)),
    }


# ──────────────────────────────────────────────
# 2. IMAGE PREPROCESSING UTILITIES
# ──────────────────────────────────────────────

def verify_image(path: str) -> bool:
    """Check that the file is a readable RGB image."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def resize_and_save(src_path: str, dst_path: str, size=(224, 224)) -> bool:
    """Resize an image to `size` and save to `dst_path`."""
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        img.save(dst_path)
        return True
    except Exception as e:
        print(f"[WARNING] Could not process {src_path}: {e}")
        return False


def preprocess_dataset_images(src_dir: Path, dst_dir: Path, size=(224, 224)) -> dict:
    """
    Resize all images in src_dir and copy to dst_dir.

    Returns a summary dict.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    success, failed = 0, 0
    for img_file in src_dir.glob("*.*"):
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        dst_file = dst_dir / img_file.name
        if resize_and_save(str(img_file), str(dst_file), size):
            success += 1
        else:
            failed += 1
    return {"processed": success, "failed": failed}


# ──────────────────────────────────────────────
# 3. SYNTHETIC SAMPLE DATASET GENERATOR
# ──────────────────────────────────────────────

SARCASTIC_TEXTS = [
    "Oh sure, because Mondays are just SO wonderful",
    "Wow, another meeting that could have been an email. How exciting!",
    "Great, it's raining again. My absolute favourite weather.",
    "Yeah, traffic jams are the best way to spend your morning",
    "Oh fantastic, the WiFi went out. Just what I needed today.",
    "Because nothing says fun like doing taxes on a Sunday",
    "Sure, I LOVE when my coffee goes cold. Perfectly planned!",
    "Oh brilliant, the printer is jammed. As always. Shocking.",
    "My alarm works GREAT. At the wrong time.",
    "Wonderful! The presentation crashed in front of everyone.",
    "Oh yes, please tell me MORE about how you're always right",
    "Nothing better than a 3-hour meeting with no agenda",
    "Just love when someone spoils the ending. So thoughtful!",
    "Oh sure, let me drop everything for a low-priority task",
    "My code works perfectly... on someone else's machine",
    "Oh great, another software update that breaks everything",
    "Sure, let the intern fix the production server. Smart move.",
    "Oh look, the deadline moved up. Totally expected that one.",
    "Yeah, gym at 5AM. Because sleep is overrated apparently.",
    "Oh wow, the client changed requirements AGAIN. Surprise!",
]

NON_SARCASTIC_TEXTS = [
    "Beautiful sunrise this morning, feeling grateful",
    "Just finished a great book on deep learning techniques",
    "The team worked hard and delivered the project on time",
    "Looking forward to the weekend hiking trip",
    "Coffee and code, the perfect morning combination",
    "Proud of my students for their amazing research work",
    "The neural network converged faster than expected today",
    "Had a very productive meeting with the research team",
    "Enjoying the cool breeze after a long day of coding",
    "The model achieved 92% accuracy on the validation set",
    "Grateful for another day to work on what I love",
    "The dataset preprocessing pipeline is finally complete",
    "Learned something new about transformer architectures today",
    "Great weather for a run in the morning",
    "The experiment results are looking very promising",
    "Our paper was accepted to the conference",
    "The system is running smoothly after the upgrade",
    "Had a great discussion with collaborators today",
    "The new framework makes training much faster",
    "Looking forward to presenting our findings next week",
]

SARCASTIC_COLORS   = [(255, 220, 100), (200, 230, 255), (255, 200, 200)]
NORMAL_COLORS      = [(230, 255, 230), (255, 245, 220), (220, 220, 255)]


def _create_meme_image(text: str, label: int, idx: int, out_dir: Path) -> str:
    """
    Generate a synthetic meme image with text overlay.

    Args:
        text    : meme caption
        label   : 0 or 1
        idx     : sample index (for unique filenames)
        out_dir : output directory

    Returns:
        filename (str)
    """
    W, H    = 224, 224
    palette = SARCASTIC_COLORS if label == 1 else NORMAL_COLORS
    bg      = random.choice(palette)

    img  = Image.new("RGB", (W, H), color=bg)
    draw = ImageDraw.Draw(img)

    # Draw a simple border
    border_color = (100, 100, 100)
    draw.rectangle([0, 0, W - 1, H - 1], outline=border_color, width=3)

    # Word-wrap the caption text
    words      = text.split()
    lines      = []
    line_words = []
    for word in words:
        line_words.append(word)
        if len(" ".join(line_words)) > 22:
            lines.append(" ".join(line_words[:-1]))
            line_words = [word]
    if line_words:
        lines.append(" ".join(line_words))

    # Draw text (no custom font needed – PIL default is fine)
    y = 20
    for line in lines[:8]:                      # cap at 8 lines
        draw.text((10, y), line, fill=(30, 30, 30))
        y += 22

    # Draw a label badge (🎭 / ✅ indicator)
    badge_text  = "SARCASM" if label == 1 else "NORMAL"
    badge_color = (220, 50, 50) if label == 1 else (50, 150, 50)
    draw.rectangle([W - 80, H - 28, W - 1, H - 1], fill=badge_color)
    draw.text((W - 74, H - 22), badge_text, fill=(255, 255, 255))

    fname = f"sample_{idx:04d}_label{label}.png"
    img.save(out_dir / fname)
    return fname


def generate_sample_dataset(n_samples: int = 200, seed: int = 42) -> list:
    """
    Generate a synthetic meme dataset of n_samples records.

    Files are saved to data/sample_dataset/images/.
    Metadata is saved to data/sample_dataset/metadata.json.

    Args:
        n_samples (int): total samples (balanced between classes)
        seed      (int): random seed

    Returns:
        list[dict]: metadata records
    """
    random.seed(seed)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    records      = []
    half         = n_samples // 2
    sarcastic    = random.choices(SARCASTIC_TEXTS,     k=half)
    non_sarcastic = random.choices(NON_SARCASTIC_TEXTS, k=half)

    all_samples = [(t, 1) for t in sarcastic] + [(t, 0) for t in non_sarcastic]
    random.shuffle(all_samples)

    for idx, (text, label) in enumerate(all_samples):
        fname = _create_meme_image(text, label, idx, IMG_DIR)
        records.append({
            "id"         : idx,
            "image_file" : fname,
            "text"       : text,
            "label"      : label,
            "clean_text" : clean_text(text),
        })

    # Save metadata
    meta_path = SAMPLE_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)

    # Also save a CSV version
    import csv
    csv_path = SAMPLE_DIR / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image_file", "text", "label", "clean_text"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[SampleGen] Created {n_samples} samples in {SAMPLE_DIR}")
    print(f"  Sarcastic    : {sum(r['label'] == 1 for r in records)}")
    print(f"  Non-sarcastic: {sum(r['label'] == 0 for r in records)}")
    return records


# ──────────────────────────────────────────────
# 4. MMSD 2.0 Preprocessing helper
# ──────────────────────────────────────────────

def preprocess_mmsd2(raw_dir: Path = Path("data/raw/MMSD2"),
                     out_dir: Path = Path("data/processed")) -> None:
    """
    Clean all text fields in MMSD 2.0 JSON splits and resize images.

    This writes *_cleaned.json files to data/processed/ and resized images
    to data/processed/images/.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    for split in ("train", "val", "test"):
        json_path = raw_dir / f"{split}.json"
        if not json_path.exists():
            print(f"[MMSD2] {json_path} not found – skipping.")
            continue

        with open(json_path, "r") as f:
            records = json.load(f)

        cleaned = []
        for rec in records:
            rec["clean_text"] = clean_text(rec.get("text", ""))
            cleaned.append(rec)

        out_path = out_dir / f"{split}_cleaned.json"
        with open(out_path, "w") as f:
            json.dump(cleaned, f, indent=2)
        print(f"[MMSD2] Wrote {out_path}  ({len(cleaned)} records)")

    # Resize images
    src_img = raw_dir / "images"
    dst_img = out_dir / "images"
    if src_img.exists():
        stats = preprocess_dataset_images(src_img, dst_img)
        print(f"[MMSD2] Images – processed={stats['processed']}  failed={stats['failed']}")


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meme Sarcasm – Preprocessing")
    parser.add_argument("--generate_sample", action="store_true",
                        help="Generate synthetic sample dataset")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of synthetic samples (default: 200)")
    parser.add_argument("--preprocess_mmsd2", action="store_true",
                        help="Preprocess the full MMSD 2.0 dataset")
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_dataset(n_samples=args.n_samples)

    if args.preprocess_mmsd2:
        preprocess_mmsd2()

    if not args.generate_sample and not args.preprocess_mmsd2:
        print("Nothing to do. Use --generate_sample or --preprocess_mmsd2")
        parser.print_help()
