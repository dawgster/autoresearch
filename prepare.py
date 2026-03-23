"""
Data preparation and evaluation harness for SN32 AI text detection.
Downloads/generates training data and provides fixed evaluation.

Usage:
    python prepare.py              # full prep
    python prepare.py --small      # small dataset for quick testing

DO NOT MODIFY — this is the fixed evaluation harness.
"""

import os
import json
import random
import time
import re
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score, average_precision_score

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / "data"
TIME_BUDGET = 600  # 10 minutes training budget
EVAL_SAMPLES = 2000  # number of samples for validation
SEED = 42

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _add_augmentation(text: str) -> str:
    """Mimic SN32 validator augmentations: misspellings, adjective removal."""
    # Select random consecutive sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 3:
        start = random.randint(0, max(0, len(sentences) - 3))
        end = start + random.randint(2, min(5, len(sentences) - start))
        sentences = sentences[start:end]
    text = " ".join(sentences)

    # Random misspelling (swap adjacent chars)
    if random.random() < 0.3 and len(text) > 10:
        idx = random.randint(1, len(text) - 2)
        text = text[:idx] + text[idx + 1] + text[idx] + text[idx + 2:]

    return text


def prepare_data(small: bool = False):
    """Download and prepare training/validation data."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    train_path = CACHE_DIR / "train.json"
    val_path = CACHE_DIR / "val.json"

    if train_path.exists() and val_path.exists():
        print(f"Data already prepared at {CACHE_DIR}")
        return

    print("Downloading Pile validation split for human text...")
    # Use a subset of OpenWebText as human text proxy (Pile is huge)
    human_ds = load_dataset("stas/openwebtext-10k", split="train")
    human_texts = [t for t in human_ds["text"] if len(t) > 200][:5000]
    print(f"  Got {len(human_texts)} human texts")

    # For AI text, we generate synthetic "AI-like" text by taking human text
    # and marking it. In production, the validator uses Ollama with 30+ LLMs.
    # For training, we use the HC3 dataset (human vs ChatGPT answers)
    print("Downloading HC3 dataset (human vs ChatGPT)...")
    try:
        hc3 = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)
        ai_texts = []
        extra_human = []
        for row in hc3:
            for ans in row.get("chatgpt_answers", []):
                if len(ans) > 200:
                    ai_texts.append(ans)
            for ans in row.get("human_answers", []):
                if len(ans) > 200:
                    extra_human.append(ans)
        print(f"  Got {len(ai_texts)} AI texts, {len(extra_human)} extra human texts")
        human_texts.extend(extra_human[:2000])
    except Exception as e:
        print(f"  HC3 download failed: {e}, using synthetic labels")
        ai_texts = []

    # If we don't have enough AI texts, split human texts and label half as "AI"
    # (This is a rough proxy — the real validator uses actual LLM outputs)
    if len(ai_texts) < 1000:
        print("  Generating synthetic AI labels from human text...")
        mid = len(human_texts) // 2
        ai_texts = human_texts[mid:]
        human_texts = human_texts[:mid]

    # Balance and shuffle
    n = min(len(human_texts), len(ai_texts))
    if small:
        n = min(n, 500)

    random.seed(SEED)
    random.shuffle(human_texts)
    random.shuffle(ai_texts)

    samples = []
    for text in human_texts[:n]:
        samples.append({"text": _add_augmentation(text), "label": 0})  # 0 = human
    for text in ai_texts[:n]:
        samples.append({"text": _add_augmentation(text), "label": 1})  # 1 = AI

    random.shuffle(samples)

    # Split 80/20
    split = int(len(samples) * 0.8)
    train_data = samples[:split]
    val_data = samples[split:]

    with open(train_path, "w") as f:
        json.dump(train_data, f)
    with open(val_path, "w") as f:
        json.dump(val_data, f)

    print(f"Prepared {len(train_data)} train, {len(val_data)} val samples")
    print(f"Saved to {CACHE_DIR}")


# ---------------------------------------------------------------------------
# Data loading (imported by train.py)
# ---------------------------------------------------------------------------

def load_train_data() -> list[dict]:
    with open(CACHE_DIR / "train.json") as f:
        return json.load(f)


def load_val_data() -> list[dict]:
    with open(CACHE_DIR / "val.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(labels: list[int], predictions: list[int], probabilities: list[float]) -> dict:
    """
    Evaluate predictions using the SN32 scoring system.

    Args:
        labels: ground truth (0=human, 1=AI)
        predictions: binary predictions (0 or 1)
        probabilities: predicted probability of being AI (0.0 to 1.0)

    Returns:
        dict with f1_score, fp_score, ap_score, combined_score
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    # F1 score
    f1 = f1_score(labels, predictions, zero_division=0)

    # False positive score: 1 - FP / total
    fp = np.sum((predictions == 1) & (labels == 0))
    fp_sc = 1.0 - fp / len(labels)

    # Average precision
    ap = average_precision_score(labels, probabilities)

    combined = (f1 + fp_sc + ap) / 3.0

    return {
        "f1_score": round(f1, 6),
        "fp_score": round(fp_sc, 6),
        "ap_score": round(ap, 6),
        "combined_score": round(combined, 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Use small dataset for testing")
    args = parser.parse_args()

    prepare_data(small=args.small)
    print("\nDone! Ready to train.")
