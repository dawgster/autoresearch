"""
Data preparation and evaluation harness for SN32 AI text detection.
Downloads/generates training data and provides fixed evaluation.

Datasets:
  - HC3 (human vs ChatGPT)
  - OpenWebText-10k (human text)
  - RAID (11 LLMs, adversarial attacks, multiple domains)
  - ai-text-detection-pile (Pile-based human + AI text)

Augmentations (matching SN32 validators):
  - Random sentence selection
  - Misspellings (char swap)
  - Adjective dropout

Usage:
    python prepare.py              # full prep
    python prepare.py --small      # small dataset for quick testing

DO NOT MODIFY the evaluate() function — it is the fixed evaluation harness.
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
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / "data"
TIME_BUDGET = 600  # 10 minutes training budget
EVAL_SAMPLES = 2000  # number of samples for validation
SEED = 42

# Common English adjectives for dropout augmentation
_ADJECTIVES = {
    "good", "great", "big", "small", "large", "old", "new", "young", "long",
    "short", "high", "low", "early", "late", "important", "different", "same",
    "able", "bad", "best", "better", "certain", "clear", "close", "common",
    "current", "dark", "deep", "easy", "entire", "even", "fair", "far",
    "final", "fine", "free", "full", "general", "happy", "hard", "heavy",
    "hot", "huge", "human", "key", "kind", "known", "last", "left", "likely",
    "little", "local", "main", "major", "many", "modern", "much", "natural",
    "nice", "obvious", "open", "other", "own", "particular", "past", "perfect",
    "physical", "political", "poor", "popular", "possible", "present",
    "private", "professional", "proper", "public", "quick", "quiet", "ready",
    "real", "recent", "red", "related", "rich", "right", "serious", "short",
    "significant", "similar", "simple", "single", "slight", "slow", "social",
    "soft", "solid", "special", "specific", "strong", "successful", "sudden",
    "sufficient", "sure", "top", "total", "traditional", "true", "typical",
    "unique", "various", "warm", "white", "whole", "wide", "wonderful",
    "wrong", "beautiful", "black", "blue", "green", "cold", "complete",
    "complex", "critical", "dangerous", "dead", "difficult", "direct",
    "dry", "effective", "empty", "essential", "exact", "excellent",
    "expensive", "famous", "fast", "favorite", "flat", "foreign", "formal",
    "former", "fresh", "funny", "glad", "global", "golden", "grand", "gray",
    "healthy", "helpful", "huge", "illegal", "immediate", "impossible",
    "independent", "individual", "inevitable", "intelligent", "interesting",
    "internal", "legal", "loud", "lovely", "lucky", "massive", "medical",
    "mental", "military", "minor", "narrow", "neat", "necessary", "negative",
    "nervous", "normal", "novel", "official", "ordinary", "original",
    "outer", "pleasant", "positive", "powerful", "previous", "primary",
    "proud", "pure", "rare", "raw", "reasonable", "regular", "relevant",
    "remarkable", "responsible", "rough", "round", "royal", "sad", "safe",
    "secret", "separate", "severe", "sharp", "sick", "silent", "silly",
    "smooth", "spare", "stable", "standard", "strange", "strict", "stupid",
    "suitable", "sweet", "tall", "terrible", "thick", "thin", "tiny",
    "tough", "ugly", "unlikely", "unusual", "upper", "useful", "usual",
    "valuable", "vast", "visible", "visual", "vital", "weak", "weird",
    "wild", "wise", "wooden", "worth",
}

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

    # Adjective dropout (remove random adjectives)
    if random.random() < 0.3:
        words = text.split()
        words = [w for w in words if w.lower().strip(",.!?;:") not in _ADJECTIVES or random.random() > 0.5]
        text = " ".join(words)

    return text


def prepare_data(small: bool = False):
    """Download and prepare training/validation data from multiple sources."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    train_path = CACHE_DIR / "train.json"
    val_path = CACHE_DIR / "val.json"

    if train_path.exists() and val_path.exists():
        print(f"Data already prepared at {CACHE_DIR}")
        return

    human_texts = []
    ai_texts = []

    # --- Source 1: OpenWebText (human text) ---
    print("Downloading OpenWebText-10k for human text...")
    human_ds = load_dataset("stas/openwebtext-10k", split="train")
    owt_human = [t for t in human_ds["text"] if len(t) > 200][:5000]
    human_texts.extend(owt_human)
    print(f"  Got {len(owt_human)} human texts from OpenWebText")

    # --- Source 2: HC3 (human vs ChatGPT) ---
    print("Downloading HC3 dataset (human vs ChatGPT)...")
    try:
        hc3 = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)
        hc3_ai = []
        hc3_human = []
        for row in hc3:
            for ans in row.get("chatgpt_answers", []):
                if len(ans) > 200:
                    hc3_ai.append(ans)
            for ans in row.get("human_answers", []):
                if len(ans) > 200:
                    hc3_human.append(ans)
        ai_texts.extend(hc3_ai)
        human_texts.extend(hc3_human[:2000])
        print(f"  Got {len(hc3_ai)} AI texts, {len(hc3_human)} human texts from HC3")
    except Exception as e:
        print(f"  HC3 download failed: {e}")

    # --- Source 3: RAID (multi-model, multi-domain, adversarial) ---
    print("Downloading RAID dataset...")
    try:
        raid = load_dataset("liamdugan/raid", split="train", streaming=True)
        raid_human = []
        raid_ai = []
        for i, row in enumerate(raid):
            text = row.get("generation", "")
            if len(text) < 200:
                continue
            if row["model"] == "human":
                raid_human.append(text)
            else:
                raid_ai.append(text)
            # Cap at 20k samples to avoid very long download
            if len(raid_human) + len(raid_ai) >= 20000:
                break
        human_texts.extend(raid_human)
        ai_texts.extend(raid_ai)
        print(f"  Got {len(raid_ai)} AI texts, {len(raid_human)} human texts from RAID")
    except Exception as e:
        print(f"  RAID download failed: {e}")

    # --- Source 4: ai-text-detection-pile (Pile-based, matches SN32 distribution) ---
    print("Downloading ai-text-detection-pile...")
    try:
        pile_det = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
        pile_human = []
        pile_ai = []
        for i, row in enumerate(pile_det):
            text = row.get("text", "")
            if len(text) < 200:
                continue
            if row["source"] == "human":
                pile_human.append(text)
            else:
                pile_ai.append(text)
            # Cap to keep dataset manageable
            if len(pile_human) + len(pile_ai) >= 20000:
                break
        human_texts.extend(pile_human)
        ai_texts.extend(pile_ai)
        print(f"  Got {len(pile_ai)} AI texts, {len(pile_human)} human texts from Pile-detect")
    except Exception as e:
        print(f"  ai-text-detection-pile download failed: {e}")

    print(f"\nTotal collected: {len(human_texts)} human, {len(ai_texts)} AI texts")

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
