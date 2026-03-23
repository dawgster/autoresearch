"""
SN32 AI Text Detection — Training Script
==========================================

This file is modified by the research agent. It contains the model,
training loop, and inference pipeline.

Current approach: DeBERTa-v3-large fine-tuned for binary classification
(human=0, AI=1).

Metric: combined_score = (F1 + FP_score + AP) / 3  (higher is better)
"""

import os
import time
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import numpy as np

from prepare import (
    load_train_data,
    load_val_data,
    evaluate,
    TIME_BUDGET,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION = 2
THRESHOLD = 0.5  # classification threshold
SEED = 42

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_data = load_train_data()
    val_data = load_val_data()
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Load model and tokenizer
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    # Datasets
    train_dataset = TextDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(val_data, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * 10 // GRADIENT_ACCUMULATION  # rough estimate
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop with fixed time budget
    print(f"Training for {TIME_BUDGET}s...")
    start_time = time.time()
    step = 0
    epoch = 0
    best_score = 0.0
    best_state = None

    model.train()
    while True:
        epoch += 1
        for batch in train_loader:
            if time.time() - start_time > TIME_BUDGET:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRADIENT_ACCUMULATION
            loss.backward()

            step += 1
            if step % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if step % (GRADIENT_ACCUMULATION * 50) == 0:
                    elapsed = time.time() - start_time
                    print(f"  step {step}, loss {loss.item() * GRADIENT_ACCUMULATION:.4f}, {elapsed:.0f}s/{TIME_BUDGET}s")

        if time.time() - start_time > TIME_BUDGET:
            break

        # Quick eval at end of each epoch
        score = _evaluate(model, val_loader, device)
        if score["combined_score"] > best_score:
            best_score = score["combined_score"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch}: NEW BEST combined={score['combined_score']:.4f} "
                  f"f1={score['f1_score']:.4f} fp={score['fp_score']:.4f} ap={score['ap_score']:.4f}")

        if time.time() - start_time > TIME_BUDGET:
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    training_seconds = time.time() - start_time
    final_score = _evaluate(model, val_loader, device)
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Save model
    save_dir = Path("checkpoints/latest")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Print results
    print("\n---")
    print(f"combined_score:   {final_score['combined_score']:.6f}")
    print(f"f1_score:         {final_score['f1_score']:.6f}")
    print(f"fp_score:         {final_score['fp_score']:.6f}")
    print(f"ap_score:         {final_score['ap_score']:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
    print(f"num_samples:      {len(train_data)}")
    print(f"epochs:           {epoch}")
    print(f"steps:            {step}")


def _evaluate(model, val_loader, device) -> dict:
    """Run evaluation on validation set."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= THRESHOLD).astype(int)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    model.train()
    return evaluate(all_labels, all_preds, all_probs)


if __name__ == "__main__":
    train()
