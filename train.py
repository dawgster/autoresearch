"""
SN32 AI Text Detection — Training Script
==========================================

Frozen-encoder approach: extract representations from a large pretrained
encoder, then train only a small classification head. This allows testing
large models (Flan-T5-Large, Pile-T5-Large) on 16GB VRAM with fast iteration.

Metric: combined_score = (F1 + FP_score + AP) / 3  (higher is better)
"""

import os
import time
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModel,
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

MODEL_NAME = "google/flan-t5-large"
MAX_LENGTH = 512
BATCH_SIZE = 32   # large batch OK since we only train a small head
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION = 1
THRESHOLD = 0.5
SEED = 42
EXTRACT_BATCH_SIZE = 16  # batch size for feature extraction (forward-only)

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
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(model, dataloader, device):
    """Extract frozen encoder representations for all samples."""
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # Get hidden states (works for both encoder-only and full models)
                hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

                # Mean pool over non-padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).float()
                pooled = (hidden.float() * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

            all_features.append(pooled.cpu())
            all_labels.append(labels)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels=2, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

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

    # Load tokenizer and frozen encoder
    print(f"Loading {MODEL_NAME} (frozen encoder)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_model = AutoModel.from_pretrained(MODEL_NAME)
    # For T5-style models, extract just the encoder to save VRAM
    if hasattr(full_model, "encoder"):
        encoder = full_model.encoder.to(device)
        hidden_size = full_model.config.d_model
        del full_model
    else:
        encoder = full_model.to(device)
        hidden_size = full_model.config.hidden_size
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    print(f"Hidden size: {hidden_size}")

    # Create datasets for feature extraction
    train_dataset = TextDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(val_data, tokenizer, MAX_LENGTH)
    extract_loader_train = DataLoader(train_dataset, batch_size=EXTRACT_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    extract_loader_val = DataLoader(val_dataset, batch_size=EXTRACT_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Extract features (one-time forward pass)
    print("Extracting train features...")
    t0 = time.time()
    train_features, train_labels = extract_features(encoder, extract_loader_train, device)
    print(f"  Done in {time.time() - t0:.1f}s, shape: {train_features.shape}")

    print("Extracting val features...")
    t0 = time.time()
    val_features, val_labels = extract_features(encoder, extract_loader_val, device)
    print(f"  Done in {time.time() - t0:.1f}s, shape: {val_features.shape}")

    # Free encoder from GPU
    del encoder
    torch.cuda.empty_cache()
    print(f"Encoder freed, VRAM after: {torch.cuda.memory_allocated() / 1024 / 1024:.0f} MB")

    # Create dataloaders for head training
    train_tensor_ds = TensorDataset(train_features, train_labels)
    val_tensor_ds = TensorDataset(val_features, val_labels)
    train_loader = DataLoader(train_tensor_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_tensor_ds, batch_size=BATCH_SIZE * 4, shuffle=False)

    # Classification head
    head = ClassificationHead(hidden_size, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * 100 // GRADIENT_ACCUMULATION  # many epochs possible since head is tiny
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"Training head for {TIME_BUDGET}s...")
    start_time = time.time()
    step = 0
    epoch = 0
    best_score = 0.0
    best_state = None

    head.train()
    while True:
        epoch += 1
        for features, labels in train_loader:
            if time.time() - start_time > TIME_BUDGET:
                break

            features = features.to(device)
            labels = labels.to(device)

            logits = head(features)
            loss = criterion(logits, labels) / GRADIENT_ACCUMULATION
            loss.backward()

            step += 1
            if step % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if step % (GRADIENT_ACCUMULATION * 200) == 0:
                    elapsed = time.time() - start_time
                    print(f"  step {step}, loss {loss.item() * GRADIENT_ACCUMULATION:.4f}, {elapsed:.0f}s/{TIME_BUDGET}s")

        if time.time() - start_time > TIME_BUDGET:
            break

        # Eval at end of each epoch
        score = _evaluate_head(head, val_loader, device)
        if score["combined_score"] > best_score:
            best_score = score["combined_score"]
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            print(f"  Epoch {epoch}: NEW BEST combined={score['combined_score']:.4f} "
                  f"f1={score['f1_score']:.4f} fp={score['fp_score']:.4f} ap={score['ap_score']:.4f}")

        if time.time() - start_time > TIME_BUDGET:
            break

    # Restore best
    if best_state:
        head.load_state_dict(best_state)

    # Final eval
    training_seconds = time.time() - start_time
    final_score = _evaluate_head(head, val_loader, device)
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Save head + tokenizer info
    save_dir = Path("checkpoints/latest")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state_dict": head.state_dict(),
        "model_name": MODEL_NAME,
        "hidden_size": hidden_size,
        "max_length": MAX_LENGTH,
    }, save_dir / "head.pt")
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


def _evaluate_head(head, val_loader, device) -> dict:
    """Evaluate the classification head on precomputed features."""
    head.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            logits = head(features).float()
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= THRESHOLD).astype(int)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    head.train()
    return evaluate(all_labels, all_preds, all_probs)


if __name__ == "__main__":
    train()
