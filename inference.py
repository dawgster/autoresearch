"""
SN32 AI Text Detection — Inference Module
==========================================

Frozen Pile-T5-Large encoder + trained classification head.
Drop-in replacement for the SN32 miner's detection pipeline.

Usage:
    from inference import Detector

    detector = Detector("checkpoints/latest")
    probs = detector.predict(["some text", "another text"])
    # probs = [0.02, 0.95]  (probability of being AI-generated)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels=2, dropout=0.0):
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


class Detector:
    """AI text detector using frozen Pile-T5-Large encoder + trained head."""

    def __init__(self, checkpoint_dir: str = "checkpoints/latest", device: str = None):
        checkpoint_dir = Path(checkpoint_dir)

        # Load head checkpoint metadata
        head_ckpt = torch.load(checkpoint_dir / "head.pt", map_location="cpu", weights_only=True)
        self.model_name = head_ckpt["model_name"]
        self.hidden_size = head_ckpt["hidden_size"]
        self.max_length = head_ckpt["max_length"]

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Encoder (frozen, eval mode)
        full_model = AutoModel.from_pretrained(self.model_name)
        if hasattr(full_model, "encoder"):
            self.encoder = full_model.encoder.to(self.device)
            del full_model
        else:
            self.encoder = full_model.to(self.device)
        self.encoder.eval()

        # Classification head
        self.head = ClassificationHead(self.hidden_size, num_labels=2)
        self.head.load_state_dict(head_ckpt["head_state_dict"])
        self.head = self.head.to(self.device)
        self.head.eval()

        torch.cuda.empty_cache()
        print(f"Detector loaded: {self.model_name} (hidden={self.hidden_size}, max_len={self.max_length})")

    @torch.no_grad()
    def predict(self, texts: list[str], batch_size: int = 16) -> list[float]:
        """
        Predict probability of text being AI-generated.

        Args:
            texts: list of text strings
            batch_size: inference batch size

        Returns:
            list of floats (0.0 = human, 1.0 = AI)
        """
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                mask_expanded = attention_mask.unsqueeze(-1).float()
                pooled = (hidden.float() * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

            logits = self.head(pooled).float()
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

        return all_probs

    def predict_single(self, text: str) -> float:
        """Predict probability of a single text being AI-generated."""
        return self.predict([text])[0]


if __name__ == "__main__":
    # Quick self-test
    detector = Detector()

    test_texts = [
        "The quick brown fox jumps over the lazy dog. This is a simple sentence written by a human.",
        "In conclusion, the implementation of artificial intelligence in healthcare systems represents a paradigm shift in medical diagnostics, offering unprecedented accuracy in disease detection while simultaneously reducing operational costs and improving patient outcomes across diverse clinical settings.",
    ]

    probs = detector.predict(test_texts)
    for text, prob in zip(test_texts, probs):
        label = "AI" if prob >= 0.5 else "Human"
        print(f"[{label} {prob:.4f}] {text[:80]}...")
