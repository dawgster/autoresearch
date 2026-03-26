"""
SN120 Affine — RL Training Script
===================================

This file is modified by the research agent. It contains the model
selection, RL training loop, and evaluation pipeline.

Current approach: QLoRA fine-tuning of Qwen2.5-3B-Instruct with
GRPO on self-play rollouts from AgentGym environments.

Metric: avg_score across all environments (higher is better)
"""

import os
import sys
import time
import json
import random
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from prepare import (
    evaluate_model,
    pull_frontier_model,
    ENVIRONMENTS,
    TIME_BUDGET,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
MAX_LENGTH = 2048
SEED = 42

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load base model with QLoRA
    print(f"Loading {BASE_MODEL} with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Step 1: Evaluate baseline
    print("\n--- Evaluating baseline ---")
    baseline_scores = evaluate_model(model, tokenizer, device)
    print(f"Baseline scores: {baseline_scores}")
    baseline_avg = sum(baseline_scores.values()) / len(baseline_scores)
    print(f"Baseline avg: {baseline_avg:.4f}")

    # Step 2: Collect rollouts from environments
    # TODO: The research agent should implement rollout collection
    # and RL training here. This is the core loop to iterate on.
    print("\n--- Collecting rollouts ---")
    print("TODO: Implement rollout collection and RL training")
    print("The research agent should modify this section.")

    # Step 3: RL training on collected rollouts
    # Possible approaches:
    # - GRPO: Group Relative Policy Optimization
    # - DPO: Direct Preference Optimization on (good, bad) rollout pairs
    # - SFT on successful rollouts only
    # - PPO with task completion reward

    start_time = time.time()
    training_seconds = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Final evaluation
    final_scores = evaluate_model(model, tokenizer, device)
    avg_score = sum(final_scores.values()) / len(final_scores)

    # Save model
    save_dir = Path("checkpoints/latest")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Print results
    print("\n---")
    for env, score in final_scores.items():
        print(f"env_{env}:{'':>{14-len(env)}}{score:.6f}")
    print(f"avg_score:        {avg_score:.6f}")
    print(f"training_minutes: {training_seconds / 60:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
    print(f"base_model:       {BASE_MODEL}")
    print(f"method:           baseline (no RL yet)")


if __name__ == "__main__":
    train()
