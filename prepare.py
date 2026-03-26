"""
SN120 Affine — Environment setup and evaluation harness.
Provides tools to pull frontier models and evaluate on AgentGym environments.

DO NOT MODIFY the evaluate_model() function — it is the fixed evaluation harness.
"""

import os
import json
import time
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_BUDGET = 1800  # 30 minutes training budget
SEED = 42

# AgentGym environments used by Affine validators
ENVIRONMENTS = [
    "webshop",
    "alfworld",
    "babyai",
    "sciworld",
]

AFFINE_REPO = Path(__file__).parent.parent / "bittensor" / "miners" / "sn120-affine"

# ---------------------------------------------------------------------------
# Model pulling
# ---------------------------------------------------------------------------

def pull_frontier_model(uid: int = None, model_path: str = "./frontier_model"):
    """Pull the current frontier model from the Affine network."""
    model_path = Path(model_path)
    if model_path.exists() and any(model_path.iterdir()):
        print(f"Frontier model already exists at {model_path}")
        return model_path

    model_path.mkdir(parents=True, exist_ok=True)
    if uid is None:
        uid = 0

    print(f"Pulling model from UID {uid} to {model_path}...")
    try:
        result = subprocess.run(
            ["af", "pull", str(uid), "--model-path", str(model_path)],
            capture_output=True, text=True, timeout=300
        )
        print(result.stdout[-200:] if result.stdout else "No output")
    except Exception as e:
        print(f"Pull error: {e}")

    return model_path


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed evaluation harness)
# ---------------------------------------------------------------------------

def evaluate_model(model, tokenizer, device, environments=None, max_tasks=10) -> dict[str, float]:
    """
    Evaluate a model on AgentGym-style environments (local proxy).

    The real Affine evaluation happens on validators via Chutes.
    This provides a fast local approximation for iterative development.
    """
    import torch

    if environments is None:
        environments = ENVIRONMENTS

    scores = {}
    for env_name in environments:
        print(f"  Evaluating {env_name}...")
        env_score = _evaluate_env(model, tokenizer, device, env_name, max_tasks)
        scores[env_name] = env_score
        print(f"    {env_name}: {env_score:.4f}")

    return scores


def _evaluate_env(model, tokenizer, device, env_name: str, max_tasks: int) -> float:
    """Evaluate model on a single environment using local proxy tasks."""
    import torch

    tasks = _get_proxy_tasks(env_name, max_tasks)
    if not tasks:
        return 0.0

    correct = 0
    total = len(tasks)

    model.eval()
    with torch.no_grad():
        for task in tasks:
            prompt = task["prompt"]
            expected = task["expected"]

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=1024).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs, max_new_tokens=256, temperature=0.1,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip().lower()

            if any(exp.lower() in response for exp in expected):
                correct += 1

    return correct / total if total > 0 else 0.0


def _get_proxy_tasks(env_name: str, max_tasks: int) -> list[dict]:
    """Generate local proxy tasks approximating AgentGym environments."""

    if env_name == "webshop":
        tasks = [
            {"prompt": "You are a shopping assistant. Find the cheapest red shoes under $50. What action do you take first?\nActions: [search, click, buy]\nAnswer:", "expected": ["search"]},
            {"prompt": "Search results: 1) Red Nike $45 2) Blue Adidas $30 3) Red Puma $55. Which matches 'red shoes under $50'?\nAnswer:", "expected": ["1", "nike", "red nike"]},
            {"prompt": "You need to buy a cotton t-shirt in size M. The page shows sizes: S, M, L, XL. Which do you click?\nAnswer:", "expected": ["m", "medium"]},
            {"prompt": "Product: 'Blue Cotton T-Shirt, $25, Rating: 4.5/5'. You need a blue cotton shirt under $30. Buy?\nAnswer:", "expected": ["yes", "buy"]},
            {"prompt": "Search results: 1) Wireless Mouse $15 2) Wired Mouse $8 3) Gaming Mouse $45. Find cheapest.\nAnswer:", "expected": ["2", "wired", "$8"]},
        ]
    elif env_name == "alfworld":
        tasks = [
            {"prompt": "You are in a kitchen. You need to heat an apple. The apple is on the counter. Actions: [go to microwave, go to fridge, pick up apple]. What first?\nAnswer:", "expected": ["pick up", "pick"]},
            {"prompt": "You hold a dirty mug. Need to clean it. Actions: [go to sink, go to table, put mug]. What?\nAnswer:", "expected": ["go to sink", "sink"]},
            {"prompt": "Find a book. Checked: desk (no), shelf (no). Remaining: drawer, cabinet. Where?\nAnswer:", "expected": ["drawer", "cabinet"]},
            {"prompt": "At sink with dirty plate. Actions: [clean plate, turn on faucet, put down]. What first?\nAnswer:", "expected": ["turn on faucet", "faucet", "clean"]},
            {"prompt": "Task: put cool egg in fridge. You have a hot egg. Actions: [go to fridge, go to sink, wait]. What?\nAnswer:", "expected": ["wait", "sink", "cool"]},
        ]
    elif env_name == "babyai":
        tasks = [
            {"prompt": "Grid world. You at (2,3). Goal: green square at (5,3). Actions: [left, right, up, down]. What?\nAnswer:", "expected": ["right"]},
            {"prompt": "You see: red ball (1,1), blue key (3,2), green door (3,3). Open green door. Need key. Pick up?\nAnswer:", "expected": ["blue key", "key"]},
            {"prompt": "Face north. Wall ahead. Goal to right. Actions: [turn left, turn right, forward]. What?\nAnswer:", "expected": ["turn right", "right"]},
            {"prompt": "Pick up purple box (ahead, 2 steps). Actions: [forward, left, right, pick up]. What?\nAnswer:", "expected": ["forward"]},
            {"prompt": "Holding red key. Red door ahead. Actions: [use key, forward, drop key]. What?\nAnswer:", "expected": ["use key", "use", "toggle"]},
        ]
    elif env_name == "sciworld":
        tasks = [
            {"prompt": "Test if salt dissolves in water. Have: beaker, water, salt, thermometer. First step?\nAnswer:", "expected": ["pour water", "add water", "water"]},
            {"prompt": "Measure temp of boiling water. Have thermometer. Water in pot, stove is on. What?\nAnswer:", "expected": ["put thermometer", "thermometer"]},
            {"prompt": "Grow a plant. Have: seed, soil, pot, water, lamp. First step?\nAnswer:", "expected": ["put soil", "soil in pot", "add soil"]},
            {"prompt": "Separate sand and water. Tools: filter paper, beaker, funnel, magnifying glass. How?\nAnswer:", "expected": ["filter", "funnel"]},
            {"prompt": "Test: metals conduct electricity. Have: battery, wire, bulb, iron nail, wood stick. How?\nAnswer:", "expected": ["connect", "iron", "nail", "circuit"]},
        ]
    else:
        return []

    return tasks[:max_tasks]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull", type=int, default=None, help="Pull frontier model from UID")
    parser.add_argument("--model-path", default="./frontier_model", help="Model path")
    args = parser.parse_args()

    if args.pull is not None:
        pull_frontier_model(uid=args.pull, model_path=args.model_path)

    print("\nDone! Ready to train.")
