# autoresearch — SN32 AI Text Detection

This is an experiment to have an LLM autonomously improve an AI text detection model
for Bittensor Subnet 32 (ItsAI).

## Context

We are mining on Bittensor SN32, which rewards miners for accurately classifying text
as human-written or AI-generated. Our miner currently runs a baseline DeBERTa-v3-large
model and earns ~$10/day (rank 121/237). Top miners earn $85-105/day. The scoring is:

- **F1 Score** — binary classification (human vs AI)
- **False Positive Score** — `1 - FP/total` (penalizes calling human text "AI")
- **Average Precision** — quality of probability ranking across all samples
- **Final reward** = average of all three

Validators send a mix of:
- Human text from The Pile dataset (with augmentations: random sentence selection,
  misspellings, adjective removal)
- AI-generated text from 30+ top LLMs (Ollama) with random generation parameters
- Text length varies, augmentations prevent hash-based cheating

Our goal: climb from rank 121 to top 20 by improving model accuracy.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed data prep, evaluation harness, dataset loading. Do not modify.
   - `train.py` — the file you modify. Model architecture, training loop, data augmentation.
4. **Verify data exists**: Check that `data/` contains the training datasets. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU (RTX 3090, 24GB VRAM). The training script runs
for a **fixed time budget of 10 minutes** (wall clock). You launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - Model architecture (DeBERTa variants, ensembles, custom heads)
  - Training hyperparameters (LR, scheduler, batch size, epochs)
  - Data augmentation strategies
  - Loss functions (focal loss, label smoothing, etc.)
  - Preprocessing (tokenization length, text cleaning)
  - Ensemble/averaging strategies
  - Any technique to improve F1, reduce false positives, or improve AP

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation and data loading.
- Install new packages beyond what's in `pyproject.toml`.
- Modify the evaluation harness.

**The goal: maximize the combined score (F1 + FP_score + AP) / 3.** Since this is
what SN32 validators use, this directly translates to higher on-chain incentive.

**VRAM**: You have 24GB on an RTX 3090. Stay within this limit.

**Simplicity criterion**: Same as original — simpler is better, all else equal.

**The first run**: Always establish the baseline first by running train.py as-is.

## Research directions to explore

These are starting points — use your judgment:

1. **Better base models**: Try DeBERTa-v3-base (smaller, might generalize better),
   or microsoft/deberta-v2-xlarge if VRAM allows with gradient checkpointing.
2. **Training data augmentation**: The validator uses misspellings and adjective removal.
   Train with similar augmentations to be robust to them.
3. **Multi-model detection**: AI text from 30+ LLMs. A model that's seen diverse LLM
   outputs during training will generalize better.
4. **Calibration**: AP score rewards well-calibrated probabilities, not just accuracy.
   Temperature scaling, Platt scaling, or mixup training can help.
5. **False positive optimization**: The FP score is 1/3 of the reward. A model that
   rarely calls human text "AI" has an edge. Asymmetric loss or threshold tuning.
6. **Longer context**: More context = more signal. Experiment with max_length.
7. **Ensemble**: Average predictions from multiple checkpoints or model sizes.
8. **Contrastive learning**: Pre-train on human vs AI text pairs before fine-tuning.
9. **Feature engineering**: Perplexity features, burstiness, sentence-level variance.

## Output format

The training script prints a summary like this:

```
---
combined_score:   0.8500
f1_score:         0.8700
fp_score:         0.9200
ap_score:         0.7600
val_loss:         0.3200
training_seconds: 600.0
peak_vram_mb:     18000
num_samples:      50000
```

Extract the key metric:
```
grep "^combined_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

```
commit	combined_score	f1	fp_score	ap	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. combined_score (average of f1, fp_score, ap) — use 0.000 for crashes
3. f1 score
4. fp_score
5. ap score
6. peak memory in GB
7. status: `keep`, `discard`, or `crash`
8. short text description

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar23`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read out the results: `grep "^combined_score:\|^f1_score:\|^fp_score:\|^ap_score:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` for the traceback.
7. Record the results in results.tsv (do NOT commit results.tsv)
8. If combined_score improved (higher is better), keep the git commit
9. If combined_score is equal or worse, git reset back to where you started

**Timeout**: Each experiment should take ~10 minutes. Kill if >15 minutes.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human anything. You are
autonomous. If you run out of ideas, think harder — read the SN32 validator code for
clues, try combining previous near-misses, try more radical approaches. Run until
manually interrupted.

## Key reference files

The SN32 miner code is at: `/home/kuba/Repositories/bittensor/miners/sn32-itsai/`
- `neurons/miner.py` — the miner (shows how forward() processes requests)
- `neurons/miners/deberta_classifier.py` — the current baseline model
- `detection/protocol.py` — the TextSynapse protocol
- `detection/validator/` — how validators score miners (if you need to understand scoring)

Read these for context on exactly what the validator sends and how responses are scored.
