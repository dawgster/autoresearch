# autoresearch — SN120 Affine (Reinforcement Learning)

This is an experiment to have an LLM autonomously improve a reasoning model
for Bittensor Subnet 120 (Affine). The potential reward is ~$5,000/GPU/day
for a competitive model.

## Context

Affine is an incentivized RL competition on Bittensor. Miners submit models that
are evaluated on multi-turn agent tasks. The mechanism is winners-take-all on the
Pareto frontier — your model must outperform all others across ALL environments
to earn maximum rewards.

**How it works:**
1. Pull an existing frontier model from the network
2. Improve it with RL training (RLHF, DPO, GRPO, etc.)
3. Upload to HuggingFace
4. Deploy as a Chutes inference endpoint
5. Commit on-chain
6. Validators evaluate your model on tasks → emissions if you're on the Pareto frontier

**Current environments (from AgentGym suite):**
- `webshop` — web shopping agent tasks
- `alfworld` — embodied household tasks
- `babyai` — grid-world navigation/instruction following
- `sciworld` — science experiment simulation
- More are added over time

**Scoring:** Models are evaluated on task completion across all environments.
A model must dominate the Pareto frontier — meaning it's better than all existing
models on at least one environment while not worse on others. Statistical significance
via Beta distribution confidence intervals prevents copy-mining.

**Revenue:** 6 outsiders earn $1K-$35K/day. Median: $4,390/day per GPU.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar26`).
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — environment setup, evaluation harness, model pulling. Do not modify.
   - `train.py` — the file you modify. RL training pipeline.
4. **Pull a frontier model**: Use the Affine CLI to pull the current best model as starting point.
5. **Initialize results.tsv**: Create with header row.
6. **Confirm and go**.

## Experimentation

Each experiment modifies the RL training approach in `train.py`. The workflow differs
from traditional autoresearch because we're doing RL, not supervised learning:

**What you CAN do:**
- Modify `train.py` — everything is fair game:
  - RL algorithm (PPO, DPO, GRPO, REINFORCE, etc.)
  - Reward shaping and reward models
  - Training data generation (rollout collection)
  - Base model selection (Qwen, Llama, Mistral, DeepSeek, etc.)
  - Hyperparameters (LR, KL penalty, batch size, epochs)
  - Multi-task training strategies
  - Curriculum learning across environments
  - Prompt engineering for agent behavior
  - Tool-use and chain-of-thought optimization
  - Model merging / DARE / TIES techniques
  - LoRA / QLoRA for efficient fine-tuning on limited VRAM

**What you CANNOT do:**
- Modify `prepare.py`
- Install new packages beyond pyproject.toml
- Modify the evaluation harness

**The goal: maximize task completion rate across ALL environments simultaneously.**
A model that's great at webshop but terrible at alfworld won't earn — it needs to be
on the Pareto frontier across all environments.

**VRAM**: You have 24GB on an RTX 3090. Use QLoRA/LoRA for 7B+ models.
Consider smaller models (1.5B-3B) that fit fully for faster iteration.

**Time budget**: Each training run should target ~30 minutes. RL needs more
iterations than supervised learning. Collect rollouts, train, evaluate, repeat.

**The first run**: Pull the current frontier model, evaluate it on available
environments to establish baseline scores.

## Research directions

1. **Start with a strong base**: Qwen2.5-3B-Instruct or similar — small enough
   for full fine-tuning on 24GB, strong enough for agent tasks.
2. **GRPO/DPO on rollout data**: Collect rollouts from the environments, rank by
   reward, train with preference optimization.
3. **ReAct/tool-use prompting**: Agent tasks benefit from structured reasoning.
   Fine-tune the model to use ReAct-style thought-action-observation loops.
4. **Environment-specific strategies**: Each AgentGym environment has different
   optimal strategies. Train on diverse rollouts across all environments.
5. **Model merging**: Train separate LoRA adapters per environment, then merge
   with DARE/TIES to get a single model good at everything.
6. **Distillation from larger models**: Use GPT-4/Claude to generate expert
   trajectories, then distill into your smaller model.
7. **Self-play / iterative refinement**: Generate rollouts with current model,
   filter for successes, train on those, repeat.
8. **Reward hacking prevention**: The eval uses confidence intervals, so your
   model needs genuinely better task completion, not statistical flukes.

## Output format

```
---
env_webshop:      0.7500
env_alfworld:     0.6800
env_babyai:       0.9200
env_sciworld:     0.5400
avg_score:        0.7225
training_minutes: 30.0
peak_vram_mb:     22000
base_model:       Qwen/Qwen2.5-3B-Instruct
method:           GRPO on self-play rollouts
```

Extract: `grep "^avg_score:\|^env_" run.log`

## Logging results

TSV with header:

```
commit	avg_score	webshop	alfworld	babyai	sciworld	memory_gb	status	description
```

## The experiment loop

LOOP FOREVER:

1. Check current branch/commit state
2. Modify `train.py` with an RL improvement idea
3. git commit
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^avg_score:\|^env_" run.log`
6. If crashed, check `tail -n 50 run.log`, attempt fix
7. Log to results.tsv
8. If avg_score improved AND no environment regressed significantly, keep
9. Otherwise git reset and try another approach

**NEVER STOP.** Run until manually interrupted.

## Key reference files

The Affine miner code is at: `/home/kuba/Repositories/bittensor/miners/sn120-affine/`
- `docs/MINER.md` — complete miner workflow
- `docs/FAQ.md` — common issues and solutions
- `affine/cli.py` — CLI commands (pull, push, commit)
- `examples/sdk.py` — how to evaluate models on environments
- `examples/sdk2.py` — evaluate custom models

The Affine SDK can be used to:
- Pull frontier models: `af pull <UID> --model-path ./model`
- Evaluate models: see `examples/sdk.py`
- List environments: see SDK docs

## Deployment workflow (after training)

Once you have a model that improves on the frontier:

1. Upload to HuggingFace: `huggingface-cli upload <user>/affine-model ./model`
2. Deploy to Chutes: `af chutes_push --repo <user>/affine-model --revision <SHA>`
3. Commit on-chain: `af commit --repo <user>/affine-model --revision <SHA> --chute-id <id>`

The human will handle deployment. Your job is to produce the best model.
