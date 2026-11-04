# Blackjack LLM Project

## Phase 1: Environment and data audit

Install the Phase 1 dependency:

```powershell
python -m pip install -r requirements.txt
```

Create the configured Gymnasium environment (`Blackjack-v1`, `natural=False`,
`sab=False`):

```python
from env_setup import make_blackjack_env

with make_blackjack_env(seed=42) as env:
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(1)  # hit
```

Audit one or more local candidate CSV datasets:

```powershell
python data_auditor.py .\data\candidate_900k.csv .\data\candidate_50m.csv
```

The auditor reads only the header and up to 1,000 records by default. Pass
`--row-limit 0` to scan each file fully. Its `transition_candidate` status means
the expected columns were found, not that their values or semantics have been
fully verified. `outcome_only` datasets are explicitly unsuitable as transition
data. Inspect `hidden_card_columns` and exclude those fields from decision-time
observations.

Run Phase 1 checks:

```powershell
python -m unittest discover -s tests
```

## Phase 2: Traditional baselines and expert data

### `traditional_baselines.py`

Build the Blackjack transition and reward models, averaging over the hidden
dealer card. Add policy iteration as the known-model reference and tabular
Q-learning with epsilon-greedy exploration and a configurable learning rate.

Run policy iteration, train Q-learning, and compare both policies in the
configured environment:

```powershell
python traditional_baselines.py --q-episodes 100000 --eval-episodes 10000
```

The results are saved to `outputs/phase2_baselines.json`.

### `sft_data_generator.py`

Generate state/action examples labeled by policy iteration. Include only the
player total, dealer up-card, and usable-ace status—never hidden cards or future
outcomes.

The generator samples hands from `Blackjack-v1` and writes one JSONL record per
decision:

```powershell
python sft_data_generator.py --episodes 100000
```

The default output is `outputs/expert_blackjack.jsonl`.

## Phase 3: LLM training pipelines

### `llm_prompts.py`

Use one fixed prompt containing only the current game state. Accept only `Hit`
or `Stick` as a model response.

### `train_sft.py`

Fine-tune a small open-source language model with LoRA on the generated
examples. Save the adapter and training settings.

### `train_rl.py`

Use REINFORCE on new Blackjack games to improve expected reward. Include a
state-independent running-return baseline and save training results.

## Phase 4: Evaluation, ablation, and visualization

### `evaluate.py`

Compare each frozen policy across three training seeds and 100,000 fresh games
per seed. Include policy iteration, Q-learning, SFT/RL models, the “hit below
17” rule, and the base LLM.

### `metrics_and_plots.py`

Report average return with 95% confidence intervals, win/draw/loss rates,
return gap from policy iteration, inference latency, and training interactions.
Plot policy-disagreement heatmaps against policy iteration.

## Implementation status

Phase 1 and Phase 2 are implemented. Phase 3–4 modules are still placeholders:
`llm_prompts.py`, `train_sft.py`, `train_rl.py`, `evaluate.py`, and
`metrics_and_plots.py`.
