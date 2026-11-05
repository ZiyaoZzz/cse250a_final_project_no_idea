# Blackjack Q-Table Project

CSE 250A final project. We train a tabular Q-learning agent, start from a small
observation, make 2–3 intuitive changes, and show how performance moves after each one.

## Results so far

6-deck shoe, 5 seeds, 30M training hands and 10M evaluation hands per seed. Scored on
the true reward. Full table: [outputs/q_table_results.md](outputs/q_table_results.md).

| Policy | Mean return / hand | Change vs baseline |
|---|---:|---:|
| Random play | -0.3953 | -0.3495 |
| **Baseline Q-table** (total, upcard, usable ace) | -0.0458 ± 0.0005 | — |
| + Hi-Lo true count | -0.0455 ± 0.0004 | +0.0003 ± 0.0002 |
| + double down | -0.0332 ± 0.0005 | +0.0126 ± 0.0004 |
| + count + double down | -0.0329 ± 0.0004 | +0.0129 ± 0.0002 |
| + double, bust penalty 0.1 | -0.0361 ± 0.0009 | +0.0097 ± 0.0008 |
| Basic strategy + double | -0.0329 ± 0.0004 | +0.0129 ± 0.0003 |

- **Observation:** the count adds +0.0003 on 6 decks, +0.0011 on 2 decks, and +0.0024 on
  1 deck. The number of cards adds nothing.
- **Action:** double down is the largest gain and removes about 28% of the house edge.
- **Reward:** a bust penalty cuts busts from 17.7% to 11.7% but lowers return, and larger
  penalties do worse.

![Learning curves](outputs/learning_curves.png)

## Team TODO

| Who | Owns | To do |
|---|---|---|
| Ziyao | Environment and baseline | Merge everyone's work, keep experiment settings consistent, write the methods section |
| Member A | Observation | Find which states change their decision with the count (for example 16 vs 10); compare count buckets of ±2 and ±3; write the observation bullet |
| Member B | Action | Plot where the Q-table doubles against basic strategy and explain the differences; write the action bullet |
| Member C | Reward | Explain why the bust penalty hurts; try potential-based shaping and check whether it learns faster; write the reward bullet |
| Member D | Reproduction and report | Rerun everything from a clean clone with new seeds; confirm the conclusions hold; finalize the table and figures; assemble the report |

## Setup notes

`Blackjack-v1` deals from an infinite deck and has no double down, so `blackjack_shoe.py`
keeps its rules and adds a 6-deck shoe and an optional double. All stages run there. With
`num_decks=None` it matches Blackjack-v1 and the exact policy-iteration value.

Compare stages with the same settings: fixed evaluation seeds, 5 training seeds, and
millions of evaluation hands. At 50,000 hands the 95% interval is about ±0.009, which
hides the count effect.

## Running it

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python q_table.py                 # Full ladder: about 25 minutes on 14 CPU workers
python metrics_and_plots.py       # Writes the results table and learning curves
```

Options: `--decks 2`, `--stages baseline count`, and `--hands 1000000` for a quick run.

| File | Purpose |
|---|---|
| `blackjack_shoe.py` | Shoe environment, Hi-Lo count, double down |
| `q_table.py` | Stage definitions, Q-learning, experiment runner |
| `evaluate.py` | Shared evaluation and reference policies |
| `metrics_and_plots.py` | Results table and learning curves |
| `traditional_baselines.py` | Exact model and policy iteration, used for basic strategy |
