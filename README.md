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
- **Count + double on 1 deck:** +0.0177 ± 0.0001, beating basic strategy + double
  (+0.0147 ± 0.0001). With few decks the Q-table learns to use the count and outperforms
  the fixed strategy. Full table: [outputs/q_table_results_1deck.md](outputs/q_table_results_1deck.md).
- **Action:** double down is the largest gain and removes about 28% of the house edge.
- **Reward:** a bust penalty cuts busts from 17.7% to 11.7% but lowers return, and larger
  penalties do worse.

![Learning curves](outputs/learning_curves.png)

### Potential-based shaping

Shaping F = Φ(s′) − Φ(s) with Φ(terminal) = 0 telescopes over a hand, so it cannot change
the optimal policy (Ng et al., 1999). Full table:
[outputs/q_table_results_shaping.md](outputs/q_table_results_shaping.md); curves:
[outputs/learning_curves_shaping.png](outputs/learning_curves_shaping.png).

| Reward used in training (all with double) | Change vs baseline |
|---|---:|
| True reward | +0.0126 ± 0.0004 |
| Bust penalty 0.1 | +0.0097 ± 0.0008 |
| Shaping, Φ = total / 21 | +0.0127 ± 0.0002 |
| Shaping, Φ = basic-strategy value | +0.0128 ± 0.0003 |

- Both potentials end at the unshaped result, unlike the bust penalty. This confirms the
  invariance.
- Neither learns noticeably faster: the curves overlap at every budget from 10k to 30M
  hands. A hand lasts only 1–3 decisions, so there is little delayed credit to speed up.
  In tabular Q-learning, shaping is also equivalent to starting Q at Φ (Wiewiora, 2003),
  and the step size visits^-0.85 forgets the starting value quickly.

### Policy analysis (`python policy_analysis.py`, about 15 minutes)

- **Learned vs basic strategy (6 decks, + double):** 4 of 260 first-decision cells differ
  (hard 12 vs 4; soft 13 vs 6, 14 vs 5 and 15 vs 5, where it hits instead of doubling).
  These are near-ties in value. See [outputs/policy_vs_basic.png](outputs/policy_vs_basic.png).
- **Count deviations (1 deck, count + double):** 54 cells change action with the count.
  The hard-total ones match known count plays: stand on 16 vs 10, 15 vs 10 and 12 vs 2–6
  at +2; double 11 vs 10 and 9 vs 2 at +2; hit 13–15 vs 2–6 at −2. Soft-hand deviations
  are noisier because those states are rare. See [outputs/count_deviations.png](outputs/count_deviations.png).
- **Sensitivity (6 decks, + double, 3M hands):** the step-size exponent matters more than
  epsilon. An exponent of 0.6 is about 0.003 worse; 0.85 and 1.0 are within 0.001 of each
  other. See [outputs/sensitivity.png](outputs/sensitivity.png).

## Team TODO

| Who | Owns | To do |
|---|---|---|
| Ziyao | Environment and baseline | Merge everyone's work, keep experiment settings consistent, write the methods section |
| Member A | Observation | Find which states change their decision with the count (for example 16 vs 10); compare count buckets of ±2 and ±3; write the observation bullet |
| Member B | Action | Plot where the Q-table doubles against basic strategy and explain the differences; write the action bullet |
| Member C | Reward | Explain why the bust penalty hurts and why shaping does not (results above); write the reward bullet |
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
python q_table.py --stages baseline double bust_0.1 shaped_heuristic shaped_value --output outputs/q_table_shaping.json
python metrics_and_plots.py --series shaping --input outputs/q_table_shaping.json --table outputs/q_table_results_shaping.md --figure outputs/learning_curves_shaping.png
python policy_analysis.py         # Policy charts, count deviations, sensitivity
```

Options: `--decks 2`, `--stages baseline count`, and `--hands 1000000` for a quick run.

| File | Purpose |
|---|---|
| `blackjack_shoe.py` | Shoe environment, Hi-Lo count, double down |
| `q_table.py` | Stage definitions, Q-learning, experiment runner |
| `evaluate.py` | Shared evaluation and reference policies |
| `metrics_and_plots.py` | Results table and learning curves |
| `policy_analysis.py` | Policy charts, count deviations, hyperparameter sensitivity |
| `traditional_baselines.py` | Exact model and policy iteration, used for basic strategy |
