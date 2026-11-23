# Report Notes

Everything the final report needs, in the order of the guideline's report structure
(sections 1–7). Every number comes from a file in `outputs/`, named next to it, so the
report can be written without rerunning code. Methods details are in
[METHODS.md](METHODS.md).

These are notes, not report text: write the report in your own words (course GenAI
policy).

Format: NeurIPS LaTeX template, at most 7 pages for sections 1–6. Due Nov 23, 11:59 PM
on Gradescope (report + code), no grace period.

## Figures and tables

| Use in | File | Shows |
|---|---|---|
| §4.1 | `outputs/q_table_results.md` | Main results table, 6 decks |
| §4.1 | `outputs/q_table_results_1deck.md`, `_2deck.md` | Same for 1 and 2 decks |
| §4.2 | `outputs/count_deviations.png` | Plays that change with the count (1 deck, ±2) |
| §4.2 | `outputs/q_table_results_buckets*.md`, `count_deviations_buckets3.png` | ±3 vs ±2 buckets |
| §4.3 | `outputs/policy_vs_basic.png` | Learned policy vs basic strategy |
| §4.4 | `outputs/bust_penalty_policy.png` | Plays changed by the bust penalty |
| §4.4 | `outputs/q_table_results_shaping.md`, `learning_curves_shaping.png` | Shaping |
| §4.6 | `outputs/learning_curves.png`, `learning_curves_1deck.png` | Convergence |
| §4.6 | `outputs/sensitivity.png` | ε and step-size sensitivity |
| §3 | Validation table in [METHODS.md](METHODS.md) | Simulator vs exact model vs Gymnasium |

Raw data: `outputs/q_table_*.json` (per seed), `outputs/policy_analysis.json`
(policy comparisons, count deviations, bust-penalty analysis, sensitivity, validation).

## 1. Problem description

- **Question:** starting from a minimal observation, how does each change to what a
  tabular Q-learning agent sees (card count), can do (double down) or is rewarded for
  (bust penalty, shaping) change what it learns in Blackjack, and why?
- **Why Blackjack:** it is a small MDP with stochastic transitions and hidden
  information (dealer hole card, shoe composition). On an infinite deck it can be
  solved exactly with policy iteration, so learned values and policies can be checked
  against ground truth.
- **Relevance to probabilistic reasoning:**
  - Model-based (policy iteration with exact marginalisation over the hole card) vs
    model-free (Q-learning).
  - A finite shoe breaks the independence of card draws, so the standard state is no
    longer Markov. The count tests how much a summary statistic of the hidden state
    helps.
  - Reward design tests which reward changes preserve the optimal policy.
- **Scope change:** the proposal also planned an LLM comparison. After discussing with
  our TA, we focused on Q-learning.

## 2. Data sourcing and processing

See [METHODS.md](METHODS.md), "Data":

- simulated hands only
- why the two Kaggle datasets were not usable
- discretisation of the count and the number of cards, and the resulting table sizes

Data volume: 30M training hands × 5 seeds per configuration, 10M evaluation hands per
seed.

Distribution of the true count (rounded) at the first decision, which motivates the
clipping (`policy_analysis.json` → `true_count_shares`):

| Decks | \|count\| = 0 | 1 | 2 | ≥ 3 |
|---|---:|---:|---:|---:|
| 1 | 14.8% | 14.0% | 13.5% | 57.7% |
| 2 | 11.1% | 27.3% | 17.9% | 43.8% |
| 6 | 24.8% | 33.5% | 18.0% | 23.7% |

## 3. Modeling and inference

All in [METHODS.md](METHODS.md):

- the MDP and the Bellman equation
- independence assumptions (infinite deck, hole card, finite shoe, count)
- policy iteration and Q-learning with the convergence conditions
- the stage table, the evaluation protocol and the validation table

## 4. Results and discussion

### 4.1 Main results

6 decks, mean return per hand ± 95% interval over 5 seeds (`q_table_results.md`):

| Policy | Return / hand | Change vs baseline | Win | Bust | Double |
|---|---:|---:|---:|---:|---:|
| Baseline Q-table | −0.0458 ± 0.0005 | — | 43.0% | 18.4% | 0.0% |
| + number of cards | −0.0459 ± 0.0005 | −0.0001 ± 0.0001 | 43.0% | 18.4% | 0.0% |
| + Hi-Lo count | −0.0455 ± 0.0004 | +0.0003 ± 0.0002 | 43.0% | 18.1% | 0.0% |
| + double down | −0.0332 ± 0.0005 | +0.0126 ± 0.0004 | 42.9% | 17.7% | 8.3% |
| + count + double | −0.0329 ± 0.0004 | +0.0129 ± 0.0002 | 43.0% | 17.5% | 7.6% |
| + double, bust penalty 0.1 | −0.0361 ± 0.0009 | +0.0097 ± 0.0008 | 43.1% | 11.7% | 8.4% |
| Random | −0.3953 ± 0.0004 | −0.3495 | 28.1% | 31.2% | — |
| Hit below 17 | −0.0793 ± 0.0007 | −0.0335 | 40.8% | 28.2% | — |
| Basic strategy | −0.0458 ± 0.0004 | +0.0000 | 43.0% | 18.3% | — |
| Basic strategy + double | −0.0329 ± 0.0004 | +0.0129 ± 0.0003 | 42.9% | 17.8% | 8.0% |

Change vs baseline by shoe size (`q_table_results_1deck.md`, `_2deck.md`):

| Stage | 6 decks | 2 decks | 1 deck |
|---|---:|---:|---:|
| + count | +0.0003 ± 0.0002 | +0.0011 ± 0.0002 | +0.0024 ± 0.0001 |
| + double | +0.0126 ± 0.0004 | +0.0135 ± 0.0002 | +0.0146 ± 0.0001 |
| + count + double | +0.0129 ± 0.0002 | +0.0146 ± 0.0005 | +0.0177 ± 0.0001 |
| Basic strategy + double | +0.0129 ± 0.0003 | +0.0136 ± 0.0002 | +0.0147 ± 0.0001 |

- Overall, the Q-table improves from −0.0458 to −0.0329 and matches basic strategy +
  double on 6 decks.
- On 1 deck, count + double (−0.0258) beats basic strategy + double (−0.0288) by
  0.003 per hand, because basic strategy ignores the count.

### 4.2 Observation: Card Count 

- **The count helps more as the shoe gets smaller** (+0.0003 → +0.0011 → +0.0024).
  With fewer decks, each dealt card has a larger effect on the remaining shoe
  composition, so the dependence between draws is stronger.

- **The number of cards adds almost nothing** (−0.0001 ± 0.0001).
  This suggests that the useful state information comes from the count, not from
  hand length.

- **Count deviations** (1 deck, count + double, `count_deviations.png`): 54 of 240
  first-decision cells change action when count is included.
  - At +2, the table stands on 12 vs 2–6, 13 vs 2, 15 vs 10, and 16 vs 10, and doubles
    8 vs 5–6, 9 vs 2–4, 9 vs 7, and 11 vs 10.
  - At −2, it hits 13 vs 3–6, 14 vs 2–3, and 15 vs 2, and stops doubling 9 vs 5–6,
    10 vs 7–9, and 11 vs 8–9.
  - These changes follow the same direction as standard Hi-Lo index plays
    (Schlesinger, 2005): stand and double more at high counts, hit more and double
    less at low counts.
  - Soft-hand deviations are noisier because those states are rarer.

- **Buckets ±3 vs ±2** (paired difference, `q_table_results_buckets*.md`):

  | Decks | Count | Count + double | Hands with \|count\| ≥ 3 |
  |---|---:|---:|---:|
  | 1 | +0.0003 ± 0.0001 | +0.0002 ± 0.0002 | 58% |
  | 2 | +0.0002 ± 0.0003 | +0.0001 ± 0.0005 | 44% |
  | 6 | +0.0000 ± 0.0002 | −0.0002 ± 0.0003 | 24% |

  - Many hands fall beyond ±2, but finer buckets only help clearly in the 1-deck case,
    by about +0.0003.
  - The table also grows by 40% (2,600 → 3,640 states), which gives fewer visits per
    state.
  - With ±3 buckets, some plays shift from +2 to +3. For example, 12 vs 2–3 and
    15 vs 10 stand only at +3 (`count_deviations_buckets3.png`).
  - Overall trade-off: finer count resolution vs lower sample efficiency.

### 4.3 Action: double down (Member B's topic)

- **Double is the largest gain**, +0.0126 ± 0.0004, removing about 28% of the house
  edge.
- **The learned table differs from basic strategy in 4 of 240 first-decision cells**
  (`policy_vs_basic.png`; data in `policy_analysis.json` → `policy_vs_basic_6deck`):

  | Cell | Basic | Learned | Seeds choosing learned | Exact value gap |
  |---|---|---|---:|---:|
  | hard 12 vs 4 | S | H | 5/5 | 0.0025 |
  | soft 13 vs 6 | D | H | 4/5 | 0.018 |
  | soft 14 vs 5 | D | H | 3/5 | 0.014 |
  | soft 15 vs 5 | D | H | 3/5 | 0.034 |

- **Hard 12 vs 4 is a finite-shoe effect, not a learning error.**
  - On an infinite deck, standing is better by only 0.0025.
  - An infinite-deck control run stands there.
  - The same 6-deck table also stands on 12 vs 4 at later decisions (3+ cards), so it
    likely reflects removal of the player's first two cards.
- **The soft doubles are estimation noise in rare states.**
  - Each is dealt in 0.09% of hands.
  - Q(learned) − Q(basic) ranges from −0.07 to +0.06 across seeds, larger than the
    exact gaps.
  - The infinite-deck control flips similar soft cells (13 vs 6, 14 vs 5, 17 vs 2,
    17 vs 3).
- **No maximisation bias:** in the infinite-deck control, the learned Q-values' mean
  error against the exact values is under 0.0005 for every action (`q_error_by_action`).
- The 4 cells together cost 0.00008 per hand (exact gap × deal probability).

### 4.4 Reward: bust penalty and shaping (Member C's topic)

| Bust penalty | Bust rate | Return lost (measured, paired) | Predicted by exact solution |
|---:|---:|---:|---:|
| 0 | 17.7% | — | — |
| 0.1 | 11.7% | 0.0030 ± 0.0006 | 0.0033 |
| 0.25 | 6.6% | 0.0112 ± 0.0007 | 0.0109 |
| 0.5 | 1.1% | 0.0307 ± 0.0009 | 0.0305 |

- **Why the bust penalty hurts:**
  - Busting and losing to the dealer both cost 1 unit, but the penalty charges busts
    extra, so the agent optimizes a different objective. It stands on stiff hands where
    hitting is slightly better.
  - At 0.1 the learned table stands on 12 vs 3, 15 vs 10 and 16 vs 8–10, and also on
    16 vs 7 after the first decision (`bust_penalty_policy.png`).
- **Exact check:** solving the penalized objective on the infinite deck gives 13 policy
  changes at 0.1, and the learned table makes 11 of them. The predicted cost matches the
  measured loss at every penalty, and larger penalties change more cells (29, 49).
- **Why doubling rises:** the exact solution doubles 11 vs 10 instead of hitting,
  because a double on 11 cannot bust. This fits the double rate rising from 8.3% to
  about 9%.
- **Why shaping does not hurt:** F = Φ(s′) − Φ(s) with Φ(terminal) = 0 telescopes over a
  hand, so it cannot change the optimal policy (Ng et al., 1999).

  | Reward used in training (all with double) | Change vs baseline |
  |---|---:|
  | True reward | +0.0126 ± 0.0004 |
  | Bust penalty 0.1 | +0.0097 ± 0.0008 |
  | Shaping, Φ = total / 21 | +0.0127 ± 0.0002 |
  | Shaping, Φ = basic-strategy value | +0.0128 ± 0.0003 |

- **Shaping does not speed up learning either.** The curves overlap from 10k to 30M
  hands (`learning_curves_shaping.png`). At 100k hands: −0.0487 unshaped, −0.0478 and
  −0.0477 shaped.
  - A hand has only 1–3 decisions, so there is little delayed credit to speed up.
  - Shaping equals starting Q at Φ (Wiewiora, 2003), and the step size N^−0.85 forgets
    the starting value quickly.

### 4.5 Comparison with the "hit below 17" heuristic

- The heuristic loses −0.0793 per hand vs −0.0458 for the baseline Q-table, and busts
  28.2% of hands vs 18.4%.
- Where they differ (`policy_vs_basic.png`):
  - The learned table stands on 13–16 against a dealer 2–6 (the dealer is likely to
    bust), where the heuristic hits.
  - It hits soft 18 against 9, 10 and A, where the heuristic stands.

### 4.6 Convergence, stability and computational cost

Mean return over 5 seeds at each training budget, each checkpoint evaluated on 1M
hands, so each value is about ±0.001 (`q_table_ladder*.json`, `learning_curves*.png`):

| 6 decks | 10k | 100k | 1M | 10M | 30M |
|---|---:|---:|---:|---:|---:|
| Baseline | −0.0718 | −0.0506 | −0.0464 | −0.0458 | −0.0459 |
| + count | −0.1043 | −0.0588 | −0.0473 | −0.0458 | −0.0453 |
| + double | −0.0900 | −0.0487 | −0.0352 | −0.0329 | −0.0330 |
| + count + double | −0.1586 | −0.0718 | −0.0408 | −0.0338 | −0.0327 |

| 1 deck | 10k | 100k | 1M | 10M | 30M |
|---|---:|---:|---:|---:|---:|
| Baseline | −0.0652 | −0.0462 | −0.0441 | −0.0433 | −0.0433 |
| + count | −0.0888 | −0.0529 | −0.0430 | −0.0409 | −0.0408 |
| + double | −0.0888 | −0.0440 | −0.0308 | −0.0287 | −0.0283 |
| + count + double | −0.1292 | −0.0599 | −0.0336 | −0.0266 | −0.0253 |

- **Convergence:** the baseline and double tables converge by about 10M hands.
- **Larger tables need more data.**
  - On 6 decks the count tables are worse than their no-count versions at 1M hands
    (−0.0473 vs −0.0464; −0.0408 vs −0.0352), and pass them only at 10–30M.
  - On 1 deck, count + double is still improving between 10M and 30M, so more data
    would likely help it.
- **Stability across seeds:** 95% intervals are ±0.0005 or less for true-reward
  stages, and up to ±0.0010 with a bust penalty.
- **Sensitivity** (6 decks, + double, 3M hands, `sensitivity.png`): the step-size
  exponent matters more than ε. An exponent of 0.6 is 0.003–0.004 worse; 0.85 and 1.0
  are within 0.0005.

  | ε \ exponent | 0.6 | 0.85 | 1.0 |
  |---|---:|---:|---:|
  | 0.1 | −0.0390 | −0.0346 | −0.0346 |
  | 0.3 | −0.0370 | −0.0340 | −0.0335 |
  | 0.5 | −0.0371 | −0.0337 | −0.0331 |

- **Cost:**
  - Pure-Python training runs at about 190k hands per second per CPU core.
  - One 30M-hand run with its evaluations takes about 4–5 minutes on one core.
  - The full 6-deck ladder (40 runs + reference evaluations) took 30 minutes on
    14 cores; 1 deck took 39 minutes, because of more reshuffles.
  - Each added feature multiplies the table size (count × 5, ±3 buckets × 7), and so
    the data needed per state.

## 5. Conclusion

**Findings:**

- Q-learning recovers basic strategy on 6 decks.
- Double down is the most valuable change.
- The count matters only on small shoes, where count + double beats basic strategy.
- A bust penalty changes the optimal policy and costs exactly what the model predicts.
  Potential-based shaping does not change it.

**Limitations:**

- `Blackjack-v1` rules (a natural pays 1:1, no hole-card peek, no split, surrender or
  insurance) differ from casino rules, so returns are not casino house edges.
- The count changes only playing decisions, not bet size. In practice most of the value
  of counting comes from betting more at high counts, so our count gains understate it.
- Tabular states do not generalise; every added feature multiplies the data needed
  (§4.2, §4.6).
- Basic strategy is computed for an infinite deck, not the 6-deck shoe, so a few cells
  (12 vs 4) differ for real reasons.
- The count is a summary, not the full shoe composition.
- Rare soft states stay noisy even after 30M hands.
- All results are in simulation.

**Possible extensions:**

- bet sizing as an action (e.g., Kelly betting by count)
- splitting pairs
- function approximation over a continuous count
- an exact finite-shoe solution for 1 deck as a stronger reference
- decaying ε, or more hands for the count tables

## 6. Reflections and contributions

**Advice for future students (draft; edit as a team):**

- Evaluate on millions of hands with paired seeds. At 50k hands the 95% interval is
  about ±0.009, larger than the count effect.
- Validate the simulator against an exact model before training anything.
- An exact solution (policy iteration) is the best tool for explaining what the
  learned table does.

**Contributions** (1–2 sentences each, plus a short personal reflection in each
member's own words):

| Member | Contribution | Personal reflection |
|---|---|---|
| Ziyao Zhou | Built the finite-shoe environment with the Hi-Lo count and double down, the exact model and policy-iteration baseline, the Q-learning ladder and the shared evaluation protocol. | After meeting with our TA, we narrowed the project from LLM fine-tuning to Q-learning alone. I learned that plain data and direct computation often explain a model better than an LLM can, and that chasing a harder, more complex method is not always the most cost-effective choice. |
| Hao HUANG | Analyzed the observation-space modification by adding the Hi-Lo true count to the Q-table state. Compared ±2 and ±3 count buckets, identified count-dependent policy changes, and explained why count information helps more in smaller shoes. | I learned that adding more state information can improve decision quality, but it also increases the size of the Q-table and the amount of data needed. The count feature was most useful in one-deck Blackjack, where each observed card has a larger effect on the remaining shoe composition. |
| [Member B] | [placeholder] | [placeholder] |
| [Member C] | [placeholder] | [placeholder] |
| [Member D] | [placeholder] | [placeholder] |

**Use of generative AI (draft):**

> We used Claude (Anthropic) through Claude Code as a coding assistant. The research
> question, the experiment design (the observation, action and reward ladder, the
> count-bucket, bust-penalty and shaping comparisons) and the interpretation of results
> are our own. Claude implemented code from our specifications, helped debug, ran
> experiments and drafted repository documentation, including the exact checks we used
> to explain the double-down and bust-penalty results. We reviewed all code and checked
> every reported number against the outputs. [Add any other AI tools you used.]

## 7. References (APA)

- Even-Dar, E., & Mansour, Y. (2003). Learning rates for Q-learning. *Journal of
  Machine Learning Research, 5*, 1–25.
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward
  transformations: Theory and application to reward shaping. *Proceedings of the 16th
  International Conference on Machine Learning*, 278–287.
- Schlesinger, D. (2005). *Blackjack attack: Playing the pros' way* (3rd ed.). RGE
  Publishing.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*
  (2nd ed.). MIT Press.
- Thorp, E. O. (1966). *Beat the dealer* (2nd ed.). Random House.
- Towers, M., et al. (2024). Gymnasium: A standard interface for reinforcement learning
  environments. *arXiv:2407.17032*.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning, 8*(3–4),
  279–292.
- Wiewiora, E. (2003). Potential-based shaping and Q-value initialization are
  equivalent. *Journal of Artificial Intelligence Research, 19*, 205–208.
- Kaggle datasets checked in §2: *900,000 Hands of BlackJack Results* and *50 Million
  Blackjack Hands*. [Add authors and URLs.]
- Tools: Python, Gymnasium, Matplotlib (Hunter, J. D. (2007). Matplotlib: A 2D
  graphics environment. *Computing in Science & Engineering, 9*(3), 90–95.)
