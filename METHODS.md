# Methods

## Model: Blackjack as a Markov decision process

A hand is an episode. A state s is what the player sees at a decision: in the baseline,
the player total, the dealer upcard and whether the player holds a usable ace. Actions
are stick and hit, plus double where allowed. The reward is 0 until the hand ends, then
+1, 0 or −1 (doubled after a double down). With γ = 1 the optimal action values satisfy

Q\*(s, a) = Σ_s′ P(s′ | s, a) [ r(s, a, s′) + max_a′ Q\*(s′, a′) ].

**Independence assumptions.**

- *Infinite deck* (`Blackjack-v1`): each card is drawn independently, with probability
  1/13 per rank and 4/13 for ten-valued cards. The next card does not depend on the
  history, so (total, upcard, usable ace) is a Markov state, and P(s′ | s, a) and the
  dealer's outcome distribution can be computed exactly.
- *Dealer hole card*: hidden from the player. On an infinite deck it is independent of
  the player's cards, so the value of sticking marginalises it out:
  P(dealer final total | upcard) = Σ_hole P(hole) · P(final | upcard, hole).
  - The last term is computed by recursion over the dealer's draws. Let F(h) be the
    distribution of the dealer's final total from hand h, which is a total plus
    whether it holds a usable ace.
  - If the total of h is 17 or more, F(h) puts all its mass on that total. Every total
    over 21 counts as one "bust" outcome.
  - Otherwise F(h) = Σ_c P(c) · F(h + c), where h + c adds card c and an ace counts 11
    unless that busts.
  - Sticking on total t against upcard u is then worth
    Q(t, u, stick) = Σ_d P(d | u) · sign(t − d), with a dealer bust counted as d = 0.
- *Finite shoe*: cards are drawn without replacement, so they are not independent. The
  next card depends on every card already dealt, and the shoe composition is a hidden
  part of the true state. The baseline state is then not Markov (the problem is
  partially observable), and a table over it learns an average over shoe compositions.
- *Hi-Lo true count*: a one-number summary of the hidden composition (the excess of
  high over low cards per unseen deck). Adding it to the state recovers part of the
  dependence between cards. It is a summary, not a sufficient statistic.

**Algorithms.**

- *Policy iteration* (model-based) solves the infinite-deck MDP exactly. It starts from
  "always stick" and alternates two steps until the policy stops changing:
  - evaluation: V^π(s) = Σ_s′ P(s′ | s, π(s)) [ r(s, π(s), s′) + V^π(s′) ], iterated
    until no value changes by more than 10^−10;
  - improvement: π(s) ← argmax_a Σ_s′ P(s′ | s, a) [ r(s, a, s′) + V^π(s′) ].

  The resulting policy is basic strategy, our reference.
- *Q-learning* (model-free) learns from sampled hands without the model:
  Q(s, a) ← Q(s, a) + α (r + max_a′ Q(s′, a′) − Q(s, a)), with α = N(s, a)^−0.85.
  This step size satisfies the Robbins–Monro conditions (Σα = ∞, Σα² < ∞), so on a
  Markov state with every action tried infinitely often, tabular Q-learning converges
  to Q\* (Watkins & Dayan, 1992). On a finite shoe the state is not Markov, so this
  guarantee does not strictly apply.

## Data

All data are hands simulated from the rules below; no external dataset is used. We
checked two Kaggle Blackjack datasets (*900,000 Hands of BlackJack Results* and
*50 Million Blackjack Hands*, 1,000 rows each). The first has only hand outcomes. The
second has action histories but no state at each decision. Neither provides state
transitions, so neither suits Q-learning.

**Processing (discretisation).** A table needs a finite state space, so continuous or
unbounded quantities are discretised:

- The true count is rounded to the nearest integer and clipped to ±2 (±3 in one
  comparison). Clipping merges rare extreme counts, so each table entry gets more
  visits, at the cost of resolution.
- The number of cards is capped at 4.

Table sizes: baseline 280 states, + number of cards 740, + count 1,400, + double 520,
+ count + double 2,600 (±3 buckets: 1,960 and 3,640).

## Environment

We use the rules of Gymnasium `Blackjack-v1` (`natural=False`, `sab=False`): the dealer
stands on all 17s, a natural pays 1:1, the dealer does not check for blackjack, and there
is no split, surrender or insurance. `Blackjack-v1` draws from an infinite deck and has no
double down, so `blackjack_shoe.py` changes two things:

- **Finite shoe.** Cards are dealt from a shuffled shoe of 1, 2 or 6 decks (6 unless
  stated). The shoe is reshuffled before a hand once 75% of it has been dealt.
- **Double down (optional).** On the first decision the player may double the bet, take
  exactly one card and stand.

**Hi-Lo count.** Each card the player sees (own cards, the upcard, the dealer's draws,
and the hole card once revealed) adds +1 for 2–6, 0 for 7–9 and −1 for tens and aces.
The count resets at each shuffle. The true count is the running count divided by the
number of unseen decks.

**Validation.** With an infinite deck the simulator matches the exact model and
`Blackjack-v1`. The table uses 20M simulated hands per policy and 10M Gymnasium hands
(`outputs/validation_20m.json`). Infinite-deck hands are independent, so the 95%
intervals here are computed over hands.

| Policy | Exact | Our simulator | Gymnasium |
|---|---:|---:|---:|
| Basic strategy | −0.0466 | −0.0467 ± 0.0004 | −0.0466 ± 0.0006 |
| Basic strategy + double | −0.0340 | −0.0343 ± 0.0005 | — |

- Every simulated value is within its 95% interval of the exact value. The largest gap
  is 0.0003.
- The 2M-hand run in `policy_analysis.json` (−0.0452 ± 0.0013 for basic strategy) sat
  at the edge of its interval, so we reran it with 10 times as many hands.
- To reproduce with the same function (about 9 minutes on one core):

  ```
  python -c "import json; from policy_analysis import validation; json.dump(validation(20_000_000), open('outputs/validation_20m.json', 'w'), indent=2)"
  ```

## Q-learning ladder

Every stage is tabular Q-learning with the same settings: γ = 1, ε-greedy exploration
over legal actions with ε = 0.3, Q initialised to 0, and step size α = N(s, a)^−0.85,
where N(s, a) counts visits to that state–action pair. The target is
r + max_a′ Q(s′, a′) over the legal actions at s′, or r at the end of the hand. Each stage
changes one thing from the baseline:

| Stage | State | Actions | Training reward |
|---|---|---|---|
| Baseline | player total, dealer upcard, usable ace | stick, hit | true |
| + number of cards | baseline + min(cards, 4) | stick, hit | true |
| + Hi-Lo count | baseline + true-count bucket | stick, hit | true |
| + double down | baseline + "first decision" flag | stick, hit, double | true |
| + count + double | both of the above | stick, hit, double | true |
| Bust penalty p | as "+ double down" | stick, hit, double | true − p if the player busts |
| Potential shaping | as "+ double down" | stick, hit, double | true + Φ(s′) − Φ(s) |

Bust penalties are p ∈ {0.1, 0.25, 0.5}, independent of the bet. For shaping, Φ is
either total / 21 or the infinite-deck basic-strategy value V\*(s), with Φ = 0 at the end
of a hand.

Each stage is trained with 5 seeds (0–4) for 30M hands per seed.

## Evaluation

After training, the greedy policy is frozen; states never visited fall back to stick.
Each policy plays 10M hands on a shoe seeded by the training seed. Every stage sees the
same evaluation shoes, so differences between stages are paired by seed. Policies are
always scored on the true reward.

We report the mean return per hand with a 95% t-interval over the 5 seeds, the paired
change from the baseline, and win, bust and double rates. Hands within one shoe are not
independent, so intervals are computed across seeds (independent runs), not across
hands. Learning curves evaluate checkpoints from 10k to 30M training hands on 1M hands
each.

Reference policies: uniform random, "hit below 17", and basic strategy. Basic strategy
is the optimal infinite-deck policy from policy iteration on the exact model, with the
dealer's hole card marginalised out; "+ double" adds double wherever its exact value
exceeds both stick and hit. Basic strategy ignores the count.

## Policy analysis

- **Policy charts.** For every first decision (two cards, true count 0) we take the
  majority greedy action over the 5 seeds and compare it with basic strategy. Each
  disagreement is described by its seed votes, the learned Q-value gap, the exact
  infinite-deck value gap, and the expected cost per hand (gap × probability of being
  dealt that state). A control run with an infinite deck separates learning error from
  finite-shoe effects.
- **Count deviations.** On 1 deck, where the count matters most, we list the cells
  whose majority action changes across count buckets, for buckets clipped at ±2 and ±3.
- **Bust penalty.** We compare the bust-penalty (p = 0.1) policy with the true-reward
  policy, and solve the penalised objective exactly on the infinite deck: value
  iteration with a bust reward of −1 − p, then policy evaluation on the true reward.
  This gives the predicted policy changes and their cost per hand.
- **Reproduction check.** `report/make_figures.py --maps` retrains the 15 tables behind
  the policy maps (6-deck + double, 1-deck count + double, 6-deck bust penalty 0.1;
  seeds 0–4) and stops unless they give the same 4 disagreements with basic strategy
  and the same 54 count-dependent cells as `outputs/policy_analysis.json`.
- **Sensitivity.** "+ double down" is trained with ε ∈ {0.1, 0.3, 0.5} and step-size
  exponents {0.6, 0.85, 1.0}: 3M training hands and 2M evaluation hands per seed,
  5 seeds.
