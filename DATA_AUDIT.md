# Kaggle Data Check (from Ziyao)

I checked 1,000 rows from each dataset. The 900,000-hands file only has hand
results, so it cannot be used for state transitions. The larger blackjack-hands
file has action histories, but no per-action states or next states, so it also
does not directly provide compatible transitions. I skipped cleaning both
datasets and will generate expert examples from our Gymnasium environment.
