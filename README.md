# GTA VI Trophy Rarity Forecast — RACE

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`: forecast,
for each of GTA VI's PlayStation 5 trophies, the percent of the players
Exophase tracks who will have earned it at 2026-11-26T00:00:00Z, as five
quantiles (P05, P25, P50, P75, P95) per trophy, scored by pinball loss on
the logit scale.

## The model: RACE (Refresh-Aware Cohort Ensemble)

The number on Exophase's page is a ratio, `p = 100 * K / N`: earners over
tracked players. The two counts move for different reasons — `K` grows as
people play through the game, while `N` jumps whenever Exophase refreshes
its counts and newly synced players join the denominator. A refresh can
therefore move a rate a lot while nobody's progress changed, and a direct
trend on the rate confuses dilution with progression. RACE works on the
counts instead:

1. **Refresh detection.** The house reads every 10 minutes but the page
   only changes at a refresh, so the reads are collapsed into distinct
   page states (refresh events). Hundreds of identical reads are one
   observation, not hundreds.
2. **Cohort decomposition.** Each event's rates are inverted to earner
   counts, `K_i = p_i * N / 100`.
3. **Saturating fractions.** Each trophy's earn fraction `f_i(t) = K_i/N`
   is fitted with a saturating curve `F_i * (1 - exp(-t / tau_i))`, with
   `tau_i` shrunk toward its tier's median (partial pooling). A dilution
   trough — the fraction falling early while the tracked base surges, then
   rising — is handled by fitting the rising phase from the trough onward.
   The fraction is what saturates; the counts keep growing with the player
   base, so a log-linear extrapolation of the counts would run away.
4. **Refresh lag.** The answer is the page state at the *last* refresh
   before the target, so the curves are read at the target minus the
   typical refresh lag.
5. **Predictive spread.** A deterministic bootstrap of whole refresh-event
   residual vectors (a dilution shock hits every trophy at once), plus
   refresh-lag draws and a floor that grows when few events pin the fits
   down.
6. **Ensemble.** The cohort forecast is blended 0.85 / 0.15 with a robust
   damped logit-trend forecast, as a hedge against the cohort's functional
   form being wrong for the real data. When the history holds only one
   distinct state (the practice cases), a prior takes over: the kit's own
   documentation records the first launch-night refresh (638 -> 3,868
   tracked players), and the forecast expects a refresh of that kind, with
   the new cohort earning a few of the earliest trophies.

Any recoverable failure falls back to the house persistence forecast, so
the program always writes a valid file.

## Results

Backtested on synthetic launch weeks (9 scenarios x 7 rolling-origin cuts,
several data-generating shapes including a genuine dilution dip), mean
pinball loss:

| model | loss |
|---|---|
| persistence (house baseline) | 0.532 |
| house fitted-curve baseline | 0.532 |
| robust logit-trend | 0.436 |
| **RACE (cohort)** | **0.264** |
| RACE ensemble (shipped) | ~0.263 |

Practice cases (the two 90-minute forecasts over the first refresh):
RACE loss ≈ 0.13 against the practice answers reconstructed from the
documented refresh, versus 1.51 for the house baselines (documented).
The practice is only a 90-minute, single-refresh check; the final horizon
is 6,510 minutes, which is what the model above is built for.

## Running it

Python 3.12+ (grader runs 3.13), standard library only, no network,
deterministic (fixed-seed generator; identical inputs give byte-identical
outputs), well under the 8 s / 1,024 MB limits (about 0.1 s on the practice
cases).

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `*.out` per input:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.
