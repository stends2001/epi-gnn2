# Attribution study, pilot (2 replicates)

Raw rows: `attribution_study_pilot.csv`. Reproduce / extend with
`python run.py configs/attribution_study.yaml` (default 3 replicates).

Setup: 4x4 grid of regions, 9 years (7 train, 1 val, 1 test), test season
in-season weeks 44-14. Neural model: node-specific dispersion, seasonal rates,
GRU rate dynamics, 80 epochs (patience 15). Forecasts scored 4 weeks ahead;
hhh4 and the one-step neural model forecast by simulation.

Sources with a known split:

- `sir_c0.0 / 0.3 / 0.6`: seasonal SIR waves per region, coupling c between
  neighbours (model-neutral: neither hhh4 nor the neural model);
- `hhh4_fitted / no_ne / strong_ne`: hhh4 fitted to an SIR series, simulated with
  the fitted, no, or a strong (75%) neighbourhood route.

## Attribution error (half the L1 distance of in-season shares to the truth; 0 = exact)

| source | hhh4 | neural, 1 week ahead | neural, 4 weeks ahead directly |
|---|---|---|---|
| sir_c0.0 | 0.107 | **0.089** | 0.140 |
| sir_c0.3 | 0.429 | **0.105** | 0.124 |
| sir_c0.6 | 0.423 | **0.127** | 0.150 |
| hhh4_fitted | **0.022** | 0.135 | 0.371 |
| hhh4_no_ne | **0.051** | 0.195 | 0.226 |
| hhh4_strong_ne | **0.054** | 0.199 | 0.459 |
| mean | 0.181 | **0.142** | 0.245 |

Lead-4 WIS (mean over sources): hhh4 0.544, neural 1-week 0.536, neural 4-week 0.536.
PIT coverage 95%: 0.947 / 0.934 / 0.948.

## Neighbourhood share in season (truth in last column)

| source | hhh4 | neural 1w | neural 4w | truth |
|---|---|---|---|---|
| sir_c0.0 | 0.075 | 0.048 | 0.074 | 0.000 |
| sir_c0.3 | 0.626 | 0.301 | 0.287 | 0.197 |
| sir_c0.6 | 0.652 | 0.321 | 0.307 | 0.229 |
| hhh4_fitted | 0.492 | 0.387 | 0.274 | 0.497 |
| hhh4_no_ne | 0.042 | 0.149 | 0.226 | 0.000 |
| hhh4_strong_ne | 0.550 | 0.396 | 0.276 | 0.595 |

## Reading

1. **One week ahead beats four weeks ahead directly for attribution** on every
   source (mean error 0.142 vs 0.245), at the same lead-4 WIS. The direct model's
   components are prediction weights; on hhh4 data it shifts mass from the
   epidemic to the endemic part and under-states the neighbourhood.
2. **hhh4 recovers its own data almost exactly** (home advantage), but on the
   model-neutral SIR data with spread it attributes about 0.63-0.65 to the
   neighbourhood where the truth is about 0.2: last week's neighbour counts act
   as a proxy for the shared seasonal force of infection. The one-step neural
   model, whose seasonal and GRU-driven rates absorb that common force, is much
   closer.
3. The neural model still reports some neighbourhood share where there is
   none on hhh4 data (0.15 in `hhh4_no_ne`), and it under-states a strong
   neighbourhood route (0.40 vs 0.60).
4. Per-region attribution is weak for all estimators on SIR data (correlations of
   per-region neighbourhood shares near 0 or negative): claims should be about
   overall / seasonal shares, not individual regions.
5. Anchoring the neural model on the hhh4 split (penalty weight 1 or 10, replicate
   0 only) reproduces hhh4's split almost exactly, including its SIR bias
   (e.g. sir_c0.3: neighbourhood 0.69-0.72 vs truth 0.23): it makes the neural
   model an hhh4 copy, not a better estimator. Kept as an option, not a default.

Caveats: 2 replicates, one grid, one SIR parameterisation; run the 3-replicate
default (and a larger grid) before quoting numbers.
