# epi-gnn2

Probabilistic, explainable GNN forecasting of regional epidemic incidence.
Follow-up to *Spatial Resolution and Calibration: When Graph Neural Networks
Improve Epidemic Forecasting*.

**Current situation:** interval forecasts for baselines and GNNs are in place.
The hhh4-style GNN works on case counts with node-specific seasonality, and
diagnostics and run scripts cover calibration, the component split and graph
controls. Next: graph-control experiments per disease (norovirus,
campylobacter, influenza).

## Running experiments

Set `data.graph_file` once in `configs/base.yaml`, then:

```bash
python run.py configs/smoke_test.yaml          # 3 epochs: does the pipeline run on your data?
python run.py configs/hhh4_norovirus.yaml      # HHH4 vs baselines, 3 seeds, all diagnostics
python run.py configs/hhh4_norovirus.yaml configs/hhh4_campylobacter.yaml   # several in a row
python run.py --list                           # what is available
```

| Config | Task |
|---|---|
| `smoke_test.yaml` | quick end-to-end check, no figures |
| `baselines_norovirus.yaml` | Persistence and Seasonal Average, additive and log1p residuals |
| `hhh4_norovirus.yaml`, `hhh4_campylobacter.yaml`, `hhh4_influenza.yaml` | neural model vs hhh4 in R vs baselines, 3 seeds, one test season |
| `seasons_norovirus.yaml`, `seasons_campylobacter.yaml`, `seasons_influenza.yaml` | the same over 4 test seasons (2015/16-2018/19), pooled |
| `ablations_norovirus.yaml`, `ablations_influenza.yaml` | model ladder (hhh4 in R, no GRU, full) and branch ablations |
| `recovery_norovirus.yaml`, `recovery_influenza.yaml` | simulate from fitted hhh4 with known components, refit, compare |
| `graph_controls_norovirus.yaml`, `graph_controls_campylobacter.yaml` | real vs identity vs 10 rewired graphs x 5 seeds, permutation p-value (slow) |
| `compare_diseases.yaml` | norovirus, campylobacter and influenza side by side |

Change anything without editing files, with dotted keys:

```bash
python run.py configs/hhh4_norovirus.yaml --set data.lead=2 data.level=nuts2 train.seeds=[0,1,2,3,4]
python run.py configs/hhh4_norovirus.yaml --dry-run      # show the resolved config, run nothing
```

A new experiment is a small YAML file that `extends:` another one and lists only
what differs. Each run writes `results/<name>/<timestamp>/` with `summary.txt`
(headline numbers), `log.txt`, `config.yaml` (the resolved config; run it again
to reproduce), CSV tables and `figures/`.

## hhh4 in R as reference model

`run.py` fits hhh4 (R package `surveillance`) itself whenever `hhh4_r.enabled: true`
(the default): Python writes counts, graph and population to the run folder, calls
`Rscript src/experiments/r/hhh4_fit.R`, and reads the forecasts back as model
`hhh4_R`, which then appears in all score tables and calibration plots.

Setup once: install R, then in R `install.packages("surveillance")`. If `Rscript`
is not on the PATH, set `hhh4_r.rscript` to its full path. Without R the runs
still work and say that hhh4_R was skipped.

- Model: region random intercepts and week-of-year seasonality in the endemic,
  epidemic and neighbourhood parts, power-law neighbourhood weights, NegBin.
- Lead-L forecasts come from simulating the fitted model forward from each
  forecast origin (`nsim` paths), as hhh4 forecasts are normally made.
- Components are one step ahead (`hhh4_R_component_table.csv`); compare them with
  the neural model at `data.lead: 1`.

## Coverage for counts

Quantiles of a count forecast are whole numbers, so a "50% interval" holds more
than 50% of the probability, most of all for small counts. For models with a full
predictive distribution (neural model, hhh4_R) the tables therefore also report
`pitcov50/80/95`: coverage from the randomised PIT, which is exactly nominal for a
calibrated count forecast. Read `pitcov`, not `cov`, for these models. The sanity
report does the same.

## Recovery study and ablations

- `task: recovery` fits hhh4 to the real counts, simulates new series under
  scenarios with a known split (`fitted`, `no_ne` = no spread between regions,
  `strong_ne` = 75% via neighbours), refits the neural model and hhh4 on them, and
  compares the estimated shares and per-region correlations with the truth
  (`recovery.csv`, `figures/recovery_shares.png`). On test data this showed the
  neural model inflating the neighbourhood share when there is no spread, which
  hhh4 did not: the simulated data come from hhh4, so hhh4 has a home advantage.
- `task: ablations` trains the full model and variants that each change one thing
  (no GRU, no seasonal rates, no neighbourhood, no epidemic, no node effects), and
  hhh4_R, on the same seeds; WIS is relative to the full model.

## Interval mode

Set quantile levels in `EpiConfig`, as decimals:

```python
cfg = EpiConfig(..., quantiles=[0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
```

The list must be odd-length, strictly increasing and symmetric around 0.5
(each pair `q, 1-q` is one central interval with nominal coverage `1 - 2q`).
Predictions are then stored as `pred_q1 ... pred_qN`; the median is the
central forecast for WIS.

## Models and how they produce intervals

| Model | Output | Intervals | Loss |
|---|---|---|---|
| `Persistence` | last observation | residual quantiles per (horizon, week of year) | none |
| `SeasonalAverage` | node mean per week of year (train) | residual quantiles per week of year | none |
| `GCNModel` / `GATModel`, `output_head='point'` | point | split-conformal on val, per (horizon, week) | `mse` |
| `GCNModel` / `GATModel`, `output_head='quantile'` | non-crossing quantiles | direct; optional CQR with `conformalize=True` | `pinball` |
| `HHH4Model` | `(mu, alpha)`, three branches | exact NB2 quantiles | `nb` |

`loss='auto'` (the default in `set_global_hparams`) picks the loss that matches
the head.

### Baseline calibration options

```python
Persistence(db, residual_scale='additive', min_bin_obs=30, calibration_splits=['train', 'val'])
SeasonalAverage(db, residual_scale='log1p', train_residuals='leave_one_year_out')
```

- `residual_scale='log1p'` uses relative errors: suits regions of different
  size, never negative, no zero-width interval at `pred = 0`.
- Seasonal bins with fewer than `min_bin_obs` residuals use the pooled
  quantiles. `model.calibration_summary()` lists `n_obs` and fallbacks per bin.
- `train_residuals='leave_one_year_out'` removes the in-sample optimism of
  Seasonal Average's train residuals (its mean is fit on train).
- The test split is never used for calibration.

### Case counts

Set `target_column='cases'` (and `lag_column='cases'`). The target and its lags
then stay raw counts, whatever `normalization_method` says; only other features
(population size / density) are normalised. `'cases'` may not appear in
`log_transform`. This is the setup the NB likelihood and the rate form of
`HHH4Model` need.

### hhh4-style GNN

`HHH4Model` splits the NB mean into three non-negative parts, in the hhh4 form:

```
endemic_i       = exp(a_i + seasonal_i(week))                  node-specific seasonality
epidemic_i      = lambda_i * (lag-weighted own cases)
neighbourhood_i = phi_i    * (lag-weighted mean of neighbours' cases, self-loops removed)
```

- **Node-specific seasonality:** each region has its own level and its own
  coefficients on the week-of-year sin/cos features, so its own peak week and
  amplitude. Node deviations are centred and ridge-penalised (`node_penalty`),
  pooling regions with little data towards the shared curve.
- **Time-varying rates:** the rates also get week-of-year terms, and a GRU
  (or LSTM, `rate_dynamics`) reads the recent trajectory of own and neighbour
  counts and shifts both rates per region and week, bounded and starting at 0.
  This lets the model follow epidemics that grow several-fold within the lead time
  and then collapse (influenza). On simulated influenza-like seasons (lead 4,
  three test seasons) WIS dropped from 3.23 (constant rates) to 2.29 (seasonal
  terms) and 2.06 (GRU). `forecast_components` reports the weekly rate
  multipliers; `plot_rate_multipliers` shows them.
- **Interval calibration:** `calibrate_dispersion()` picks one multiplier for the
  NB dispersion that minimises the in-season WIS on the validation split
  (`train.calibrate_dispersion: true` in the configs).
- **Seeds:** with `train.shuffle: true` the training weeks are visited in a new
  seeded order every epoch. Without it the rate-form model trains identically for
  every seed.
- **Rates:** `lambda_i`, `phi_i` are a shared rate times a node effect. The neural
  alternatives (`epidemic_mode='neural'`, `neighbourhood_mode='linear'|'gcn'`) are
  kept, but on simulated data with real spread between regions they let the
  neighbourhood branch collapse to a zero share; the rate form recovered it.

```python
m = HHH4Model(gdb)
m.set_model_hparams(alpha_mode='node')                # defaults: loglinear endemic, rate forms
m.set_global_hparams(lr=5e-3, n_epochs=300, patience=30)   # loss='nb' picked automatically
m.train(); m.forecast('test')

m.node_parameters()                                  # per region: peak week, amplitude, rates, alpha
m.forecast_components('test')                        # per-branch means and shares
total, parts = m.sample_component_draws('test')      # joint draws, parts sum to total
```

Notes:

- Only the total is observed, so there is one dispersion for the total. Branch
  draws come from multinomial thinning of the total draw; they are attributions,
  not separately calibrated forecasts.
- **The neighbourhood share is not a transmission measure on its own.** When all
  regions follow the same seasonal curve, the neighbours' mean also predicts the
  seasonal level. On simulated data without any spread the neighbourhood share
  was still about 0.15 (0.5 with spread). Use the graph controls (identity,
  rewired) and contrasts between diseases.
- Learning rate matters: around `5e-3` worked well on simulated data; `3e-2` was
  noisy and let branches collapse.

## Evaluation

```python
from src.models.utils import evaluate_model_intervals

evaluate_model_intervals(model, 'test')                         # per horizon
evaluate_model_intervals(model, 'test', group_cols=['node'])    # per node
```

Returns WIS with its decomposition (dispersion, under-, overprediction),
coverage, share below / above, and mean width per central interval.
`quantile_ranks` gives a discrete PIT for histograms.

In-season scoring and model comparison:

```python
from src.models.utils.intervalmetrics import evaluate_model_intervals, compare_models

evaluate_model_intervals(model, season='in')        # season window per disease, by target week
compare_models({'hhh4': m, 'persistence': p, 'seasonal': s}, season='in', reference='seasonal')
```

Off-season weeks with zero counts are covered by almost any interval, so pooled
coverage looks better than it is. Default seasons: influenza and norovirus weeks
40-15, campylobacter weeks 22-40 (`DEFAULT_SEASONS`).

## Diagnostics

```python
from src.models import diagnostics as dg

dg.print_sanity(dg.sanity_report(m, baselines={'persistence': p, 'seasonal': s}))
dg.component_table(m)        # shares: all / in-season / off-season, mu-weighted and median
dg.component_by_node(m)      # per region, with node parameters
dg.compare_components({'noro': m1, 'campy': m2})
```

`sanity_report` marks each check PASS / WARN / FAIL: missing values, crossing
quantiles, bias of the median and of mu, correlation with the truth, a lag check
(does the median just copy the last observation?), in-season coverage per
interval, zero-width intervals, PIT tails, relative WIS against baselines, and
for `HHH4Model` whether the components add up and each branch is used.

Figures (all return a matplotlib figure): `plot_decomposition`,
`plot_component_shares`, `plot_node_maps` (peak week, amplitude, rates, shares,
bias per region), `plot_seasonal_curves`, `plot_calibration` (coverage and PIT),
`plot_lag_check`, `plot_pred_vs_obs`, `plot_model_comparison`.

## Graph controls

```python
from src.graphconstruction import identity_graph, rewired_graph

gdb = GraphDataBuilder(edo).retrieve_static_graph(GRAPH_FILE)
gdb.use_graph(rewired_graph(gdb.graph, seed=3)).build()   # same degrees, scrambled geography
```

## Interactive scripts

The same steps as notebook-style scripts, for exploring step by step. They read
the graph file and dates from `configs/base.yaml`. Run from the repo root, whole
or cell by cell (`# %%`):

| Script | What it does |
|---|---|
| `scripts/01_baselines.py` | baselines, both residual scales, in-season and all-week scores |
| `scripts/02_hhh4.py` | HHH4 vs baselines, sanity report, components, all figures |
| `scripts/03_graph_controls.py` | real vs identity vs rewired graphs x seeds, permutation p-value |
| `scripts/04_compare_diseases.py` | norovirus, campylobacter, influenza side by side |

Outputs go to `results/` (not tracked). The default test season is 2018/19,
before COVID-19 measures.

## Tests

```bash
python -m pytest tests -q
```

The tests use synthetic data and stubs, so they run without the surveillance data.
