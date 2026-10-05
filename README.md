# epi-gnn2

Probabilistic, explainable GNN forecasting of regional epidemic incidence.
Follow-up to *Spatial Resolution and Calibration: When Graph Neural Networks
Improve Epidemic Forecasting*.

**Current situation:** interval forecasts for baselines and GNNs are in place.
The hhh4-style GNN works on case counts with node-specific seasonality, and
diagnostics and run scripts cover calibration, the component split and graph
controls. Next: graph-control experiments per disease (norovirus,
campylobacter, influenza).

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

## Scripts

Set `GRAPH_FILE` (and dates if needed) in `scripts/_common.py`, then run from the
repo root, whole or cell by cell (`# %%`):

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
