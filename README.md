# epi-gnn2

Probabilistic, explainable GNN forecasting of regional epidemic incidence.
Follow-up to *Spatial Resolution and Calibration: When Graph Neural Networks
Improve Epidemic Forecasting*.

**Current situation:** moving from point predictions to interval predictions.
Persistence calibrates well; Seasonal Average and the GNNs are being brought in
line, and an hhh4-style GNN with a Negative Binomial output is drafted.

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

### hhh4-style GNN

`HHH4Model` splits the NB mean into endemic (seasonal / covariate MLP),
epidemic (own incidence, linear) and neighbourhood (GCN over neighbours,
self-loops stripped) contributions, each non-negative.

```python
m = HHH4Model(db)
m.set_model_hparams(hidden_size=32, num_layers=1, alpha_mode='global')
m.set_global_hparams(lr=1e-3, n_epochs=200)          # loss='nb' picked automatically
m.train(); m.forecast('test')

components = m.forecast_components('test')          # per-branch means and shares
total, parts = m.sample_component_draws('test')     # joint draws, parts sum to total
```

Notes:

- The NB likelihood needs the raw target: set `normalization_method=None` and
  keep the target out of `log_transform`. Counts are the natural scale.
- Only the total is observed, so there is one dispersion for the total. Branch
  draws come from multinomial thinning of the total draw; they are attributions,
  not separately calibrated forecasts.
- Keep `num_layers=1` for a strict neighbourhood branch. With more layers a
  node's own signal returns through 2-hop paths (i -> j -> i).

## Evaluation

```python
from src.models.utils import evaluate_model_intervals

evaluate_model_intervals(model, 'test')                         # per horizon
evaluate_model_intervals(model, 'test', group_cols=['node'])    # per node
```

Returns WIS with its decomposition (dispersion, under-, overprediction),
coverage, share below / above, and mean width per central interval.
`quantile_ranks` gives a discrete PIT for histograms.

## Tests

```bash
python -m pytest tests -q
```

The tests use synthetic data and stubs, so they run without the surveillance data.
