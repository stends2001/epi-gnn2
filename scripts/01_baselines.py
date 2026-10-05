# %% [markdown]
# # 01 - Baselines with calibrated intervals
#
# Persistence and Seasonal Average on case counts, both residual scales, scored
# in season and over all weeks. Run from the repo root:
#     python scripts/01_baselines.py
# or cell by cell in VS Code / PyCharm.

# %% setup
import sys, pathlib  # make scripts/_common.py importable, also from the repo root or a notebook
_here = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / 'scripts' / '_')).resolve().parent
sys.path[:0] = [str(_here), str(pathlib.Path.cwd() / 'scripts')]

from _common import make_config, build_data, results_dir
from src.models import Persistence, SeasonalAverage
from src.models.utils.intervalmetrics import compare_models
from src.models.diagnostics import sanity_report, print_sanity, plot_calibration, plot_model_comparison

DISEASE = 'norovirus'
LEVEL   = 'nuts3'
LEAD    = 4

cfg = make_config(DISEASE, LEVEL, lead=LEAD)
edo, db = build_data(cfg)
out = results_dir(f'{DISEASE}_{LEVEL}_lead{LEAD}', 'baselines')

# %% fit
models = {
    'persistence (additive)': Persistence(db, name='persistence_add', residual_scale='additive'),
    'persistence (log1p)':    Persistence(db, name='persistence_log', residual_scale='log1p'),
    'seasonal avg (additive)': SeasonalAverage(db, name='seasonal_add', residual_scale='additive'),
    'seasonal avg (log1p)':    SeasonalAverage(db, name='seasonal_log', residual_scale='log1p'),
}
for m in models.values():
    m.forecast('test')

# %% scores: in season (the part that matters) and all weeks
tab_in  = compare_models(models, season='in',  reference='persistence (log1p)')
tab_all = compare_models(models, season=None,  reference='persistence (log1p)')
print('IN SEASON\n', tab_in.round(3).to_string(index=False))
print('\nALL WEEKS\n', tab_all.round(3).to_string(index=False))
tab_in.to_csv(out / 'scores_in_season.csv', index=False)
tab_all.to_csv(out / 'scores_all_weeks.csv', index=False)

# %% thin bins: share of week-of-year bins that fell back to pooled quantiles
for label, m in models.items():
    s = m.calibration_summary()
    print(f'{label:26s} fallback share: {s["fallback"].mean():.2f}')

# %% sanity checks
for label, m in models.items():
    print_sanity(sanity_report(m))

# %% figures
plot_model_comparison(tab_in).savefig(out / 'comparison_in_season.png', dpi=150, bbox_inches='tight')
plot_calibration(models).savefig(out / 'calibration.png', dpi=150, bbox_inches='tight')
print(f'\nSaved tables and figures to {out}')
