# %% [markdown]
# # 02 - HHH4 GNN on case counts: train, check, explain
#
# Trains the hhh4-style GNN (node-specific seasonality, rate-form epidemic and
# neighbourhood branches, NB output), compares it with the baselines on the same
# config, runs the sanity checks and saves the tables and figures.
#     python scripts/02_hhh4.py

# %% setup
import sys, pathlib  # make scripts/_common.py importable, also from the repo root or a notebook
_here = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / 'scripts' / '_')).resolve().parent
sys.path[:0] = [str(_here), str(pathlib.Path.cwd() / 'scripts')]

from _common import make_config, build_data, graph_builder, train_hhh4, fit_baselines, results_dir
from src.models.utils.intervalmetrics import compare_models
from src.models import diagnostics as dg

DISEASE = 'norovirus'
LEVEL   = 'nuts3'
LEAD    = 4
SEED    = 0

cfg = make_config(DISEASE, LEVEL, lead=LEAD)
edo, db = build_data(cfg)
gdb = graph_builder(edo, 'real')
out = results_dir(f'{DISEASE}_{LEVEL}_lead{LEAD}', 'hhh4')

# %% train
hhh = train_hhh4(gdb, name=f'hhh4_{DISEASE}', seed=SEED)
hhh.show_monitoring_metrics()

# %% baselines on the same config, and the score table (in season)
baselines = fit_baselines(db)
models = {'hhh4': hhh, **baselines}
scores = compare_models(models, season='in', reference='seasonal_average')
print(scores.round(3).to_string(index=False))
scores.to_csv(out / 'scores_in_season.csv', index=False)

# %% sanity checks: read the WARN / FAIL lines
report = dg.sanity_report(hhh, baselines=baselines)
dg.print_sanity(report)
report.to_csv(out / 'sanity.csv', index=False)

# %% the three components
comp = dg.component_table(hhh)
print(comp.round(3).to_string(index=False))
comp.to_csv(out / 'components.csv', index=False)

by_node = dg.component_by_node(hhh)
by_node.to_csv(out / 'components_by_node.csv', index=False)
cols = ['node_name', 'share_endemic', 'share_epidemic', 'share_neighbourhood',
        'endemic_peak_week', 'seasonal_amplitude', 'epidemic_rate', 'neighbourhood_rate', 'bias_ratio']
print('\nLargest neighbourhood shares (in season):')
print(by_node.sort_values('share_neighbourhood', ascending=False)[cols].head(10).round(3).to_string(index=False))

# %% figures
figs = {
    'decomposition':     dg.plot_decomposition(hhh, nodes=[0, 1, 2]),
    'component_shares':  dg.plot_component_shares(hhh),
    'node_maps':         dg.plot_node_maps(hhh),
    'seasonal_curves':   dg.plot_seasonal_curves(hhh),
    'calibration':       dg.plot_calibration(models),
    'lag_check':         dg.plot_lag_check(models),
    'pred_vs_obs':       dg.plot_pred_vs_obs(hhh),
    'model_comparison':  dg.plot_model_comparison(scores),
}
for name, fig in figs.items():
    fig.savefig(out / f'{name}.png', dpi=150, bbox_inches='tight')
print(f'\nSaved tables and figures to {out}')
