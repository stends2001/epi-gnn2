# %% [markdown]
# # 03 - Does the model use the geography? Graph controls x seeds
#
# Trains the same HHH4 GNN on the real graph, an identity graph (no neighbours)
# and several degree-preserving rewired graphs (right number of neighbours, wrong
# ones), each with several seeds. Then:
#
# - WIS per graph (in season), mean and spread over seeds;
# - permutation p-value: is the real graph better than the rewired ones?
# - neighbourhood share per graph.
#
# If the real graph is not better than rewired graphs, the neighbourhood branch
# is not using who-borders-whom, whatever its share says.
#     python scripts/03_graph_controls.py      (slow: N_SEEDS x (2 + N_REWIRED) trainings)

# %% setup
import sys, pathlib  # make scripts/_common.py importable, also from the repo root or a notebook
_here = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / 'scripts' / '_')).resolve().parent
sys.path[:0] = [str(_here), str(pathlib.Path.cwd() / 'scripts')]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import make_config, build_data, graph_builder, train_hhh4, results_dir
from src.models.utils.intervalmetrics import evaluate_model_intervals
from src.models import diagnostics as dg

DISEASE   = 'norovirus'
LEVEL     = 'nuts3'
LEAD      = 4
N_SEEDS   = 5          # training seeds per graph
N_REWIRED = 10         # rewired graph draws (each trained with N_SEEDS seeds)
EPOCHS    = 200

cfg = make_config(DISEASE, LEVEL, lead=LEAD)
edo, db = build_data(cfg)
out = results_dir(f'{DISEASE}_{LEVEL}_lead{LEAD}', 'graph_controls')

# %% run all graphs x seeds
graphs = [('real', 0), ('identity', 0)] + [('rewired', g) for g in range(N_REWIRED)]
rows = []
for kind, gseed in graphs:
    gdb = graph_builder(edo, kind, seed=gseed)
    for seed in range(N_SEEDS):
        label = f'{kind}{gseed if kind == "rewired" else ""}_s{seed}'
        m = train_hhh4(gdb, name=label, seed=seed, global_hparams=dict(n_epochs=EPOCHS))
        wis_in = evaluate_model_intervals(m, season='in')['wis'].iloc[0]
        ct = dg.component_table(m)
        ins = ct[ct['period'] == 'in-season'].iloc[0]
        rows.append({'graph': kind, 'graph_draw': gseed, 'seed': seed, 'wis_in_season': wis_in,
                     'share_endemic': ins['share_endemic'], 'share_epidemic': ins['share_epidemic'],
                     'share_neighbourhood': ins['share_neighbourhood']})
        print(rows[-1])
res = pd.DataFrame(rows)
res.to_csv(out / 'runs.csv', index=False)

# %% summary per graph type
summary = (res.groupby('graph')[['wis_in_season', 'share_neighbourhood']]
              .agg(['mean', 'std', 'count']).round(4))
print(summary.to_string())
summary.to_csv(out / 'summary.csv')

# %% permutation test: real graph vs rewired draws (unit = graph, seeds averaged)
per_graph = res.groupby(['graph', 'graph_draw'])['wis_in_season'].mean()
real  = per_graph.loc[('real', 0)]
rewired = per_graph.loc['rewired'].to_numpy()
p_value = (1 + np.sum(rewired <= real)) / (len(rewired) + 1)      # lower WIS is better
seed_sd = res[res['graph'] == 'real']['wis_in_season'].std()
print(f'real WIS {real:.4f} | rewired mean {rewired.mean():.4f} (range {rewired.min():.4f}-{rewired.max():.4f})')
print(f'permutation p = {p_value:.3f} (smallest possible {1/(len(rewired)+1):.3f})')
print(f'difference rewired - real = {rewired.mean() - real:.4f}; seed-to-seed SD on real = {seed_sd:.4f}')

# %% figure: WIS and neighbourhood share per graph type
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
order = ['real', 'rewired', 'identity']
for ax, col, title in [(axes[0], 'wis_in_season', 'WIS in season (lower is better)'),
                       (axes[1], 'share_neighbourhood', 'Neighbourhood share in season')]:
    data = [res.loc[res['graph'] == g, col].to_numpy() for g in order]
    ax.boxplot(data, widths=0.5)
    ax.set_xticks(range(1, len(order) + 1), order)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1) + np.random.uniform(-0.12, 0.12, len(d)), d,
                   s=14, color=dg.plots.SERIES_COLORS[i], zorder=3)
    dg.plots._style(ax, title)
fig.tight_layout()
fig.savefig(out / 'graph_controls.png', dpi=150, bbox_inches='tight')
print(f'\nSaved to {out}')
