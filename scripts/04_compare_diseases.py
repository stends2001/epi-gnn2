# %% [markdown]
# # 04 - Compare diseases: norovirus vs campylobacter vs influenza
#
# Same model and settings for each disease (each scored in its own season):
# baselines vs HHH4 (relative WIS) and the three components side by side. The
# transmission contrast belongs in 03 (graph controls) per disease; this script
# gives the overview.
#     python scripts/04_compare_diseases.py

# %% setup
import sys, pathlib  # make scripts/_common.py importable, also from the repo root or a notebook
_here = pathlib.Path(globals().get('__file__', pathlib.Path.cwd() / 'scripts' / '_')).resolve().parent
sys.path[:0] = [str(_here), str(pathlib.Path.cwd() / 'scripts')]

import pandas as pd

from _common import make_config, build_data, graph_builder, train_hhh4, fit_baselines, results_dir
from src.models.utils.intervalmetrics import compare_models
from src.models import diagnostics as dg

DISEASES = ['norovirus', 'campylobacter', 'influenza']
LEVEL    = 'nuts3'
LEAD     = 4
SEED     = 0

out = results_dir(f'diseases_{LEVEL}_lead{LEAD}')

# %% run
models, scores = {}, []
for disease in DISEASES:
    cfg = make_config(disease, LEVEL, lead=LEAD)
    edo, db = build_data(cfg)
    hhh = train_hhh4(graph_builder(edo, 'real'), name=f'hhh4_{disease}', seed=SEED)
    models[disease] = hhh
    tab = compare_models({'hhh4': hhh, **fit_baselines(db)}, season='in', reference='seasonal_average')
    tab.insert(0, 'disease', disease)
    scores.append(tab)

scores = pd.concat(scores, ignore_index=True)
print(scores[['disease', 'model', 'wis', 'rel_wis', 'cov50', 'cov80', 'cov95']].round(3).to_string(index=False))
scores.to_csv(out / 'scores_in_season.csv', index=False)

# %% components per disease (in season)
comp = dg.compare_components(models)
print(comp.round(3).to_string(index=False))
comp.to_csv(out / 'components_in_season.csv', index=False)

# %% sanity overview: number of WARN / FAIL per disease
for disease, m in models.items():
    rep = dg.sanity_report(m)
    print(f"{disease:14s} fail={int((rep['status'] == 'FAIL').sum())} warn={int((rep['status'] == 'WARN').sum())}")

# %% figures
for disease, m in models.items():
    dg.plot_component_shares(m).savefig(out / f'{disease}_component_shares.png', dpi=150, bbox_inches='tight')
    dg.plot_node_maps(m).savefig(out / f'{disease}_node_maps.png', dpi=150, bbox_inches='tight')
print(f'\nSaved to {out}')
