#!/usr/bin/env python3
"""
Run an experiment from a YAML config.

    python run.py configs/hhh4_norovirus.yaml
    python run.py configs/hhh4_norovirus.yaml --set data.disease=campylobacter train.seeds=[0,1,2]
    python run.py configs/graph_controls_norovirus.yaml --dry-run
    python run.py --list

Several configs run one after another:

    python run.py configs/hhh4_norovirus.yaml configs/hhh4_campylobacter.yaml

Each run writes config.yaml, log.txt, summary.txt, CSV tables and figures to
results/<run name>/<timestamp>/ (or ``output_dir`` from the config).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.experiments.config import ConfigError, load_config, run_name  # noqa: E402

CONFIG_DIR = ROOT / 'configs'


def _list_configs() -> None:
    print(f'Configs in {CONFIG_DIR}:')
    for p in sorted(CONFIG_DIR.glob('*.yaml')):
        with open(p) as f:
            first = f.readline().strip().lstrip('#').strip()
        print(f'  {p.name:36s} {first}')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Run epi-gnn2 experiments from YAML configs.')
    ap.add_argument('configs', nargs='*', help='one or more YAML config files')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                    help='override config values, e.g. data.lead=2 train.seeds=[0,1]')
    ap.add_argument('--dry-run', action='store_true', help='print the resolved config and stop')
    ap.add_argument('--list', action='store_true', help='list the available configs')
    ap.add_argument('--output-dir', help='override output_dir')
    ap.add_argument('--keep-going', action='store_true',
                    help='with several configs, continue after a failed run')
    args = ap.parse_args(argv)

    if args.list or not args.configs:
        _list_configs()
        return 0

    import yaml
    status = 0
    for path in args.configs:
        p = Path(path)
        if not p.exists() and (CONFIG_DIR / p.name).exists():
            p = CONFIG_DIR / p.name
        try:
            if args.dry_run:
                cfg = load_config(p, args.set, check=False)
                try:
                    from src.experiments.config import validate
                    validate(cfg)
                except ConfigError as e:
                    print(f'# NOTE: {e}'.replace('\n', '\n# '))
            else:
                cfg = load_config(p, args.set)
        except ConfigError as e:
            print(f'\n[{path}] {e}', file=sys.stderr)
            status = 2
            if args.keep_going:
                continue
            return status
        if args.output_dir:
            cfg['output_dir'] = args.output_dir

        if args.dry_run:
            print(f'# {path} -> {run_name(cfg)}')
            print(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith('_')},
                                 sort_keys=False, default_flow_style=None))
            continue

        from src.experiments.runner import Runner
        try:
            Runner(cfg, out_root=ROOT / cfg.get('output_dir', 'results')).run()
        except Exception:
            traceback.print_exc()
            status = 1
            if not args.keep_going:
                return status
    return status


if __name__ == '__main__':
    sys.exit(main())
