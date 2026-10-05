"""
Config-driven experiments: ``python run.py configs/<file>.yaml``.

- ``config``: YAML loading with ``extends`` and dotted overrides, validation.
- ``runner``: the tasks (baselines, hhh4, graph_controls, compare_diseases).
"""
from .config import load_config, ConfigError, TASKS, run_name
from .runner import Runner
