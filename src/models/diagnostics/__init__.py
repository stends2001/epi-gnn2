"""
Diagnostics for probabilistic forecasts and the hhh4-style GNN.

- ``components``: tables of the endemic / epidemic / neighbourhood split.
- ``sanity``: automated checks with PASS / WARN / FAIL, to catch models that
  produce plausible-looking but meaningless forecasts.
- ``plots``: decomposition, parameter maps, seasonal curves, calibration plots.
"""
from .components import component_table, component_by_node, compare_components, components_with_target_time
from .sanity import sanity_report, print_sanity, lag_correlation
from .plots import (
    plot_decomposition, plot_component_shares, plot_node_maps, plot_seasonal_curves, plot_rate_multipliers,
    plot_calibration, plot_lag_check, plot_pred_vs_obs, plot_model_comparison,
)
