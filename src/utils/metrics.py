import numpy as np
import scipy.stats as stats
import json
import os

def compute_paired_ci_95(diffs):
    """
    Computes sample mean and 95% confidence interval for paired differences
    using the Student-t distribution (required for >= 5 seeds).
    """
    diffs = np.array(diffs, dtype=np.float64)
    n = len(diffs)
    if n < 2:
        return float(np.mean(diffs)), 0.0, float(np.mean(diffs)), float(np.mean(diffs))
    
    mean_diff = float(np.mean(diffs))
    sem = float(stats.sem(diffs))
    # t critical value for 95% two-sided CI with df = n - 1
    t_val = float(stats.t.ppf(0.975, df=n - 1))
    margin = t_val * sem
    ci_lower = mean_diff - margin
    ci_upper = mean_diff + margin
    return mean_diff, margin, ci_lower, ci_upper


def compute_overlap_auc(frontier_a, frontier_b):
    """
    Calculates overlap AUC strictly on [c_min, c_max] where c_min is max(min_a, min_b)
    and c_max is min(max_a, max_b).
    Each frontier is a list of tuples: (measured_cost, return_val)
    """
    costs_a, rets_a = zip(*sorted(frontier_a, key=lambda x: x[0]))
    costs_b, rets_b = zip(*sorted(frontier_b, key=lambda x: x[0]))
    
    c_min = max(min(costs_a), min(costs_b))
    c_max = min(max(costs_a), max(costs_b))
    
    if c_max <= c_min:
        return None, c_min, c_max
    
    grid = np.linspace(c_min, c_max, 100)
    interp_a = np.interp(grid, costs_a, rets_a)
    interp_b = np.interp(grid, costs_b, rets_b)
    
    auc_a = float(np.trapezoid(interp_a, grid) / (c_max - c_min))
    auc_b = float(np.trapezoid(interp_b, grid) / (c_max - c_min))
    
    return {'auc_a': auc_a, 'auc_b': auc_b, 'c_min': float(c_min), 'c_max': float(c_max)}


def save_jsonl(records, filepath=None):
    if isinstance(records, str) and filepath is not None and not isinstance(filepath, str):
        # Allow swapped arguments
        records, filepath = filepath, records
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
