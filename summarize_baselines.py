import os
import glob
import json
import numpy as np

BASE_DIR = "results/phase2_baselines"
files = glob.glob(os.path.join(BASE_DIR, "*.json"))

data = {}
for f in files:
    with open(f, 'r') as fp:
        rec = json.load(fp)
    m = rec['mode']
    bw = rec['bw']
    bgt = rec['budget']
    s = rec['seed']
    key = (m, bw, bgt)
    if key not in data:
        data[key] = []
    data[key].append(rec['eval'])

print(f"{'Mode':<10} {'b':<3} {'B':<4} {'Return (mean ± std)':<24} {'Cost (mean ± std)':<22} {'Err':<8} {'nu'}")
print("-" * 80)

summary = {}
for (m, bw, bgt) in sorted(data.keys()):
    group = data[(m, bw, bgt)]
    rets = [x['return_mean'] for x in group]
    costs = [x['cost_mean'] for x in group]
    errs = [x['cost_error'] for x in group]
    nus = [x['nu'] for x in group]
    
    m_ret, s_ret = np.mean(rets), np.std(rets, ddof=1) if len(rets)>1 else 0.0
    m_cost, s_cost = np.mean(costs), np.std(costs, ddof=1) if len(costs)>1 else 0.0
    m_err = np.mean(errs)
    m_nu = np.mean(nus)
    
    print(f"{m:<10} {bw:<3} {bgt:<4} {m_ret:6.2f} ± {s_ret:4.2f}            {m_cost:5.1f} ± {s_cost:4.1f}            {m_err:+.1%}   {m_nu:.3f}")
    
    summary[f"{m}_b{bw}_B{bgt}"] = {
        'mode': m,
        'bw': bw,
        'budget': bgt,
        'return_mean': float(m_ret),
        'return_std': float(s_ret),
        'cost_mean': float(m_cost),
        'cost_std': float(s_cost),
        'cost_error': float(m_err),
        'nu_mean': float(m_nu),
        'per_seed_returns': [float(r) for r in rets],
        'per_seed_costs': [float(c) for c in costs]
    }

with open(os.path.join(BASE_DIR, "matched_baselines_summary.json"), 'w') as fp:
    json.dump(summary, fp, indent=2)
print("\nSaved summary -> results/phase2_baselines/matched_baselines_summary.json")
