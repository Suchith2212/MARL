import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Setup style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

os.makedirs("results/figures", exist_ok=True)

# 1. Load Phase 1 data
phase1_file = "results/phase1/phase1_results.jsonl"
phase1_data = defaultdict(list)
if os.path.exists(phase1_file):
    with open(phase1_file, "r") as f:
        for line in f:
            rec = json.loads(line)
            phase1_data[rec['mode']].append(rec)

# 2. Load Phase 2 data
phase2_file = "results/phase2/phase2_results.jsonl"
phase2_data = defaultdict(list)
if os.path.exists(phase2_file):
    with open(phase2_file, "r") as f:
        for line in f:
            rec = json.loads(line)
            phase2_data[rec['budget_target_B']].append(rec)

# --- FIGURE 1: Return vs Measured Payload Cost Frontier ---
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

# Plot No-comm
if 'no_comm' in phase1_data:
    no_rets = [r['mean_return'] for r in phase1_data['no_comm']]
    ax.scatter([0.0], [np.mean(no_rets)], color='#475569', s=120, zorder=5, marker='s', label=f'No Communication ({np.mean(no_rets):.1f})')
    ax.errorbar([0.0], [np.mean(no_rets)], yerr=[np.std(no_rets)], fmt='none', ecolor='#475569', capsize=5, capthick=1.5)

# Plot Fixed Precision Screens (Full-Comm b in {1, 2, 4, 8})
fp_b_list = [1, 2, 4, 8]
fp_costs = []
fp_rets = []
fp_errs = []
for b in fp_b_list:
    tag = f'full_comm_b{b}'
    if tag in phase1_data:
        b_rets = [r['mean_return'] for r in phase1_data[tag]]
        b_costs = [r['mean_c_bar'] for r in phase1_data[tag]]
        fp_costs.append(np.mean(b_costs))
        fp_rets.append(np.mean(b_rets))
        fp_errs.append(np.std(b_rets))

if fp_costs:
    ax.errorbar(fp_costs, fp_rets, yerr=fp_errs, fmt='-o', color='#0284c7', linewidth=2, markersize=8, capsize=5, capthick=1.5, label='Fixed-Width Full-Comm ($b \\in \\{1, 2, 4, 8\\}$)')
    for b, c, r in zip(fp_b_list, fp_costs, fp_rets):
        ax.annotate(f'$b={b}$', (c, r), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color='#0369a1')

# Plot Phase 2 Primal-Dual Learned Gating Frontier
pd_budgets = sorted(phase2_data.keys())
pd_costs = []
pd_rets = []
pd_errs = []
for B in pd_budgets:
    b_records = phase2_data[B]
    b_rets = [r['mean_return'] for r in b_records]
    b_costs = [r['mean_c_bar'] for r in b_records]
    pd_costs.append(np.mean(b_costs))
    pd_rets.append(np.mean(b_rets))
    pd_errs.append(np.std(b_rets))

ax.errorbar(pd_costs, pd_rets, yerr=pd_errs, fmt='-s', color='#16a34a', linewidth=2.2, markersize=8, capsize=5, capthick=1.5, label='Primal-Dual Learned Gating (Fixed $b=4$)')
for B, c, r in zip(pd_budgets, pd_costs, pd_rets):
    ax.annotate(f'$B={int(B)}$', (c, r), textcoords="offset points", xytext=(0, -16), ha='center', fontsize=9, fontweight='bold', color='#15803d')

ax.set_xlabel('Measured Plain Payload Cost $\\bar{C}$ (bits/step)')
ax.set_ylabel('Mean Team Return')
ax.set_title('Figure 1: Return–Cost Frontier at Matched Measured Payload (RQ1/RQ2)')
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
plt.tight_layout()
fig1_path = "results/figures/fig1_return_cost_frontier.png"
fig.savefig(fig1_path)
plt.close(fig)
print(f"Saved: {fig1_path}")

# --- FIGURE 2: Constraint Satisfaction (Realized C vs Target B) ---
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
pd_c_disc = [np.mean([r['mean_c_disc'] for r in phase2_data[B]]) for B in pd_budgets]
pd_c_err = [np.std([r['mean_c_disc'] for r in phase2_data[B]]) for B in pd_budgets]

# Ideal equality line
max_val = max(max(pd_budgets), max(pd_c_disc)) * 1.15
ax.plot([0, max_val], [0, max_val], '--', color='#94a3b8', linewidth=1.8, label='Ideal Budget Equality ($C = B$)')
ax.errorbar(pd_budgets, pd_c_disc, yerr=pd_c_err, fmt='-o', color='#ea580c', linewidth=2, markersize=8, capsize=5, capthick=1.5, label='Primal-Dual Realized Cost $C(\\psi)$')

ax.set_xlabel('Target Budget $B$ (discounted-normalised payload bits/step)')
ax.set_ylabel('Measured Realized Cost $C(\\psi)$ (bits/step)')
ax.set_title('Figure 2: Expected Budget Constraint Enforcement (RQ2)')
ax.set_xlim(0, 145)
ax.set_ylim(0, 145)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
plt.tight_layout()
fig2_path = "results/figures/fig2_budget_satisfaction.png"
fig.savefig(fig2_path)
plt.close(fig)
print(f"Saved: {fig2_path}")

# --- FIGURE 3: Episode Violation Rate nu(B) ---
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
pd_nu = [np.mean([r['violation_rate_nu'] for r in phase2_data[B]]) for B in pd_budgets]
pd_nu_err = [np.std([r['violation_rate_nu'] for r in phase2_data[B]]) for B in pd_budgets]

ax.plot(pd_budgets, pd_nu, '-^', color='#dc2626', linewidth=2.2, markersize=9, label='Empirical Violation Rate $\\nu(B) = \\Pr(C_e > B)$')
ax.errorbar(pd_budgets, pd_nu, yerr=pd_nu_err, fmt='none', ecolor='#dc2626', capsize=5, capthick=1.5)

ax.set_xlabel('Target Budget $B$ (payload bits/step)')
ax.set_ylabel('Episode Violation Probability $\\nu(B)$')
ax.set_title('Figure 3: Empirical Episode Budget Violation Rate vs Target $B$ (RQ2)')
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(pd_budgets)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
plt.tight_layout()
fig3_path = "results/figures/fig3_violation_rate.png"
fig.savefig(fig3_path)
plt.close(fig)
print(f"Saved: {fig3_path}")
