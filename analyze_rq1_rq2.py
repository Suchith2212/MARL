"""
Pre-Registered Statistical Analysis for RQ1 (Learned Precision Advantage) and RQ2 (Budget Reliability).

Pre-Registered Criteria:
1. Shared Evaluation Budget Grid: B in {8, 16, 32, 64} bits/step.
2. 5 Independent Training Seeds: [42, 43, 44, 45, 46].
3. Fixed-Width Pareto Envelope per seed s:
     Env_s(C) = max_{b in {1,2,4,8}} R_{b,s}(C)
4. Primary Metric: Paired-seed AUC difference in Return vs. Measured Cost plane.
     AUC = trapz(R, C)
     Delta_AUC_s = AUC_learned,s - AUC_envelope,s
5. Minimum Effect Size: Delta_min = 5% of (Lead Oracle - Heuristic) = 0.6906 return points (0.050 normalized).
6. Decision Rule:
     Reject H0 if mean(Delta_AUC) >= Delta_min AND paired t-test p < 0.05.
     Otherwise accept H0 (Null result).
7. RQ2 Reliability Criterion:
     Reliable if mean cost in [0.90 B, 1.00 B] (within -10% to 0% of B, no overspending).
     Per-seed violation rate nu_s(B) reported with [min, max] range.
"""
import os
import glob
import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

BUDGETS = [8, 16, 32, 64]
WIDTHS = [1, 2, 4, 8]
SEEDS = [42, 43, 44, 45, 46]

R_MIN = -16.0
LEAD_R_MAX = -2.138
DELTA_MIN = 0.05 * (LEAD_R_MAX - R_MIN) # 0.6906

P2_DIR = "results/phase2_full"
P3_DIR = "results/phase3_full"
FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load_phase2_data():
    # p2_data[seed][bw][bgt] = {'return': ret, 'cost': cost, 'nu': nu}
    data = {s: {b: {} for b in WIDTHS} for s in SEEDS}
    for b in WIDTHS:
        for bgt in BUDGETS:
            for s in SEEDS:
                fpath = os.path.join(P2_DIR, f"primal_dual_b{b}_B{bgt}_seed{s}.json")
                if os.path.exists(fpath):
                    with open(fpath, 'r') as fp:
                        rec = json.load(fp)
                        ev = rec['eval']
                        data[s][b][bgt] = {
                            'return': ev['return_mean'],
                            'cost': ev['cost_mean'],
                            'cost_error': ev['cost_error'],
                            'nu': ev['nu']
                        }
    return data


def load_phase3_data():
    # p3_data[seed][bgt] = {'return': ret, 'cost': cost, 'nu': nu, 'widths': {...}}
    data = {s: {} for s in SEEDS}
    for bgt in BUDGETS:
        for s in SEEDS:
            fpath = os.path.join(P3_DIR, f"learned_precision_B{bgt}_seed{s}.json")
            if os.path.exists(fpath):
                with open(fpath, 'r') as fp:
                    rec = json.load(fp)
                    ev = rec['eval']
                    data[s][bgt] = {
                        'return': ev['return_mean'],
                        'cost': ev['cost_mean'],
                        'cost_error': ev['cost_error'],
                        'nu': ev['nu'],
                        'width_fracs': ev.get('width_fracs', {})
                    }
    return data


def compute_envelope_per_seed(p2_seed_data):
    # Collect all (cost, return) points across all fixed widths for this seed
    all_points = []
    for b in WIDTHS:
        for bgt in BUDGETS:
            if bgt in p2_seed_data[b]:
                c = p2_seed_data[b][bgt]['cost']
                r = p2_seed_data[b][bgt]['return']
                all_points.append((c, r, b, bgt))
    
    # Sort by cost
    all_points.sort(key=lambda x: x[0])
    
    # Pareto upper envelope: monotonic upper hull
    envelope_points = []
    max_r = -float('inf')
    for c, r, b, bgt in all_points:
        if r > max_r:
            envelope_points.append((c, r, b, bgt))
            max_r = r
            
    return envelope_points, all_points


def main():
    p2_data = load_phase2_data()
    p3_data = load_phase3_data()

    # Check completeness
    p3_complete = all(bgt in p3_data[s] for s in SEEDS for bgt in BUDGETS)
    if not p3_complete:
        print("Phase 3 runs are not yet fully complete. Partial analysis:")

    print("=" * 80)
    print("PRE-REGISTERED ANALYSIS: RQ1 (LEARNED PRECISION) & RQ2 (BUDGET RELIABILITY)")
    print("=" * 80)

    # 1. Fixed-Width Envelope Summary
    print("\n--- PHASE 2: FIXED-WIDTH PARETO ENVELOPE ---")
    for b in WIDTHS:
        print(f"\nWidth b={b} (Packet size = {16*b} bits):")
        print(f"  {'B':<4} {'Return (mean ± std)':<24} {'Cost (mean ± std)':<22} {'Cost Error':<12} {'nu Range':<16} {'RQ2 Reliable?'}")
        for bgt in BUDGETS:
            rets = [p2_data[s][b][bgt]['return'] for s in SEEDS if bgt in p2_data[s][b]]
            costs = [p2_data[s][b][bgt]['cost'] for s in SEEDS if bgt in p2_data[s][b]]
            errs = [p2_data[s][b][bgt]['cost_error'] for s in SEEDS if bgt in p2_data[s][b]]
            nus = [p2_data[s][b][bgt]['nu'] for s in SEEDS if bgt in p2_data[s][b]]

            if rets:
                m_ret, s_ret = np.mean(rets), np.std(rets, ddof=1) if len(rets) > 1 else 0.0
                m_cost, s_cost = np.mean(costs), np.std(costs, ddof=1) if len(costs) > 1 else 0.0
                m_err = np.mean(errs)
                nu_min, nu_max = min(nus), max(nus)
                reliable = "YES ([-10%, 0%])" if -0.10 <= m_err <= 0.00 else ("OVER (+)" if m_err > 0 else "UNDER (-)")
                print(f"  {bgt:<4} {m_ret:6.2f} ± {s_ret:4.2f}            {m_cost:5.1f} ± {s_cost:4.1f}            {m_err:+.1%}        [{nu_min:.3f}, {nu_max:.3f}]    {reliable}")

    # 2. Phase 3 Learned Precision Summary
    print("\n--- PHASE 3: LEARNED PRECISION PRIMAL-DUAL ---")
    print(f"  {'B':<4} {'Return (mean ± std)':<24} {'Cost (mean ± std)':<22} {'Cost Error':<12} {'nu Range':<16} {'Width Fracs (1/2/4/8)':<22} {'RQ2 Reliable?'}")
    for bgt in BUDGETS:
        rets = [p3_data[s][bgt]['return'] for s in SEEDS if bgt in p3_data[s]]
        costs = [p3_data[s][bgt]['cost'] for s in SEEDS if bgt in p3_data[s]]
        errs = [p3_data[s][bgt]['cost_error'] for s in SEEDS if bgt in p3_data[s]]
        nus = [p3_data[s][bgt]['nu'] for s in SEEDS if bgt in p3_data[s]]

        if rets:
            m_ret, s_ret = np.mean(rets), np.std(rets, ddof=1) if len(rets) > 1 else 0.0
            m_cost, s_cost = np.mean(costs), np.std(costs, ddof=1) if len(costs) > 1 else 0.0
            m_err = np.mean(errs)
            nu_min, nu_max = min(nus), max(nus)
            reliable = "YES ([-10%, 0%])" if -0.10 <= m_err <= 0.00 else ("OVER (+)" if m_err > 0 else "UNDER (-)")

            # Width fractions
            w_avg = {}
            for w in WIDTHS:
                w_avg[str(w)] = np.mean([p3_data[s][bgt]['width_fracs'].get(str(w), 0.0) for s in SEEDS if bgt in p3_data[s]])
            w_str = f"{w_avg['1']:.2f}/{w_avg['2']:.2f}/{w_avg['4']:.2f}/{w_avg['8']:.2f}"

            print(f"  {bgt:<4} {m_ret:6.2f} ± {s_ret:4.2f}            {m_cost:5.1f} ± {s_cost:4.1f}            {m_err:+.1%}        [{nu_min:.3f}, {nu_max:.3f}]    {w_str:<22} {reliable}")

    # 3. Paired-Seed Statistical Test for RQ1
    if p3_complete:
        print("\n" + "=" * 80)
        print("PRE-REGISTERED HYPOTHESIS TEST: RQ1 (AUC DIFFERENCE)")
        print(f"Minimum Effect Size Threshold Delta_min: {DELTA_MIN:.4f} return points (0.050 normalized)")
        print("=" * 80)

        # Integration range: shared cost span [C_min, C_max]
        # Across all seeds, find min and max measured cost
        all_costs = []
        for s in SEEDS:
            for bgt in BUDGETS:
                all_costs.append(p3_data[s][bgt]['cost'])
                for b in WIDTHS:
                    all_costs.append(p2_data[s][b][bgt]['cost'])
        c_eval_min = max(6.0, min(all_costs))
        c_eval_max = min(60.0, max(all_costs))
        cost_grid = np.linspace(c_eval_min, c_eval_max, 100)

        auc_envelope = []
        auc_learned = []
        delta_auc = []

        print(f"Shared cost integration domain: [{c_eval_min:.2f}, {c_eval_max:.2f}] bits/step\n")
        print(f"{'Seed':<6} {'AUC Envelope':<16} {'AUC Learned':<16} {'Delta AUC':<14} {'Normalized Delta'}")
        print("-" * 65)

        for s in SEEDS:
            # Envelope curve for seed s
            env_pts, _ = compute_envelope_per_seed(p2_data[s])
            env_c = [pt[0] for pt in env_pts]
            env_r = [pt[1] for pt in env_pts]
            
            # Interpolate envelope onto cost_grid
            interp_env_r = np.interp(cost_grid, env_c, env_r)
            auc_env_s = np.trapezoid(interp_env_r, cost_grid) / (c_eval_max - c_eval_min)

            # Learned curve for seed s
            lrn_pts = sorted([(p3_data[s][bgt]['cost'], p3_data[s][bgt]['return']) for bgt in BUDGETS], key=lambda x: x[0])
            lrn_c = [pt[0] for pt in lrn_pts]
            lrn_r = [pt[1] for pt in lrn_pts]
            interp_lrn_r = np.interp(cost_grid, lrn_c, lrn_r)
            auc_lrn_s = np.trapezoid(interp_lrn_r, cost_grid) / (c_eval_max - c_eval_min)

            d_auc = auc_lrn_s - auc_env_s
            d_norm = d_auc / (LEAD_R_MAX - R_MIN)

            auc_envelope.append(auc_env_s)
            auc_learned.append(auc_lrn_s)
            delta_auc.append(d_auc)

            print(f"{s:<6} {auc_env_s:12.3f}    {auc_lrn_s:12.3f}    {d_auc:+10.4f}     {d_norm:+10.4f}")

        mean_delta = float(np.mean(delta_auc))
        std_delta = float(np.std(delta_auc, ddof=1))
        se_delta = std_delta / np.sqrt(len(SEEDS))
        t_stat, p_val_two = stats.ttest_1samp(delta_auc, popmean=DELTA_MIN)
        # One-tailed test against H0: mu <= DELTA_MIN
        p_val_one = p_val_two / 2.0 if t_stat > 0 else 1.0 - (p_val_two / 2.0)

        # 95% Confidence Interval for paired difference
        ci_95 = stats.t.interval(0.95, df=len(SEEDS)-1, loc=mean_delta, scale=se_delta)

        print("-" * 65)
        print(f"Mean Delta AUC:       {mean_delta:+.4f} (std={std_delta:.4f}, SE={se_delta:.4f})")
        print(f"95% Paired-Seed CI:   [{ci_95[0]:+.4f}, {ci_95[1]:+.4f}]")
        print(f"Pre-Registered Threshold: +{DELTA_MIN:.4f}")
        print(f"t-statistic:          {t_stat:.4f} (df={len(SEEDS)-1})")
        print(f"p-value (one-tailed): {p_val_one:.5f}")

        print("\nDECISION OUTCOME:")
        if mean_delta >= DELTA_MIN and p_val_one < 0.05:
            print("  --> REJECT H0: Learned precision significantly outperforms the fixed-width Pareto envelope (Positive result).")
        else:
            print("  --> FAIL TO REJECT H0 (NULL RESULT): Learned precision does not exceed the fixed-width Pareto envelope by the pre-registered margin.")
            print(f"      (Margin observed: {mean_delta:+.4f} vs. threshold {DELTA_MIN:+.4f}, p={p_val_one:.4f})")

        # Save analysis JSON
        analysis_res = {
            'c_eval_range': [float(c_eval_min), float(c_eval_max)],
            'auc_envelope_per_seed': auc_envelope,
            'auc_learned_per_seed': auc_learned,
            'delta_auc_per_seed': delta_auc,
            'mean_delta_auc': mean_delta,
            'std_delta_auc': std_delta,
            'ci_95': [float(ci_95[0]), float(ci_95[1])],
            'delta_min': float(DELTA_MIN),
            't_statistic': float(t_stat),
            'p_value_one_tailed': float(p_val_one),
            'null_rejected': bool(mean_delta >= DELTA_MIN and p_val_one < 0.05)
        }
        with open("results/rq1_analysis.json", 'w') as fp:
            json.dump(analysis_res, fp, indent=2)

        # Generate publication plots
        plot_results(p2_data, p3_data, cost_grid)


def plot_results(p2_data, p3_data, cost_grid):
    plt.figure(figsize=(12, 5), dpi=300)

    # Subplot 1: Pareto Frontier vs Learned Precision
    plt.subplot(1, 2, 1)
    
    # Plot fixed-width baselines
    colors = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728'}
    markers = {1: 'o', 2: 's', 4: '^', 8: 'd'}

    for b in WIDTHS:
        c_means = [np.mean([p2_data[s][b][bgt]['cost'] for s in SEEDS]) for bgt in BUDGETS]
        r_means = [np.mean([p2_data[s][b][bgt]['return'] for s in SEEDS]) for bgt in BUDGETS]
        r_stds = [np.std([p2_data[s][b][bgt]['return'] for s in SEEDS], ddof=1) for bgt in BUDGETS]
        plt.errorbar(c_means, r_means, yerr=r_stds, fmt=f'-{markers[b]}', color=colors[b],
                     label=f'Fixed b={b}', capsize=3, alpha=0.7)

    # Plot learned precision
    p3_c_means = [np.mean([p3_data[s][bgt]['cost'] for s in SEEDS]) for bgt in BUDGETS]
    p3_r_means = [np.mean([p3_data[s][bgt]['return'] for s in SEEDS]) for bgt in BUDGETS]
    p3_r_stds = [np.std([p3_data[s][bgt]['return'] for s in SEEDS], ddof=1) for bgt in BUDGETS]
    plt.errorbar(p3_c_means, p3_r_means, yerr=p3_r_stds, fmt='-k*', linewidth=2.5,
                 markersize=10, label='Learned Precision (Ours)', capsize=4, zorder=5)

    # Reference lines
    plt.axhline(LEAD_R_MAX, color='purple', linestyle='--', alpha=0.6, label='Lead Oracle (-2.14)')
    plt.axhline(-16.0, color='gray', linestyle=':', alpha=0.6, label='No-Comm Heuristic (-16.00)')

    plt.xlabel('Measured Cost (bits/step)', fontsize=11)
    plt.ylabel('Episode Return', fontsize=11)
    plt.title('RQ1: Return vs. Communication Cost', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=9, loc='lower right')

    # Subplot 2: Dynamic Width Usage Distribution
    plt.subplot(1, 2, 2)
    width_matrix = np.zeros((4, 4)) # [4 budgets, 4 widths]
    for i, bgt in enumerate(BUDGETS):
        for j, w in enumerate(WIDTHS):
            width_matrix[i, j] = np.mean([p3_data[s][bgt]['width_fracs'].get(str(w), 0.0) for s in SEEDS])

    x = np.arange(len(BUDGETS))
    bottom = np.zeros(len(BUDGETS))
    for j, w in enumerate(WIDTHS):
        plt.bar(x, width_matrix[:, j], bottom=bottom, label=f'Width b={w}',
                color=colors[w], alpha=0.85, width=0.55)
        bottom += width_matrix[:, j]

    plt.xticks(x, [f'B={b}' for b in BUDGETS], fontsize=11)
    plt.ylabel('Proportion of Transmissions', fontsize=11)
    plt.title('Learned Bit-Width Allocation across Budgets', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, "rq1_rq2_learned_precision.png")
    plt.savefig(out_fig)
    print(f"\nPublication plot saved -> {out_fig}")


if __name__ == '__main__':
    main()
