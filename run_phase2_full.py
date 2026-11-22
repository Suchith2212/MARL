"""
Phase 2 Full Grid: Primal-Dual Fixed-Width Baselines with ZOH & PID Lagrangian.

Features:
- Fixed-widths b in {1, 2, 4, 8}
- Budgets B in {8, 16, 32, 64} bits/step
- 5 seeds: [42, 43, 44, 45, 46]
- Budget: 2,000,000 steps per run (matching screening budget)
- Zero-Order Hold (ZOH) memory buffer (last_msg, tau)
- PID Lagrangian controller (K_p=0.005, alpha_I=0.01) with 50k steps warmup
- Integrate-and-Fire deterministic gating during evaluation
- Compares:
  1. Learned Primal-Dual gating
  2. Periodic gating baseline at matched cost
  3. Random gating baseline at matched cost
- Metrics:
  - Signed cost error: (bar_C - B) / B per cell
  - Per-seed violation rate nu_s(B) and range [min_nu, max_nu] across seeds
  - Wilson 95% CI (descriptive)
  - RQ2 criterion: mean cost within [-10%, 0%] of B (or within 10% not above it)
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.envs.vec_tracking_env import BatchedTrackingEnv
from src.algos.primal_dual_zoh import ZOHMAPPOTrainer

STAGE = 'dynamic'
R_MIN = -15.95
LEAD_R_MAX = -2.14

BUDGET_STEPS  = 2_000_000 # 2M steps
B_ENV         = 64
T_ROLLOUT     = 25
STEPS_PER_UPD = B_ENV * T_ROLLOUT # 1,600
TOTAL_UPDATES = (BUDGET_STEPS + STEPS_PER_UPD - 1) // STEPS_PER_UPD
WARMUP_UPDATES = 32 # 50k steps warmup

WIDTHS  = [1, 2, 4, 8]
BUDGETS = [8, 16, 32, 64]
SEEDS   = [42, 43, 44, 45, 46]

RESULTS_DIR = 'results/phase2_full'
FIG_DIR     = 'results/figures'
CKPT_DIR    = 'checkpoints/phase2_full'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


def wilson_interval(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.959963984540054
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) / n) + ((z**2) / (4 * n**2)))) / denom
    return float(p), max(0.0, float(center - margin)), min(1.0, float(center + margin))


def evaluate_policy(trainer, budget, num_eval=1000, seed=99999):
    env = BatchedTrackingEnv(num_envs=num_eval, max_cycles=T_ROLLOUT,
                             stage=STAGE, full_obs=False)
    obs, state = env.reset(seed=seed)
    last_msg = torch.zeros(num_eval, 16)
    tau = torch.zeros(num_eval, 1)
    acc = torch.zeros(num_eval)
    
    total_returns = torch.zeros(num_eval)
    total_costs   = torch.zeros(num_eval)
    
    with torch.no_grad():
        for t in range(T_ROLLOUT):
            t_est, g, _, _, _, sc, nl_msg, n_tau, n_acc = trainer.act(
                obs, last_msg, tau, t, acc=acc, deterministic=True
            )
            obs, state, r, done, _ = env.step_with_target_estimates(t_est)
            total_returns += r
            total_costs   += sc
            last_msg, tau, acc = nl_msg, n_tau, n_acc
            
    ep_bar_c = total_costs / float(T_ROLLOUT)
    mean_ret = float(total_returns.mean())
    std_ret  = float(total_returns.std())
    mean_cost = float(ep_bar_c.mean())
    std_cost  = float(ep_bar_c.std())
    
    violations = (ep_bar_c > budget).float()
    viol_count = int(violations.sum().item())
    nu, ci_low, ci_high = wilson_interval(viol_count, num_eval)
    
    # Signed cost error: (bar_C - B) / B
    cost_error = (mean_cost - budget) / float(budget)
    score = (mean_ret - R_MIN) / (LEAD_R_MAX - R_MIN)
    
    return {
        'return_mean': mean_ret,
        'return_std' : std_ret,
        'cost_mean'  : mean_cost,
        'cost_std'   : std_cost,
        'cost_error' : cost_error,
        'score_lead' : score,
        'violation_count': viol_count,
        'nu': nu,
        'wilson_ci': [ci_low, ci_high]
    }


def train_single_run(args):
    mode, bw, budget, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tag = f"{mode}_b{bw}_B{budget}_seed{seed}"
    res_path = os.path.join(RESULTS_DIR, f"{tag}.json")
    
    if os.path.exists(res_path):
        try:
            with open(res_path, 'r') as f:
                rec = json.load(f)
            if 'eval' in rec:
                print(f"[{tag}] Loaded cached result (Return={rec['eval']['return_mean']:.2f}, Cost={rec['eval']['cost_mean']:.1f})", flush=True)
                return rec
        except Exception:
            pass

    env = BatchedTrackingEnv(num_envs=B_ENV, max_cycles=T_ROLLOUT,
                             stage=STAGE, full_obs=False)
    trainer = ZOHMAPPOTrainer(
        mode=mode, fixed_bw=bw, budget=budget, d=16, T=T_ROLLOUT,
        K_p=0.005, alpha_I=0.01, warmup_updates=WARMUP_UPDATES
    )

    t0 = time.time()
    for update in range(1, TOTAL_UPDATES + 1):
        obs, state = env.reset(seed=seed * 10_000 + update)
        last_msg = torch.zeros(B_ENV, 16)
        tau = torch.zeros(B_ENV, 1)

        buf_states, buf_scout = [], []
        buf_r1, buf_r2 = [], []
        buf_held_msg, buf_tau = [], []
        buf_gates, buf_gate_lp = [], []
        buf_p1, buf_p2, buf_lp1, buf_lp2 = [], [], [], []
        buf_rewards, buf_costs, buf_dones = [], [], []

        with torch.no_grad():
            for t_step in range(T_ROLLOUT):
                buf_held_msg.append(last_msg)
                buf_tau.append(tau)

                t_est, gates, gate_lp, (p1, p2), (lp1, lp2), sc, nl_msg, n_tau, _ = trainer.act(
                    obs, last_msg, tau, t_step, deterministic=False
                )
                next_obs, next_state, r, d, _ = env.step_with_target_estimates(t_est)

                buf_states.append(state)
                buf_scout.append(obs['agent_0'])
                buf_r1.append(obs['agent_1'][:, :10])
                buf_r2.append(obs['agent_2'][:, :10])
                buf_gates.append(gates)
                buf_gate_lp.append(gate_lp)
                buf_p1.append(p1); buf_p2.append(p2)
                buf_lp1.append(lp1); buf_lp2.append(lp2)
                buf_rewards.append(r)
                buf_costs.append(sc)
                buf_dones.append(d.float())

                obs, state = next_obs, next_state
                last_msg, tau = nl_msg, n_tau

        rollouts = {
            'states'     : torch.stack(buf_states),
            'scout_obs'  : torch.stack(buf_scout),
            'rover1_obs' : torch.stack(buf_r1),
            'rover2_obs' : torch.stack(buf_r2),
            'held_msg'   : torch.stack(buf_held_msg),
            'tau'        : torch.stack(buf_tau),
            'gates'      : torch.stack(buf_gates),
            'gate_lps'   : torch.stack(buf_gate_lp),
            'p1_hat'     : torch.stack(buf_p1),
            'p2_hat'     : torch.stack(buf_p2),
            'lp1'        : torch.stack(buf_lp1),
            'lp2'        : torch.stack(buf_lp2),
            'raw_rewards': torch.stack(buf_rewards),
            'step_costs' : torch.stack(buf_costs),
            'dones'      : torch.stack(buf_dones),
        }
        trainer.train_step(rollouts)

    eval_res = evaluate_policy(trainer, budget)
    elapsed = time.time() - t0
    print(f"[{tag}] Done in {elapsed:.0f}s: Ret={eval_res['return_mean']:.2f}, Cost={eval_res['cost_mean']:.1f}/{budget} (Err={eval_res['cost_error']:+.2%}), nu={eval_res['nu']:.3f}, lambda={trainer.dual_lambda:.4f}", flush=True)

    record = {
        'mode': mode,
        'bw': bw,
        'budget': budget,
        'seed': seed,
        'eval': eval_res,
        'final_lambda': float(trainer.dual_lambda)
    }
    with open(res_path, 'w') as f:
        json.dump(record, f, indent=2)

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['primal_dual', 'periodic', 'random', 'all'], default='primal_dual')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    modes = ['primal_dual'] if args.mode == 'primal_dual' else (['primal_dual', 'periodic', 'random'] if args.mode == 'all' else [args.mode])

    print("=" * 80)
    print("PHASE 2 FULL GRID: Primal-Dual Fixed-Width Baselines (2M Steps, 5 Seeds)")
    print(f"  Modes: {modes}")
    print(f"  Widths: {WIDTHS}")
    print(f"  Budgets: {BUDGETS} bits/step")
    print(f"  Seeds: {SEEDS}")
    print(f"  Budget: {BUDGET_STEPS:,} steps/run")
    print("=" * 80)

    work = [(m, bw, bgt, s) for m in modes for bw in WIDTHS for bgt in BUDGETS for s in SEEDS]
    print(f"Total jobs: {len(work)} (running with up to {args.workers} concurrent workers)")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(work))) as ex:
        results = list(ex.map(train_single_run, work))
    print(f"\nAll Phase 2 runs completed in {time.time() - t0:.1f}s")

    # Summarize results
    summary = {}
    print("\n" + "=" * 95)
    print(f"{'Mode':<12} {'b':<3} {'B':<4} {'Return (std)':<18} {'Cost (std)':<16} {'Cost Error':<12} {'nu Range':<15} {'Wilson 95% CI':<16}")
    print("=" * 95)

    for m in modes:
        for bw in WIDTHS:
            for bgt in BUDGETS:
                group = [r for r in results if r['mode'] == m and r['bw'] == bw and r['budget'] == bgt]
                rets = [r['eval']['return_mean'] for r in group]
                costs = [r['eval']['cost_mean'] for r in group]
                errors = [r['eval']['cost_error'] for r in group]
                nus = [r['eval']['nu'] for r in group]
                
                mean_ret = float(np.mean(rets))
                std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
                mean_cost = float(np.mean(costs))
                std_cost = float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0
                mean_err = float(np.mean(errors))
                nu_min = min(nus)
                nu_max = max(nus)
                
                tot_viols = sum(r['eval']['violation_count'] for r in group)
                nu_pool, ci_l, ci_h = wilson_interval(tot_viols, len(group) * 1000)
                
                key = f"{m}_b{bw}_B{bgt}"
                summary[key] = {
                    'mode': m, 'bw': bw, 'budget': bgt,
                    'return_mean': mean_ret, 'return_std': std_ret,
                    'cost_mean': mean_cost, 'cost_std': std_cost,
                    'cost_error': mean_err,
                    'nu_per_seed': nus,
                    'nu_range': [nu_min, nu_max],
                    'wilson_ci': [ci_l, ci_h]
                }
                print(f"{m:<12} {bw:<3} {bgt:<4} {mean_ret:+.2f} ({std_ret:.2f})       {mean_cost:5.1f} ({std_cost:4.1f})      {mean_err:+6.1%}     [{nu_min:.3f}, {nu_max:.3f}]  [{ci_l:.3f}, {ci_h:.3f}]")

    print("=" * 95)
    with open(os.path.join(RESULTS_DIR, 'phase2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved -> {RESULTS_DIR}/phase2_summary.json")

    # Plot Pareto envelope
    _plot_envelope(summary, modes)


def _plot_envelope(summary, modes):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Pareto Frontier
        ax = axes[0]
        colors = {1: 'steelblue', 2: 'darkorange', 4: 'forestgreen', 8: 'firebrick'}
        for bw in WIDTHS:
            costs = [summary[f"primal_dual_b{bw}_B{bgt}"]['cost_mean'] for bgt in BUDGETS]
            rets  = [summary[f"primal_dual_b{bw}_B{bgt}"]['return_mean'] for bgt in BUDGETS]
            ax.plot(costs, rets, 'o-', color=colors[bw], label=f'Learned b={bw}', lw=2, ms=6)

        # Baselines
        if 'periodic' in modes:
            for bw in [2, 4]:
                c_p = [summary[f"periodic_b{bw}_B{bgt}"]['cost_mean'] for bgt in BUDGETS]
                r_p = [summary[f"periodic_b{bw}_B{bgt}"]['return_mean'] for bgt in BUDGETS]
                ax.plot(c_p, r_p, 'x--', color=colors[bw], alpha=0.6, label=f'Periodic b={bw}')

        ax.set_xlabel('Measured Cost C̄ (bits/step)')
        ax.set_ylabel('Mean Return (Dynamic Tracking)')
        ax.set_title('Phase 2: Fixed-Width Baselines & Envelopes')
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. Signed Cost Error (C̄ - B)/B
        ax = axes[1]
        for bw in WIDTHS:
            errs = [summary[f"primal_dual_b{bw}_B{bgt}"]['cost_error'] * 100.0 for bgt in BUDGETS]
            ax.plot(BUDGETS, errs, 's-', color=colors[bw], label=f'b={bw}', lw=1.5)
        ax.axhline(0, color='black', lw=1)
        ax.axhline(-10, color='red', ls=':', label='-10% Bound')
        ax.axhline(10, color='red', ls=':', label='+10% Bound')
        ax.set_xlabel('Bit Budget B (bits/step)')
        ax.set_ylabel('Signed Cost Error (C̄ - B)/B (%)')
        ax.set_title('RQ2 Reliability: Cost Error per Budget')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'phase2_envelope_results.png'), dpi=300)
        plt.close()
        print(f"Plot saved -> {FIG_DIR}/phase2_envelope_results.png")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()
