"""
Phase 2 Pilot: Primal-Dual Gating with Fixed-Width Precision.

- Fixed-widths: b in {1, 2, 4, 8}
- Budgets: B in {8, 16, 32, 64} bits/step
- 3 seeds: [42, 43, 44]
- Stage: dynamic continuous tracking
- Logs lambda and bar_C over training
- Evaluates measured cost bar_C, return, score, and violation rate nu(B) with Wilson score intervals
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.envs.vec_tracking_env import BatchedTrackingEnv
from src.algos.primal_dual_mappo import PrimalDualGatedMAPPOTrainer

STAGE = 'dynamic'
R_MIN = -15.95
R_MAX = -7.68
LEAD_R_MAX = -2.14

BUDGET_STEPS = 500_000   # 500k steps for pilot
B_ENV        = 64
T_ROLLOUT    = 25
STEPS_PER_UPD = B_ENV * T_ROLLOUT # 1,600
TOTAL_UPDATES = (BUDGET_STEPS + STEPS_PER_UPD - 1) // STEPS_PER_UPD

WIDTHS  = [1, 2, 4, 8]
BUDGETS = [8, 16, 32, 64]
SEEDS   = [42, 43, 44]

RESULTS_DIR = 'results/phase2'
FIG_DIR     = 'results/figures'
CKPT_DIR    = 'checkpoints/phase2'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


def wilson_score_interval(k, n, confidence=0.95):
    """Computes Wilson score interval for binomial proportion k/n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.959963984540054 # 95% confidence
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) / n) + ((z**2) / (4 * n**2)))) / denom
    ci_low = max(0.0, float(center - margin))
    ci_high = min(1.0, float(center + margin))
    return float(p), ci_low, ci_high


def evaluate_policy(trainer, budget, num_eval=1000, seed=99999):
    """
    Evaluates policy over num_eval episodes:
    Measures episode return, episode average cost bar_C, and budget violation rate nu(B).
    """
    env = BatchedTrackingEnv(num_envs=num_eval, max_cycles=T_ROLLOUT,
                             stage=STAGE, full_obs=False)
    obs, state = env.reset(seed=seed)
    total_returns = torch.zeros(num_eval)
    total_costs   = torch.zeros(num_eval)
    
    with torch.no_grad():
        for _ in range(T_ROLLOUT):
            t_est, gates, _, _, _, step_costs = trainer.act(obs, deterministic=True)
            obs, state, r, done, _ = env.step_with_target_estimates(t_est)
            total_returns += r
            total_costs   += step_costs
            
    # Episode-averaged cost per step
    ep_bar_c = total_costs / float(T_ROLLOUT)
    
    mean_ret = float(total_returns.mean())
    std_ret  = float(total_returns.std())
    mean_cost = float(ep_bar_c.mean())
    std_cost  = float(ep_bar_c.std())
    
    # Violation: bar_C > budget
    violations = (ep_bar_c > budget).float()
    viol_count = int(violations.sum().item())
    nu, ci_low, ci_high = wilson_score_interval(viol_count, num_eval)
    
    # Scores
    score_naive = (mean_ret - R_MIN) / (R_MAX - R_MIN)
    score_lead  = (mean_ret - R_MIN) / (LEAD_R_MAX - R_MIN)
    
    return {
        'return_mean': mean_ret,
        'return_std' : std_ret,
        'cost_mean'  : mean_cost,
        'cost_std'   : std_cost,
        'score_naive': score_naive,
        'score_lead' : score_lead,
        'violation_count': viol_count,
        'nu_b': nu,
        'nu_b_ci_95': [ci_low, ci_high]
    }


def train_single_run(args):
    bw, budget, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tag = f"primal_dual_b{bw}_B{budget}_seed{seed}"
    res_path = os.path.join(RESULTS_DIR, f"{tag}.json")
    
    # Check cache
    if os.path.exists(res_path):
        try:
            with open(res_path, 'r') as f:
                rec = json.load(f)
            if 'eval' in rec:
                print(f"[{tag}] Cached result loaded (return={rec['eval']['return_mean']:.2f}, cost={rec['eval']['cost_mean']:.1f})", flush=True)
                return rec
        except Exception:
            pass

    env = BatchedTrackingEnv(num_envs=B_ENV, max_cycles=T_ROLLOUT,
                             stage=STAGE, full_obs=False)
    trainer = PrimalDualGatedMAPPOTrainer(fixed_bw=bw, budget=budget, d=16, T=T_ROLLOUT)

    lambda_traj = []
    cost_traj   = []
    t0 = time.time()

    for update in range(1, TOTAL_UPDATES + 1):
        obs, state = env.reset(seed=seed * 10_000 + update)

        buf_states, buf_scout = [], []
        buf_r1, buf_r2 = [], []
        buf_gates, buf_gate_lp = [], []
        buf_p1, buf_p2, buf_lp1, buf_lp2 = [], [], [], []
        buf_rewards, buf_costs, buf_dones = [], [], []

        with torch.no_grad():
            for t_step in range(T_ROLLOUT):
                t_est, gates, gate_lp, (p1, p2), (lp1, lp2), step_costs = trainer.act(obs, deterministic=False)
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
                buf_costs.append(step_costs)
                buf_dones.append(d.float())

                obs, state = next_obs, next_state

        rollouts = {
            'states'     : torch.stack(buf_states),
            'scout_obs'  : torch.stack(buf_scout),
            'rover1_obs' : torch.stack(buf_r1),
            'rover2_obs' : torch.stack(buf_r2),
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
        res = trainer.train_step(rollouts)
        lambda_traj.append(res['dual_lambda'])
        cost_traj.append(res['batch_cost'])

    elapsed = time.time() - t0
    eval_res = evaluate_policy(trainer, budget)
    print(f"[{tag}] Done in {elapsed:.0f}s: Ret={eval_res['return_mean']:.2f}, Cost={eval_res['cost_mean']:.1f}/{budget}, nu={eval_res['nu_b']:.3f} [{eval_res['nu_b_ci_95'][0]:.3f}, {eval_res['nu_b_ci_95'][1]:.3f}], lambda={trainer.dual_lambda:.4f}", flush=True)

    record = {
        'bw': bw,
        'budget': budget,
        'seed': seed,
        'eval': eval_res,
        'final_lambda': float(trainer.dual_lambda),
        'lambda_traj': [float(x) for x in lambda_traj[::10]], # subsample for compact storage
        'cost_traj': [float(x) for x in cost_traj[::10]],
    }
    with open(res_path, 'w') as f:
        json.dump(record, f, indent=2)

    return record


def main():
    print("=" * 70)
    print("PHASE 2 PILOT: Primal-Dual Gate with Fixed-Width Precision")
    print(f"  Widths: {WIDTHS}")
    print(f"  Budgets: {BUDGETS} bits/step")
    print(f"  Seeds: {SEEDS}")
    print(f"  Budget Steps: {BUDGET_STEPS:,} steps/run")
    print("=" * 70)

    work = [(bw, bgt, s) for bw in WIDTHS for bgt in BUDGETS for s in SEEDS]
    print(f"Total runs: {len(work)} (parallel execution on available cores)")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(16, len(work))) as ex:
        results = list(ex.map(train_single_run, work))
    print(f"\nAll Phase 2 pilot runs completed in {time.time() - t0:.1f}s")

    # Aggregate by (bw, budget)
    summary = {}
    print("\n" + "=" * 80)
    print(f"{'Width':<6} {'Budget':<8} {'Return':<12} {'Cost (bits/step)':<18} {'Violation nu(B)':<20} {'Mean lambda':<12}")
    print("=" * 80)
    
    for bw in WIDTHS:
        for bgt in BUDGETS:
            group = [r for r in results if r['bw'] == bw and r['budget'] == bgt]
            rets = [r['eval']['return_mean'] for r in group]
            costs = [r['eval']['cost_mean'] for r in group]
            nus = [r['eval']['nu_b'] for r in group]
            lambdas = [r['final_lambda'] for r in group]

            mean_ret = float(np.mean(rets))
            mean_cost = float(np.mean(costs))
            mean_nu = float(np.mean(nus))
            mean_lam = float(np.mean(lambdas))
            
            # Wilson interval for total violations across all 3 seeds (3000 episodes)
            tot_viols = sum(r['eval']['violation_count'] for r in group)
            nu_pool, ci_low, ci_high = wilson_score_interval(tot_viols, len(group) * 1000)

            key = f"b{bw}_B{bgt}"
            summary[key] = {
                'bw': bw,
                'budget': bgt,
                'mean_return': mean_ret,
                'std_return': float(np.std(rets, ddof=1)),
                'mean_cost': mean_cost,
                'std_cost': float(np.std(costs, ddof=1)),
                'pooled_nu': nu_pool,
                'pooled_nu_ci': [ci_low, ci_high],
                'mean_lambda': mean_lam
            }
            print(f"b={bw:<4} B={bgt:<6} {mean_ret:+.3f}        {mean_cost:5.1f} (std {np.std(costs):3.1f})    {nu_pool:.3f} [{ci_low:.3f}, {ci_high:.3f}]   {mean_lam:.4f}")
    print("=" * 80)

    # Save summary
    with open(os.path.join(RESULTS_DIR, 'phase2_pilot_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved -> {RESULTS_DIR}/phase2_pilot_summary.json")

    # Generate plots
    _plot_phase2(results, summary)


def _plot_phase2(results, summary):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Pareto Frontier: Return vs. Measured Cost
        ax = axes[0]
        colors = {1: 'steelblue', 2: 'darkorange', 4: 'forestgreen', 8: 'firebrick'}
        for bw in WIDTHS:
            bw_costs = [summary[f"b{bw}_B{bgt}"]['mean_cost'] for bgt in BUDGETS]
            bw_rets  = [summary[f"b{bw}_B{bgt}"]['mean_return'] for bgt in BUDGETS]
            ax.plot(bw_costs, bw_rets, 'o-', color=colors[bw], label=f'b={bw}', lw=2, ms=7)
        ax.set_xlabel('Measured Cost C̄ (bits/step)')
        ax.set_ylabel('Mean Episode Return')
        ax.set_title('Phase 2 Pilot: Return vs. Measured Cost')
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. Budget Satisfaction: Measured Cost vs. Budget
        ax = axes[1]
        for bw in WIDTHS:
            bw_costs = [summary[f"b{bw}_B{bgt}"]['mean_cost'] for bgt in BUDGETS]
            ax.plot(BUDGETS, bw_costs, 's--', color=colors[bw], label=f'b={bw}', lw=1.5)
        ax.plot([0, 70], [0, 70], 'k:', label='Cost = Budget', lw=1.5)
        ax.set_xlabel('Bit Budget B (bits/step)')
        ax.set_ylabel('Measured Cost C̄ (bits/step)')
        ax.set_title('Budget Satisfaction (C̄ <= B)')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. Violation Rate nu(B) with Wilson CIs
        ax = axes[2]
        for bw in WIDTHS:
            nus = [summary[f"b{bw}_B{bgt}"]['pooled_nu'] for bgt in BUDGETS]
            err_low = [summary[f"b{bw}_B{bgt}"]['pooled_nu'] - summary[f"b{bw}_B{bgt}"]['pooled_nu_ci'][0] for bgt in BUDGETS]
            err_high = [summary[f"b{bw}_B{bgt}"]['pooled_nu_ci'][1] - summary[f"b{bw}_B{bgt}"]['pooled_nu_ci'][0] for bgt in BUDGETS]
            ax.errorbar(BUDGETS, nus, yerr=[err_low, err_high], fmt='o-', color=colors[bw], capsize=4, label=f'b={bw}')
        ax.axhline(0.05, color='red', ls=':', label='5% Target')
        ax.set_xlabel('Bit Budget B (bits/step)')
        ax.set_ylabel('Violation Rate ν(B)')
        ax.set_title('Budget Violation Rate ν(B) (Wilson 95% CI)')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'phase2_pilot_results.png'), dpi=300)
        plt.close()
        print(f"Plot saved -> {FIG_DIR}/phase2_pilot_results.png")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()
