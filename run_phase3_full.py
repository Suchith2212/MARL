"""
Run Phase 3: Learned Precision Gating with ZOH Memory and PID Lagrangian Controller.

Experiments:
- Budgets: [8, 16, 32, 64] bits/step
- Seeds: [42, 43, 44, 45, 46]
- Total runs: 20
- Budget steps: 2,000,000 steps per run (matching Phase 1 & Phase 2)
- Replaying sampled gates and bit-widths, STE for quantiser rounding, greedy discrete choices at test time.
- Pre-registered primary hypothesis test: paired-seed AUC difference vs fixed-width envelope.
"""
import os
import sys
import time
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
import torch
import numpy as np

from src.envs.vec_tracking_env import BatchedTrackingEnv
from src.algos.learned_precision_zoh import LearnedPrecisionMAPPOTrainer, WIDTH_VALS

BUDGETS = [8, 16, 32, 64]
SEEDS = [42, 43, 44, 45, 46]

B_ENV = 32
T_ROLLOUT = 25
BUDGET_STEPS = 2_000_000
TOTAL_UPDATES = BUDGET_STEPS // (B_ENV * T_ROLLOUT) # 2500 updates
WARMUP_UPDATES = 25
STAGE = "dynamic"

RESULTS_DIR = "results/phase3_full"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

R_MIN = -16.0
LEAD_R_MAX = -2.138


def wilson_interval(k, n, conf=0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return float(p), float(max(0.0, center - margin)), float(min(1.0, center + margin))


def evaluate_policy(trainer, budget, num_eval=200):
    eval_env = BatchedTrackingEnv(num_envs=num_eval, max_cycles=T_ROLLOUT,
                                  stage=STAGE, full_obs=False)
    obs, state = eval_env.reset(seed=999)
    last_msg = torch.zeros(num_eval, 16)
    tau = torch.zeros(num_eval, 1)
    acc = torch.zeros(num_eval)

    ep_rewards = torch.zeros(num_eval)
    ep_costs = torch.zeros(num_eval)
    width_usage = {w: 0 for w in WIDTH_VALS}
    total_active_gates = 0

    with torch.no_grad():
        for t in range(T_ROLLOUT):
            (t_est, gates, _, widx, _, widths,
             _, _, sc, nl_msg, n_tau, acc) = trainer.act(
                obs, last_msg, tau, t, acc=acc, deterministic=True
            )
            next_obs, next_state, r, d, _ = eval_env.step_with_target_estimates(t_est)

            ep_rewards += r
            ep_costs += sc
            
            # Track width usage when gate is open
            open_mask = (gates > 0.5)
            total_active_gates += int(open_mask.sum().item())
            if open_mask.any():
                active_widths = widths[open_mask].tolist()
                for w in active_widths:
                    width_usage[int(w)] = width_usage.get(int(w), 0) + 1

            obs, state = next_obs, next_state
            last_msg, tau = nl_msg, n_tau

    mean_ret = float(ep_rewards.mean().item())
    std_ret  = float(ep_rewards.std().item())
    
    ep_bar_c = ep_costs / float(T_ROLLOUT)
    mean_cost = float(ep_bar_c.mean().item())
    std_cost  = float(ep_bar_c.std().item())
    
    violations = (ep_bar_c > budget).float()
    viol_count = int(violations.sum().item())
    nu, ci_low, ci_high = wilson_interval(viol_count, num_eval)
    
    cost_error = (mean_cost - budget) / float(budget)
    score = (mean_ret - R_MIN) / (LEAD_R_MAX - R_MIN)

    # Normalize width usage percentages
    width_fracs = {}
    if total_active_gates > 0:
        for w in WIDTH_VALS:
            width_fracs[str(w)] = float(width_usage[w]) / float(total_active_gates)
    else:
        for w in WIDTH_VALS:
            width_fracs[str(w)] = 0.0

    return {
        'return_mean': mean_ret,
        'return_std' : std_ret,
        'cost_mean'  : mean_cost,
        'cost_std'   : std_cost,
        'cost_error' : cost_error,
        'score_lead' : score,
        'violation_count': viol_count,
        'nu': nu,
        'wilson_ci': [ci_low, ci_high],
        'width_fracs': width_fracs,
        'total_active_gates': total_active_gates
    }


def train_single_run(args):
    budget, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tag = f"learned_precision_B{budget}_seed{seed}"
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
    trainer = LearnedPrecisionMAPPOTrainer(
        budget=budget, d=16, T=T_ROLLOUT,
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
        buf_widx, buf_wlp, buf_widths = [], [], []
        buf_p1, buf_p2, buf_lp1, buf_lp2 = [], [], [], []
        buf_rewards, buf_costs, buf_dones = [], [], []

        with torch.no_grad():
            for t_step in range(T_ROLLOUT):
                buf_held_msg.append(last_msg)
                buf_tau.append(tau)

                (t_est, gates, gate_lp, widx, wlp, widths,
                 (p1, p2), (lp1, lp2), sc, nl_msg, n_tau, _) = trainer.act(
                    obs, last_msg, tau, t_step, deterministic=False
                )

                next_obs, next_state, r, d, _ = env.step_with_target_estimates(t_est)

                buf_states.append(state)
                buf_scout.append(obs['agent_0'])
                buf_r1.append(obs['agent_1'][:, :10])
                buf_r2.append(obs['agent_2'][:, :10])
                buf_gates.append(gates)
                buf_gate_lp.append(gate_lp)
                buf_widx.append(widx)
                buf_wlp.append(wlp)
                buf_widths.append(widths)
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
            'width_idx'  : torch.stack(buf_widx),
            'width_lps'  : torch.stack(buf_wlp),
            'widths'     : torch.stack(buf_widths),
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
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 3 FULL GRID: Learned Precision Primal-Dual MAPPO (2M Steps, 5 Seeds)")
    print(f"  Budgets: {BUDGETS} bits/step")
    print(f"  Seeds: {SEEDS}")
    print(f"  Budget: {BUDGET_STEPS:,} steps/run")
    print("=" * 80)

    work = [(bgt, s) for bgt in BUDGETS for s in SEEDS]
    print(f"Total jobs: {len(work)} (running with up to {args.workers} concurrent workers)")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(work))) as ex:
        results = list(ex.map(train_single_run, work))
    print(f"\nAll Phase 3 runs completed in {time.time() - t0:.1f}s")

    # Summarize results
    summary = {}
    print("\n" + "=" * 95)
    print(f"{'B':<4} {'Return (std)':<18} {'Cost (std)':<16} {'Cost Error':<12} {'nu Range':<15} {'Width Fracs (1/2/4/8)':<24}")
    print("=" * 95)

    for bgt in BUDGETS:
        group = [r for r in results if r['budget'] == bgt]
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

        # Average width fractions
        avg_w = {}
        for w in WIDTH_VALS:
            avg_w[str(w)] = float(np.mean([r['eval']['width_fracs'][str(w)] for r in group]))
        w_str = f"{avg_w['1']:.2f}/{avg_w['2']:.2f}/{avg_w['4']:.2f}/{avg_w['8']:.2f}"

        print(f"{bgt:<4} {mean_ret:6.2f} ({std_ret:4.2f})    {mean_cost:5.1f} ({std_cost:4.1f})      {mean_err:+.1%}     [{nu_min:.3f}, {nu_max:.3f}]  {w_str}")

        summary[f"learned_precision_B{bgt}"] = {
            'budget': bgt,
            'return_mean': mean_ret,
            'return_std': std_ret,
            'cost_mean': mean_cost,
            'cost_std': std_cost,
            'cost_error': mean_err,
            'nu_per_seed': nus,
            'nu_range': [nu_min, nu_max],
            'avg_width_fracs': avg_w
        }

    sum_path = os.path.join(RESULTS_DIR, "phase3_summary.json")
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved -> {sum_path}")


if __name__ == '__main__':
    main()
