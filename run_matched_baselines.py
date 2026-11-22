"""
Matched-Cost Periodic and Random Gating Baselines.

Evaluates fixed-width periodic and random gating schedules at matched budgets:
- Periodic: Transmits deterministically every P = round(16*b / B) steps.
- Random: Transmits with Bernoulli probability p = min(1.0, B / (16*b)) per step.

Rovers receive ZOH memory buffer (last_received_msg, tau).
Scout encoder and rover decoders are trained for 2,000,000 steps matching Phase 2.
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
from src.algos.primal_dual_zoh import ZOHMAPPOTrainer

BUDGETS = [8, 16, 32, 64]
WIDTHS = [1, 2, 4, 8]
SEEDS = [42, 43, 44, 45, 46]

B_ENV = 32
T_ROLLOUT = 25
BUDGET_STEPS = 2_000_000
TOTAL_UPDATES = BUDGET_STEPS // (B_ENV * T_ROLLOUT) # 2500 updates
STAGE = "dynamic"

RESULTS_DIR = "results/phase2_baselines"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

    with torch.no_grad():
        for t in range(T_ROLLOUT):
            (t_est, gates, _, (p1, p2), (lp1, lp2),
             sc, nl_msg, n_tau, acc) = trainer.act(
                obs, last_msg, tau, t, acc=acc, deterministic=True
            )
            next_obs, next_state, r, d, _ = eval_env.step_with_target_estimates(t_est)

            ep_rewards += r
            ep_costs += sc
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
        warmup_updates=0
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
    print(f"[{tag}] Done in {elapsed:.0f}s: Ret={eval_res['return_mean']:.2f}, Cost={eval_res['cost_mean']:.1f}/{budget} (Err={eval_res['cost_error']:+.2%}), nu={eval_res['nu']:.3f}", flush=True)

    record = {
        'mode': mode,
        'bw': bw,
        'budget': budget,
        'seed': seed,
        'eval': eval_res,
    }
    with open(res_path, 'w') as f:
        json.dump(record, f, indent=2)

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['periodic', 'random', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--envelope_only', action='store_true', help="Only run envelope widths (b=1 for B=8, b=2 for B=16/32, b=4 for B=32/64)")
    args = parser.parse_args()

    modes = ['periodic', 'random'] if args.mode == 'all' else [args.mode]
    
    if args.envelope_only:
        envelope_configs = [
            (1, 8),
            (2, 16),
            (2, 32),
            (4, 32),
            (4, 64)
        ]
        work = [(m, bw, bgt, s) for m in modes for bw, bgt in envelope_configs for s in SEEDS]
    else:
        work = [(m, bw, bgt, s) for m in modes for bw in WIDTHS for bgt in BUDGETS for s in SEEDS]

    print("=" * 80)
    print("MATCHED-COST BASELINES: Periodic and Random Gating (2M Steps, 5 Seeds)")
    print(f"  Modes: {modes}")
    print(f"  Total jobs: {len(work)} (running with up to {args.workers} workers)")
    print("=" * 80)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(work))) as ex:
        results = list(ex.map(train_single_run, work))
    print(f"\nAll baseline runs completed in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
