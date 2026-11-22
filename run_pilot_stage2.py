"""
Stage 2 Gate 0 Pilot: Dynamic (time-varying) continuous tracking.
Full-observation policy must reach >= 50% of gap from heuristic to oracle.

Frozen endpoints (1000 eval episodes, seed 12345):
  R_MIN (Scripted Heuristic): -15.95
  R_MAX (Scripted Oracle)   : -7.68
  Gate 0 threshold (score 0.50): -11.82
"""
import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.envs.vec_tracking_env import BatchedTrackingEnv
from src.algos.vec_mappo import VecMAPPOTrainer

STAGE     = 'dynamic'
R_MIN     = -15.95   # Scripted No-Info Heuristic (frozen)
R_MAX     = -7.68    # Scripted Oracle           (frozen)
MILESTONES = [200_000, 500_000, 1_000_000]
SEEDS      = [42, 43, 44]
GATE0_SCORE_THRESHOLD = 0.50

B = 64
T = 25
STEPS_PER_UPDATE = B * T    # 1,600 env steps per rollout update
MAX_BUDGET       = max(MILESTONES)
TOTAL_UPDATES    = (MAX_BUDGET + STEPS_PER_UPDATE - 1) // STEPS_PER_UPDATE


def evaluate_test_suite(trainer, num_eval=1000, seed=99999):
    env = BatchedTrackingEnv(num_envs=num_eval, max_cycles=T,
                              stage=STAGE, full_obs=True)
    obs, state = env.reset(seed=seed)
    total_r = torch.zeros(num_eval)
    with torch.no_grad():
        for _ in range(T):
            t_est, _, _ = trainer.act(obs, deterministic=True)
            obs, state, r, done, _ = env.step_with_target_estimates(t_est)
            total_r += r
    return float(total_r.mean()), float(total_r.std())


def train_seed(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env     = BatchedTrackingEnv(num_envs=B, max_cycles=T,
                                  stage=STAGE, full_obs=True)
    trainer = VecMAPPOTrainer(mode='full_obs')

    learning_curve = []
    eval_records   = {}
    t0 = time.time()

    for update in range(1, TOTAL_UPDATES + 1):
        obs, state = env.reset(seed=seed * 10_000 + update)

        buf_states = []
        buf_scout_obs = buf_rover1_obs = buf_rover2_obs = []
        buf_scout_obs, buf_rover1_obs, buf_rover2_obs = [], [], []
        buf_p1, buf_p2, buf_lp1, buf_lp2 = [], [], [], []
        buf_rewards, buf_dones = [], []

        with torch.no_grad():
            for t in range(T):
                t_est, (p1, p2), (lp1, lp2) = trainer.act(obs, deterministic=False)
                next_obs, next_state, reward, done, _ = env.step_with_target_estimates(t_est)

                buf_states.append(state)
                buf_scout_obs.append(obs['agent_0'])
                buf_rover1_obs.append(obs['agent_1'][:, 0:10])
                buf_rover2_obs.append(obs['agent_2'][:, 0:10])
                buf_p1.append(p1); buf_p2.append(p2)
                buf_lp1.append(lp1); buf_lp2.append(lp2)
                buf_rewards.append(reward)
                buf_dones.append(done.float())

                obs, state = next_obs, next_state

        rollouts = {
            'states'     : torch.stack(buf_states),
            'scout_obs'  : torch.stack(buf_scout_obs),
            'rover1_obs' : torch.stack(buf_rover1_obs),
            'rover2_obs' : torch.stack(buf_rover2_obs),
            'p1_hat'     : torch.stack(buf_p1),
            'p2_hat'     : torch.stack(buf_p2),
            'lp1'        : torch.stack(buf_lp1),
            'lp2'        : torch.stack(buf_lp2),
            'rewards'    : torch.stack(buf_rewards),
            'dones'      : torch.stack(buf_dones),
        }
        trainer.train_step(rollouts)

        curr_steps = update * STEPS_PER_UPDATE

        # --- milestone evaluation ---
        for m in MILESTONES:
            if (update - 1) * STEPS_PER_UPDATE < m <= curr_steps:
                mean_r, std_r = evaluate_test_suite(trainer)
                score = (mean_r - R_MIN) / (R_MAX - R_MIN)
                eval_records[m] = {
                    'return': mean_r, 'std': std_r,
                    'score': score, 'actual_step': curr_steps,
                }
                elapsed = time.time() - t0
                print(f"  [Seed {seed} | {m:,} steps] ({elapsed:.0f}s) "
                      f"Return={mean_r:.2f}±{std_r:.2f}  Score={score:.3f}")

        # --- periodic curve logging every 25 updates (~40k steps) ---
        if update % 25 == 0:
            mean_r, _ = evaluate_test_suite(trainer, num_eval=200, seed=update)
            learning_curve.append((curr_steps, mean_r))

    # checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(trainer, f'checkpoints/full_obs_stage2_pilot_seed_{seed}.pt')

    record = {
        'config'        : 'full_obs_stage2_pilot',
        'stage'         : STAGE,
        'seed'          : seed,
        'R_MIN'         : R_MIN,
        'R_MAX'         : R_MAX,
        'budgets'       : {str(k): v for k, v in eval_records.items()},
        'learning_curve': learning_curve,
    }
    os.makedirs('results/pilot', exist_ok=True)
    with open(f'results/pilot/stage2_full_obs_seed_{seed}.json', 'w') as f:
        json.dump(record, f, indent=2)
    return record


def main():
    print('=' * 70)
    print('STAGE 2 GATE 0 PILOT — Dynamic Continuous Tracking')
    print(f'  R_MIN (heuristic): {R_MIN:.2f}')
    print(f'  R_MAX (oracle)   : {R_MAX:.2f}')
    print(f'  Gate 0 threshold  : score >= {GATE0_SCORE_THRESHOLD} '
          f'(return >= {R_MIN + GATE0_SCORE_THRESHOLD*(R_MAX-R_MIN):.2f})')
    print(f'  Seeds: {SEEDS}   Milestones: {MILESTONES}')
    print('=' * 70)

    t0_all = time.time()
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as ex:
        results = list(ex.map(train_seed, SEEDS))
    print(f'\nAll pilot seeds done in {time.time()-t0_all:.0f}s')

    # --- Gate 0 summary ---
    print('\n' + '=' * 70)
    print('GATE 0 SUMMARY (Stage 2 — Dynamic)')
    print('=' * 70)
    for m in MILESTONES:
        scores  = [r['budgets'][str(m)]['score']  for r in results]
        returns = [r['budgets'][str(m)]['return'] for r in results]
        mean_sc = float(np.mean(scores))
        passed  = mean_sc >= GATE0_SCORE_THRESHOLD
        print(f'{m:>10,} steps | mean return {np.mean(returns):+.2f} | '
              f'scores {[round(s,3) for s in scores]} | '
              f'mean score {mean_sc:.3f} | '
              f'{"PASSED" if passed else "NOT PASSED"}')
    print('=' * 70)

    # --- plot ---
    os.makedirs('results/figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    for r in results:
        xs = [pt[0] for pt in r['learning_curve']]
        ys = [pt[1] for pt in r['learning_curve']]
        plt.plot(xs, ys, label=f"Seed {r['seed']}")
    plt.axhline(R_MAX, color='darkgreen',  ls='-',  lw=1.5, label=f'Oracle ({R_MAX:.2f})')
    plt.axhline(R_MIN, color='purple',     ls='--', lw=1.5, label=f'Heuristic ({R_MIN:.2f})')
    thr = R_MIN + GATE0_SCORE_THRESHOLD * (R_MAX - R_MIN)
    plt.axhline(thr,  color='red',         ls=':',  lw=1.5, label=f'Gate 0 threshold ({thr:.2f})')
    plt.xlabel('Environment steps')
    plt.ylabel('Episode return (test)')
    plt.title('Stage 2 Gate 0 Pilot: Full-Obs on Dynamic Tracking (3 seeds)')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig('results/figures/pilot_stage2_learning_curves.png', dpi=300)
    plt.close()
    print('Plot saved → results/figures/pilot_stage2_learning_curves.png')


if __name__ == '__main__':
    main()
