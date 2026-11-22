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

def evaluate_test_suite(trainer, stage='static', num_eval=1000, seed=99999):
    env_eval = BatchedTrackingEnv(num_envs=num_eval, max_cycles=25, stage=stage, full_obs=True)
    obs, state = env_eval.reset(seed=seed)
    total_r = torch.zeros(num_eval)
    
    with torch.no_grad():
        for t in range(25):
            target_est, _, _ = trainer.act(obs, deterministic=True)
            obs, state, r, done, _ = env_eval.step_with_target_estimates(target_est)
            total_r += r
            
    mean_r = float(total_r.mean().item())
    std_r = float(total_r.std().item())
    return mean_r, std_r

def train_pilot_seed(seed, milestones, R_MIN, R_MAX, stage='static'):
    torch.set_num_threads(1)
    B = 64
    T = 25
    steps_per_update = B * T # 1,600
    max_budget = max(milestones)
    total_updates = (max_budget + steps_per_update - 1) // steps_per_update
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = BatchedTrackingEnv(num_envs=B, max_cycles=T, stage=stage, full_obs=True)
    trainer = VecMAPPOTrainer(mode='full_obs')
    
    learning_curve = []
    eval_records = {}
    t0 = time.time()
    
    for update in range(1, total_updates + 1):
        obs, state = env.reset(seed=seed * 10000 + update)
        buf_states, buf_scout_obs, buf_rover1_obs, buf_rover2_obs = [], [], [], []
        buf_p1, buf_p2, buf_lp1, buf_lp2, buf_rewards, buf_dones = [], [], [], [], [], []
        
        with torch.no_grad():
            for t in range(T):
                t_est, (p1, p2), (lp1, lp2) = trainer.act(obs, deterministic=False)
                next_obs, next_state, reward, done, _ = env.step_with_target_estimates(t_est)
                
                buf_states.append(state)
                buf_scout_obs.append(obs['agent_0'])
                buf_rover1_obs.append(obs['agent_1'][:, 0:10])
                buf_rover2_obs.append(obs['agent_2'][:, 0:10])
                buf_p1.append(p1)
                buf_p2.append(p2)
                buf_lp1.append(lp1)
                buf_lp2.append(lp2)
                buf_rewards.append(reward)
                buf_dones.append(done.float())
                
                obs = next_obs
                state = next_state
                
        rollouts = {
            'states': torch.stack(buf_states),
            'scout_obs': torch.stack(buf_scout_obs),
            'rover1_obs': torch.stack(buf_rover1_obs),
            'rover2_obs': torch.stack(buf_rover2_obs),
            'p1_hat': torch.stack(buf_p1),
            'p2_hat': torch.stack(buf_p2),
            'lp1': torch.stack(buf_lp1),
            'lp2': torch.stack(buf_lp2),
            'rewards': torch.stack(buf_rewards),
            'dones': torch.stack(buf_dones)
        }
        trainer.train_step(rollouts)
        
        curr_steps = update * steps_per_update
        
        for m in milestones:
            if (update - 1) * steps_per_update < m <= curr_steps:
                test_r, test_std = evaluate_test_suite(trainer, stage=stage, num_eval=1000, seed=99999)
                score_unclipped = (test_r - R_MIN) / (R_MAX - R_MIN)
                eval_records[m] = {
                    'return': test_r,
                    'std': test_std,
                    'score': score_unclipped,
                    'actual_step': curr_steps
                }
                print(f"  [Seed {seed} | Milestone {m:,} steps] ({time.time()-t0:.1f}s) | Test Return: {test_r:.2f} +- {test_std:.2f} | Unclipped Score: {score_unclipped:.3f}")
            
        if update % 25 == 0:
            test_r, _ = evaluate_test_suite(trainer, stage=stage, num_eval=200, seed=update)
            learning_curve.append((curr_steps, test_r))

    # Save checkpoint
    torch.save(trainer, f"checkpoints/full_obs_pilot_seed_{seed}.pt")
    
    # Save per-run JSON
    run_record = {
        'config': 'full_obs_stage1_pilot',
        'stage': stage,
        'seed': seed,
        'budgets': {str(k): v for k, v in eval_records.items()},
        'learning_curve': learning_curve,
        'endpoints': {'R_MIN': R_MIN, 'R_MAX': R_MAX}
    }
    with open(f"results/pilot/full_obs_seed_{seed}.json", "w") as f:
        json.dump(run_record, f, indent=2)
        
    return run_record

def run_pilot():
    stage = 'static'
    milestones = [200_000, 500_000, 1_000_000]
    seeds = [42, 43, 44]
    
    print("Evaluating frozen scripted baselines on Stage 1 (Static)...")
    env_base = BatchedTrackingEnv(num_envs=1000, max_cycles=25, stage=stage)
    
    # Oracle
    env_base.reset(seed=12345)
    r_oracle = torch.zeros(1000)
    for _ in range(25):
        t_est = torch.zeros(1000, 3, 2)
        t_est[:, 1] = env_base.target_pos
        t_est[:, 2] = env_base.target_pos
        _, _, r, _, _ = env_base.step_with_target_estimates(t_est)
        r_oracle += r
    R_MAX = float(r_oracle.mean().item())
    
    # Heuristic
    env_base.reset(seed=12345)
    r_heur = torch.zeros(1000)
    for _ in range(25):
        t_est = torch.zeros(1000, 3, 2)
        _, _, r, _, _ = env_base.step_with_target_estimates(t_est)
        r_heur += r
    R_MIN = float(r_heur.mean().item())
    
    print(f"Frozen Endpoints:")
    print(f"  Scripted No-Info Heuristic (R_MIN): {R_MIN:.2f}")
    print(f"  Scripted Oracle (R_MAX)          : {R_MAX:.2f}")
    print(f"  Gap (R_MAX - R_MIN)              : {R_MAX - R_MIN:.2f}")
    print(f"  Gate 0 Target (50% gap)          : {R_MIN + 0.5 * (R_MAX - R_MIN):.2f} (Score >= 0.50)")
    
    os.makedirs("results/pilot", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"\nLaunching {len(seeds)} pilot seeds in parallel across CPU cores...")
    t0_all = time.time()
    pilot_results = []
    with ProcessPoolExecutor(max_workers=len(seeds)) as executor:
        futures = [
            executor.submit(train_pilot_seed, s, milestones, R_MIN, R_MAX, stage)
            for s in seeds
        ]
        for fut in futures:
            pilot_results.append(fut.result())
            
    print(f"All pilot runs completed in {time.time()-t0_all:.1f}s!")
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    for r in pilot_results:
        steps_axis = [pt[0] for pt in r['learning_curve']]
        rets_axis = [pt[1] for pt in r['learning_curve']]
        plt.plot(steps_axis, rets_axis, label=f"Seed {r['seed']}")
        
    plt.axhline(R_MAX, color='darkgreen', linestyle='-', label=f"Scripted Oracle ({R_MAX:.2f})")
    plt.axhline(R_MIN, color='purple', linestyle='--', label=f"Scripted Heuristic ({R_MIN:.2f})")
    plt.axhline(R_MIN + 0.5 * (R_MAX - R_MIN), color='red', linestyle=':', label=f"Gate 0 Threshold (50% gap)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Return on Test Suite")
    plt.title("Gate 0 Pilot: Full-Observation Learning Curves (Stage 1 Static)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/pilot_stage1_learning_curves.png", dpi=300)
    plt.close()
    print("Saved plot to: results/figures/pilot_stage1_learning_curves.png")
    
    # Gate 0 Assessment
    print("\n" + "=" * 75)
    print("GATE 0 (COMPETENCE CHECK) PILOT SUMMARY")
    print("=" * 75)
    for m in milestones:
        scores = [r['budgets'][str(m)]['score'] for r in pilot_results]
        rets = [r['budgets'][str(m)]['return'] for r in pilot_results]
        print(f"Budget {m:,} steps:")
        print(f"  Mean Return: {np.mean(rets):.2f} (Scores across seeds: {[round(s, 3) for s in scores]})")
        print(f"  Mean Unclipped Score: {np.mean(scores):.3f}")
        passed = np.mean(scores) >= 0.50
        print(f"  Gate 0 Status at {m:,} steps: {'PASSED (>= 0.50)' if passed else 'NOT PASSED (< 0.50)'}")
    print("=" * 75)

if __name__ == '__main__':
    run_pilot()
