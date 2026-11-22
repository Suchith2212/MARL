import time
import torch
import numpy as np
from src.envs.vec_tracking_env import BatchedTrackingEnv

def benchmark():
    torch.set_num_threads(1)
    B = 128
    T = 25
    env = BatchedTrackingEnv(num_envs=B, max_cycles=T, stage='static')
    
    # 1. Benchmark raw physics throughput
    t0 = time.time()
    n_steps = 0
    for _ in range(400): # 400 * 128 = 51,200 steps
        env.reset()
        for step in range(T):
            target_est = torch.zeros(B, 3, 2)
            _, _, r, done, _ = env.step_with_target_estimates(target_est)
            n_steps += B
    elapsed = time.time() - t0
    print(f"Simulation throughput: {n_steps / elapsed:,.0f} steps/second! (Processed {n_steps:,} steps in {elapsed:.2f}s)")
    
    # 2. Evaluate Scripted Baselines on Stage 1 (Static) over 1000 episodes
    B_eval = 1000
    env_eval = BatchedTrackingEnv(num_envs=B_eval, max_cycles=T, stage='static')
    
    # Oracle: feeds true target_pos to both rovers
    env_eval.reset(seed=12345)
    oracle_rets = torch.zeros(B_eval)
    for _ in range(T):
        target_est = torch.zeros(B_eval, 3, 2)
        target_est[:, 1] = env_eval.target_pos
        target_est[:, 2] = env_eval.target_pos
        _, _, r, _, _ = env_eval.step_with_target_estimates(target_est)
        oracle_rets += r
    print(f"Stage 1 Scripted Oracle Return: {oracle_rets.mean().item():.2f} +- {oracle_rets.std().item():.2f}")
    
    # No-Info Heuristic: feeds (0, 0) to both rovers
    env_eval.reset(seed=12345)
    heur_rets = torch.zeros(B_eval)
    for _ in range(T):
        target_est = torch.zeros(B_eval, 3, 2) # (0, 0)
        _, _, r, _, _ = env_eval.step_with_target_estimates(target_est)
        heur_rets += r
    print(f"Stage 1 Scripted Heuristic Return: {heur_rets.mean().item():.2f} +- {heur_rets.std().item():.2f}")
    
    # Random force
    env_eval.reset(seed=12345)
    rand_rets = torch.zeros(B_eval)
    for _ in range(T):
        rand_f = (torch.rand(B_eval, 3, 2) * 2.0) - 1.0
        _, _, r, _, _ = env_eval.step_physics(rand_f)
        rand_rets += r
    print(f"Stage 1 Random Force Return: {rand_rets.mean().item():.2f} +- {rand_rets.std().item():.2f}")

if __name__ == '__main__':
    benchmark()
