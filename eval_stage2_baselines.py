import torch
from src.envs.vec_tracking_env import BatchedTrackingEnv

def main():
    torch.set_num_threads(1)
    env_eval = BatchedTrackingEnv(num_envs=1000, max_cycles=25, stage='dynamic', full_obs=True)
    
    # Oracle: feeds true target_pos at every step
    env_eval.reset(seed=12345)
    r_oracle = torch.zeros(1000)
    for _ in range(25):
        t_est = torch.zeros(1000, 3, 2)
        t_est[:, 1] = env_eval.target_pos
        t_est[:, 2] = env_eval.target_pos
        _, _, r, _, _ = env_eval.step_with_target_estimates(t_est)
        r_oracle += r
        
    # Heuristic: feeds (0, 0)
    env_eval.reset(seed=12345)
    r_heur = torch.zeros(1000)
    for _ in range(25):
        t_est = torch.zeros(1000, 3, 2)
        _, _, r, _, _ = env_eval.step_with_target_estimates(t_est)
        r_heur += r
        
    print(f"Stage 2 (Dynamic) Scripted Oracle (R_MAX)    : {r_oracle.mean().item():.2f} +- {r_oracle.std().item():.2f}")
    print(f"Stage 2 (Dynamic) Scripted Heuristic (R_MIN) : {r_heur.mean().item():.2f} +- {r_heur.std().item():.2f}")
    gap = r_oracle.mean().item() - r_heur.mean().item()
    print(f"Gap: {gap:.2f}")
    print(f"Gate 0 Threshold (50% gap): {r_heur.mean().item() + 0.5 * gap:.2f}")

if __name__ == '__main__':
    main()
