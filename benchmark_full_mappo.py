import time
import torch
import numpy as np

from src.envs.vec_tracking_env import BatchedTrackingEnv
from src.algos.vec_mappo import VecMAPPOTrainer

def benchmark_full_loop():
    torch.set_num_threads(1)
    B = 64
    T = 25
    env = BatchedTrackingEnv(num_envs=B, max_cycles=T, stage='static')
    trainer = VecMAPPOTrainer(mode='full_obs')
    
    # Measure 50 updates = 50 * 25 * 64 = 80,000 steps
    n_updates = 30
    total_steps = n_updates * T * B
    
    t0 = time.time()
    for update in range(n_updates):
        obs, state = env.reset()
        
        # Buffers
        buf_states = []
        buf_r1_in = []
        buf_r2_in = []
        buf_p1 = []
        buf_p2 = []
        buf_lp1 = []
        buf_lp2 = []
        buf_rewards = []
        buf_dones = []
        
        with torch.no_grad():
            for t in range(T):
                target_est, (r1_in, r2_in), (p1, p2), (lp1, lp2) = trainer.act(obs, deterministic=False)
                next_obs, next_state, reward, done, _ = env.step_with_target_estimates(target_est)
                
                buf_states.append(state)
                buf_r1_in.append(r1_in.detach())
                buf_r2_in.append(r2_in.detach())
                buf_p1.append(p1.detach())
                buf_p2.append(p2.detach())
                buf_lp1.append(lp1.detach())
                buf_lp2.append(lp2.detach())
                buf_rewards.append(reward.detach())
                buf_dones.append(done.float().detach())
                
                obs = next_obs
                state = next_state
            
        rollouts = {
            'states': torch.stack(buf_states),
            'r1_in': torch.stack(buf_r1_in),
            'r2_in': torch.stack(buf_r2_in),
            'p1_hat': torch.stack(buf_p1),
            'p2_hat': torch.stack(buf_p2),
            'lp1': torch.stack(buf_lp1),
            'lp2': torch.stack(buf_lp2),
            'rewards': torch.stack(buf_rewards),
            'dones': torch.stack(buf_dones)
        }
        trainer.train_step(rollouts)
        
    elapsed = time.time() - t0
    sps = total_steps / elapsed
    print(f"Full MAPPO Loop Throughput: {sps:,.0f} steps/second! ({total_steps:,} env steps in {elapsed:.2f}s)")
    print(f"Time required for 1,000,000 environment steps: {1_000_000 / sps:.1f} seconds (~{1_000_000 / sps / 60:.1f} minutes per seed)")

if __name__ == '__main__':
    benchmark_full_loop()
