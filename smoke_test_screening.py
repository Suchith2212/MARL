"""
Quick smoke-test for full_comm and no_comm modes on the dynamic env.
Runs 5 updates of training + one eval. Should complete in ~10s.
Run before launching full screening to catch bugs early.
"""
import sys, os, torch
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.envs.vec_tracking_env import BatchedTrackingEnv
from src.algos.vec_mappo import VecMAPPOTrainer

STAGE = 'dynamic'
B, T  = 32, 25
R_MIN, R_MAX = -15.95, -7.68

for mode, bw in [('no_comm', 0), ('full_comm', 1), ('full_comm', 8)]:
    torch.manual_seed(0)
    env     = BatchedTrackingEnv(num_envs=B, max_cycles=T, stage=STAGE, full_obs=False)
    trainer = VecMAPPOTrainer(mode=mode, fixed_bw=bw)

    for upd in range(5):
        obs, state = env.reset(seed=upd)
        bufs = {k: [] for k in ['states','scout_obs','rover1_obs','rover2_obs',
                                  'p1_hat','p2_hat','lp1','lp2','rewards','dones']}
        with torch.no_grad():
            for t in range(T):
                t_est, (p1,p2), (lp1,lp2) = trainer.act(obs, deterministic=False)
                next_obs, next_state, reward, done, _ = env.step_with_target_estimates(t_est)
                bufs['states'].append(state)
                bufs['scout_obs'].append(obs['agent_0'])
                bufs['rover1_obs'].append(obs['agent_1'][:,0:10])
                bufs['rover2_obs'].append(obs['agent_2'][:,0:10])
                bufs['p1_hat'].append(p1); bufs['p2_hat'].append(p2)
                bufs['lp1'].append(lp1); bufs['lp2'].append(lp2)
                bufs['rewards'].append(reward); bufs['dones'].append(done.float())
                obs, state = next_obs, next_state
        rollouts = {k: torch.stack(v) for k, v in bufs.items()}
        trainer.train_step(rollouts)

    # Eval
    env_eval = BatchedTrackingEnv(num_envs=500, max_cycles=T, stage=STAGE, full_obs=False)
    obs_e, _ = env_eval.reset(seed=99999)
    total_r  = torch.zeros(500)
    with torch.no_grad():
        for _ in range(T):
            t_est, _, _ = trainer.act(obs_e, deterministic=True)
            obs_e, _, r, _, _ = env_eval.step_with_target_estimates(t_est)
            total_r += r
    mean_r = float(total_r.mean())
    score  = (mean_r - R_MIN) / (R_MAX - R_MIN)
    print(f"  PASS  mode={mode} bw={bw}  return={mean_r:.2f}  score={score:.3f}  "
          f"(random init, 5 updates only — not trained)")

print("Smoke test complete.")
