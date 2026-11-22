import torch
from src.envs.vec_tracking_env import BatchedTrackingEnv
from src.algos.primal_dual_zoh import ZOHMAPPOTrainer

B, T = 32, 25
env = BatchedTrackingEnv(B, T, stage='dynamic', full_obs=False)

for mode in ['primal_dual', 'periodic', 'random']:
    trainer = ZOHMAPPOTrainer(mode=mode, fixed_bw=4, budget=16, d=16, T=T)
    for upd in range(2):
        obs, state = env.reset(seed=upd)
        last_msg = torch.zeros(B, 16)
        tau = torch.zeros(B, 1)
        bufs = {k: [] for k in ['states','scout_obs','rover1_obs','rover2_obs','held_msg','tau',
                                  'gates','gate_lps','p1_hat','p2_hat','lp1','lp2','raw_rewards','step_costs','dones']}
        for t in range(T):
            bufs['held_msg'].append(last_msg)
            bufs['tau'].append(tau)
            
            t_est, gates, gate_lp, (p1, p2), (lp1, lp2), costs, new_last_msg, new_tau = trainer.act(obs, last_msg, tau, t, deterministic=False)
            no, ns, r, d, _ = env.step_with_target_estimates(t_est)
            
            bufs['states'].append(state)
            bufs['scout_obs'].append(obs['agent_0'])
            bufs['rover1_obs'].append(obs['agent_1'][:, :10])
            bufs['rover2_obs'].append(obs['agent_2'][:, :10])
            bufs['gates'].append(gates)
            bufs['gate_lps'].append(gate_lp)
            bufs['p1_hat'].append(p1); bufs['p2_hat'].append(p2)
            bufs['lp1'].append(lp1); bufs['lp2'].append(lp2)
            bufs['raw_rewards'].append(r); bufs['step_costs'].append(costs); bufs['dones'].append(d.float())
            obs, state = no, ns
            last_msg, tau = new_last_msg, new_tau
            
        res = trainer.train_step({k: torch.stack(v) for k, v in bufs.items()})
        print(f"[{mode}] upd={upd}: cost={res['batch_cost']:.2f}, lambda={res['dual_lambda']:.4f}")

print("ZOH end-to-end gradient smoke test complete!")
