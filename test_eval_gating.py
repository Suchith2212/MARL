import torch
from src.envs.vec_tracking_env import BatchedTrackingEnv
from src.algos.primal_dual_zoh import ZOHMAPPOTrainer

VAL_SEEDS = [42, 43]
BUDGET = 16
BW = 4

torch.set_num_threads(1)
torch.manual_seed(42)
env = BatchedTrackingEnv(num_envs=64, max_cycles=25, stage='dynamic', full_obs=False)
trainer = ZOHMAPPOTrainer(
    mode='primal_dual', fixed_bw=BW, budget=BUDGET, d=16, T=25,
    K_p=0.005, alpha_I=0.01, warmup_updates=25
)
for upd in range(1, 151):
    obs, state = env.reset(seed=42000 + upd)
    last_msg = torch.zeros(64, 16)
    tau = torch.zeros(64, 1)
    bufs = {k: [] for k in ['states','scout_obs','rover1_obs','rover2_obs','held_msg','tau',
                              'gates','gate_lps','p1_hat','p2_hat','lp1','lp2','raw_rewards','step_costs','dones']}
    for t in range(25):
        bufs['held_msg'].append(last_msg); bufs['tau'].append(tau)
        t_est, g, glp, (p1, p2), (lp1, lp2), sc, nl_msg, n_tau = trainer.act(obs, last_msg, tau, t)
        no, ns, r, d, _ = env.step_with_target_estimates(t_est)
        bufs['states'].append(state); bufs['scout_obs'].append(obs['agent_0'])
        bufs['rover1_obs'].append(obs['agent_1'][:, :10]); bufs['rover2_obs'].append(obs['agent_2'][:, :10])
        bufs['gates'].append(g); bufs['gate_lps'].append(glp)
        bufs['p1_hat'].append(p1); bufs['p2_hat'].append(p2)
        bufs['lp1'].append(lp1); bufs['lp2'].append(lp2)
        bufs['raw_rewards'].append(r); bufs['step_costs'].append(sc); bufs['dones'].append(d.float())
        obs, state = no, ns; last_msg, tau = nl_msg, n_tau
    res = trainer.train_step({k: torch.stack(v) for k, v in bufs.items()})

print(f"Training final batch cost: {res['batch_cost']:.2f}, lambda: {res['dual_lambda']:.4f}")

# Eval with integrate-and-fire vs threshold 0.5
for method in ['threshold_0.5', 'integrate_and_fire']:
    eval_env = BatchedTrackingEnv(500, 25, stage='dynamic', full_obs=False)
    e_obs, _ = eval_env.reset(seed=99999)
    e_msg, e_tau = torch.zeros(500, 16), torch.zeros(500, 1)
    acc = torch.zeros(500)
    tot_r, tot_c = torch.zeros(500), torch.zeros(500)
    with torch.no_grad():
        for t in range(25):
            _, logits = trainer.get_gate_dist(e_obs['agent_0'])
            p = torch.sigmoid(logits)
            if method == 'threshold_0.5':
                g = (p >= 0.5).float()
            else:
                acc += p
                g = (acc >= 1.0).float()
                acc = acc - g
            # Use deterministic gate
            raw_msg = trainer.encoder(e_obs['agent_0'])
            q_msg = trainer.quantiser(raw_msg, bit_width=BW)
            nl_msg, n_tau = trainer.update_zoh_buffer(e_msg, e_tau, q_msg, g)
            r1_in = torch.cat([e_obs['agent_1'][:, :6], nl_msg, n_tau], dim=-1)
            r2_in = torch.cat([e_obs['agent_2'][:, :6], nl_msg, n_tau], dim=-1)
            p1_hat, _ = trainer.rover1_policy.sample(r1_in, deterministic=True)
            p2_hat, _ = trainer.rover2_policy.sample(r2_in, deterministic=True)
            t_est = torch.zeros(500, 3, 2)
            t_est[:, 1], t_est[:, 2] = p1_hat, p2_hat
            e_obs, _, r, _, _ = eval_env.step_with_target_estimates(t_est)
            tot_r += r
            tot_c += g * (16 * BW)
            e_msg, e_tau = nl_msg, n_tau
    mean_c = float(tot_c.mean() / 25.0)
    mean_r = float(tot_r.mean())
    print(f"[{method}] Cost: {mean_c:.2f} (budget {BUDGET}), Return: {mean_r:.2f}")
