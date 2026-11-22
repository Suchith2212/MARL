import torch
import numpy as np
from src.envs.vec_tracking_env import BatchedTrackingEnv
from src.algos.learned_precision_zoh import LearnedPrecisionMAPPOTrainer

def test_learned_precision():
    torch.manual_seed(42)
    np.random.seed(42)

    B_ENV = 8
    T = 25
    env = BatchedTrackingEnv(num_envs=B_ENV, max_cycles=T, stage="dynamic", full_obs=False)
    trainer = LearnedPrecisionMAPPOTrainer(budget=16, d=16, T=T, warmup_updates=2)

    obs, state = env.reset(seed=42)
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
        for t_step in range(T):
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
        'states': torch.stack(buf_states),
        'scout_obs': torch.stack(buf_scout),
        'rover1_obs': torch.stack(buf_r1),
        'rover2_obs': torch.stack(buf_r2),
        'held_msg': torch.stack(buf_held_msg),
        'tau': torch.stack(buf_tau),
        'gates': torch.stack(buf_gates),
        'gate_lps': torch.stack(buf_gate_lp),
        'width_idx': torch.stack(buf_widx),
        'width_lps': torch.stack(buf_wlp),
        'widths': torch.stack(buf_widths),
        'p1_hat': torch.stack(buf_p1),
        'p2_hat': torch.stack(buf_p2),
        'lp1': torch.stack(buf_lp1),
        'lp2': torch.stack(buf_lp2),
        'raw_rewards': torch.stack(buf_rewards),
        'step_costs': torch.stack(buf_costs),
        'dones': torch.stack(buf_dones),
    }

    info = trainer.train_step(rollouts)
    print(f"Smoke test passed! Batch cost: {info['batch_cost']:.2f}, Lambda: {info['dual_lambda']:.4f}")

    # Test deterministic evaluation
    eval_env = BatchedTrackingEnv(num_envs=16, max_cycles=T, stage="dynamic", full_obs=False)
    obs, state = eval_env.reset(seed=999)
    last_msg = torch.zeros(16, 16)
    tau = torch.zeros(16, 1)
    acc = torch.zeros(16)
    tot_r = torch.zeros(16)
    tot_c = torch.zeros(16)

    with torch.no_grad():
        for t in range(T):
            (t_est, gates, _, widx, _, widths,
             _, _, sc, nl_msg, n_tau, acc) = trainer.act(
                obs, last_msg, tau, t, acc=acc, deterministic=True
            )
            next_obs, next_state, r, d, _ = eval_env.step_with_target_estimates(t_est)
            tot_r += r
            tot_c += sc
            obs, state = next_obs, next_state
            last_msg, tau = nl_msg, n_tau

    bar_c = float(tot_c.mean().item()) / T
    mean_r = float(tot_r.mean().item())
    print(f"Deterministic eval test passed: Return={mean_r:.2f}, Cost={bar_c:.2f} bits/step")

if __name__ == '__main__':
    test_learned_precision()
