import os
import sys
import numpy as np
import torch
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from test_continuous_env import ContinuousTrackingEnv
from src.channel.broadcast_channel import BroadcastChannel
from src.algos.mappo_trainer import MAPPOTrainer

def test_train():
    torch.set_num_threads(1)
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    test_seeds = [1000 + i * 17 for i in range(20)]
    R_MIN = -11.55
    R_MAX = 5.77
    
    for mode in ['no_comm', 'full_comm_b8', 'full_obs']:
        is_full_obs = (mode == 'full_obs')
        bw = 8 if 'b8' in mode else 0
        env = ContinuousTrackingEnv(n_agents=3, max_cycles=25, full_obs=is_full_obs)
        env.set_endpoints(R_MIN, R_MAX)
        channel = BroadcastChannel(n_agents=3, message_dim=16, p_drop=0.0, delay_tau=0) if mode == 'full_comm_b8' else None
        
        trainer = MAPPOTrainer(
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            n_agents=3,
            action_dim=env.action_dim,
            mode='full_comm_fixed' if mode == 'full_comm_b8' else 'no_comm',
            fixed_bit_width=bw,
            actor_lr=7e-4,
            critic_lr=1e-3,
            ppo_epochs=4,
            entropy_coef=0.01,
            device='cpu'
        )
        
        curve = []
        n_updates = 20
        eps_per_update = 12
        t0 = time.time()
        
        for update in range(n_updates):
            batch_obs, batch_states, batch_actions = [], [], []
            batch_action_lps, batch_gates, batch_gate_lps = [], [], []
            batch_bit_widths, batch_width_lps, batch_deliv_masks = [], [], []
            batch_rewards, batch_step_costs, batch_dones = [], [], []
            ep_returns = []
            
            for ep in range(eps_per_update):
                obs, state = env.reset(seed=seed * 1000 + update * 12 + ep)
                if channel: channel.reset()
                ep_r = 0.0
                for t in range(env.max_cycles):
                    h_list, raw_msgs, gates, gate_lps, bit_widths, width_lps = trainer.act(obs, deterministic=False)
                    if mode == 'full_comm_b8':
                        g_train = [1, 0, 0]
                        b_train = [8, 0, 0]
                        quantised = [
                            trainer.quantiser(raw_msgs[i:i+1], bit_width=b_train[i]).squeeze(0) if g_train[i] == 1 else torch.zeros(16)
                            for i in range(env.n_agents)
                        ]
                        deliv, step_cost = channel.transmit_and_deliver(quantised, g_train, b_train)
                        actions, action_lps = trainer.get_actions(h_list, deliv, deterministic=False)
                        mask_matrix = [[0 for _ in range(3)] for _ in range(3)]
                        for r_idx, items in deliv.items():
                            for it in items: mask_matrix[it['sender']][r_idx] = 1
                    else:
                        actions, action_lps = trainer.get_actions(h_list, {i: [] for i in range(3)}, deterministic=False)
                        mask_matrix = [[0 for _ in range(3)] for _ in range(3)]
                        g_train = [0, 0, 0]
                        b_train = [0, 0, 0]
                        step_cost = 0.0
                        
                    next_obs, next_state, reward, done, _ = env.step(actions)
                    ep_r += reward
                    batch_obs.append([obs[f"agent_{i}"] for i in range(env.n_agents)])
                    batch_states.append(state)
                    batch_actions.append([actions[f"agent_{i}"] for i in range(env.n_agents)])
                    batch_action_lps.append([action_lps[f"agent_{i}"] for i in range(env.n_agents)])
                    batch_gates.append(g_train)
                    batch_gate_lps.append([0.0] * env.n_agents)
                    batch_bit_widths.append(b_train)
                    batch_width_lps.append([0.0] * env.n_agents)
                    batch_deliv_masks.append(mask_matrix)
                    batch_rewards.append(reward)
                    batch_step_costs.append(step_cost)
                    batch_dones.append(1.0 if t == env.max_cycles - 1 else 0.0)
                    obs = next_obs
                    state = next_state
                    if done: break
                ep_returns.append(ep_r)
            buf = {
                'obs': batch_obs, 'state': batch_states, 'actions': batch_actions,
                'action_lps': batch_action_lps, 'gates': batch_gates, 'gate_lps': batch_gate_lps,
                'bit_widths': batch_bit_widths, 'width_lps': batch_width_lps,
                'deliv_masks': batch_deliv_masks, 'rewards': batch_rewards, 'step_costs': batch_step_costs,
                'dones': batch_dones
            }
            trainer.train_step(buf)
            curve.append(float(np.mean(ep_returns)))
            
        # Test evaluation
        test_rets = []
        for s in test_seeds:
            obs, state = env.reset(seed=s)
            if channel: channel.reset()
            ep_r = 0.0
            for t in range(env.max_cycles):
                h_list, raw_msgs, gates, _, bit_widths, _ = trainer.act(obs, deterministic=True)
                if mode == 'full_comm_b8':
                    g_eval = [1, 0, 0]
                    b_eval = [8, 0, 0]
                    quantised = [
                        trainer.quantiser(raw_msgs[i:i+1], bit_width=b_eval[i]).squeeze(0) if g_eval[i] == 1 else torch.zeros(16)
                        for i in range(env.n_agents)
                    ]
                    deliv, _ = channel.transmit_and_deliver(quantised, g_eval, b_eval)
                    actions, _ = trainer.get_actions(h_list, deliv, deterministic=True)
                else:
                    actions, _ = trainer.get_actions(h_list, {i: [] for i in range(3)}, deterministic=True)
                next_obs, next_state, reward, done, _ = env.step(actions)
                ep_r += reward
                obs = next_obs
                state = next_state
                if done: break
            test_rets.append(ep_r)
        mean_eval = float(np.mean(test_rets))
        norm_score = env.normalize_score(mean_eval)
        print(f"Mode: {mode:12s} | Train time: {time.time()-t0:.1f}s | Test Return: {mean_eval:.2f} | Score: {norm_score:.3f} | Final Train: {curve[-1]:.2f}")

if __name__ == '__main__':
    test_train()
