import numpy as np
import torch
import time

from test_info_exchange_mpe import InfoExchangeMPEEnv, eval_policy
from src.channel.broadcast_channel import BroadcastChannel
from src.algos.mappo_trainer import MAPPOTrainer
from scripts.run_phase1_screen import set_seed

def main():
    test_seeds = [1000 + i * 17 for i in range(20)]
    env = InfoExchangeMPEEnv()
    channel = BroadcastChannel(n_agents=3, message_dim=16)

    for seed in [42, 43]:
        # 1. No-Comm
        set_seed(seed)
        trainer_no = MAPPOTrainer(env.obs_dim, env.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=5e-4, critic_lr=1e-3)
        for update in range(60):
            batch_obs, batch_states, batch_actions, batch_action_lps = [], [], [], []
            batch_gates, batch_gate_lps, batch_bit_widths, batch_width_lps = [], [], [], []
            batch_deliv_masks, batch_rewards, batch_step_costs, batch_dones = [], [], [], []
            for ep in range(6):
                obs, state = env.reset(seed=seed * 1000 + update * 10 + ep)
                for t in range(env.max_cycles):
                    h_list, raw_msgs, gates, gate_lps, bit_widths, width_lps = trainer_no.act(obs, deterministic=False)
                    actions, action_lps = trainer_no.get_actions(h_list, {0:[], 1:[], 2:[]}, deterministic=False)
                    next_obs, next_state, reward, done, _ = env.step(actions)
                    batch_obs.append([obs[f'agent_{i}'] for i in range(3)])
                    batch_states.append(state)
                    batch_actions.append([actions[f'agent_{i}'] for i in range(3)])
                    batch_action_lps.append([action_lps[f'agent_{i}'] for i in range(3)])
                    batch_gates.append(gates); batch_gate_lps.append(gate_lps)
                    batch_bit_widths.append(bit_widths); batch_width_lps.append(width_lps)
                    batch_deliv_masks.append([[0]*3 for _ in range(3)])
                    batch_rewards.append(reward); batch_step_costs.append(0.0)
                    batch_dones.append(1.0 if t == 24 else 0.0)
                    obs, state = next_obs, next_state
            trainer_no.train_step({
                'obs': batch_obs, 'state': batch_states, 'actions': batch_actions, 'action_lps': batch_action_lps,
                'gates': batch_gates, 'gate_lps': batch_gate_lps, 'bit_widths': batch_bit_widths, 'width_lps': batch_width_lps,
                'deliv_masks': batch_deliv_masks, 'rewards': batch_rewards, 'step_costs': batch_step_costs, 'dones': batch_dones
            })
        ret_no = eval_policy(env, channel, trainer_no, test_seeds)

        # 2. Full-Comm b=8
        set_seed(seed)
        trainer_fc = MAPPOTrainer(env.obs_dim, env.state_dim, mode='full_comm_fixed', fixed_bit_width=8, actor_lr=5e-4, critic_lr=1e-3)
        for update in range(60):
            batch_obs, batch_states, batch_actions, batch_action_lps = [], [], [], []
            batch_gates, batch_gate_lps, batch_bit_widths, batch_width_lps = [], [], [], []
            batch_deliv_masks, batch_rewards, batch_step_costs, batch_dones = [], [], [], []
            for ep in range(6):
                obs, state = env.reset(seed=seed * 1000 + update * 10 + ep)
                channel.reset()
                for t in range(env.max_cycles):
                    h_list, raw_msgs, gates, gate_lps, bit_widths, width_lps = trainer_fc.act(obs, deterministic=False)
                    q_msgs = [trainer_fc.quantiser(raw_msgs[i:i+1], bit_width=8).squeeze(0) for i in range(3)]
                    delivered, step_cost = channel.transmit_and_deliver(q_msgs, gates, bit_widths)
                    actions, action_lps = trainer_fc.get_actions(h_list, delivered, deterministic=False)
                    mask_matrix = [[0]*3 for _ in range(3)]
                    for r_idx, items in delivered.items():
                        for it in items: mask_matrix[it['sender']][r_idx] = 1
                    next_obs, next_state, reward, done, _ = env.step(actions)
                    batch_obs.append([obs[f'agent_{i}'] for i in range(3)])
                    batch_states.append(state)
                    batch_actions.append([actions[f'agent_{i}'] for i in range(3)])
                    batch_action_lps.append([action_lps[f'agent_{i}'] for i in range(3)])
                    batch_gates.append(gates); batch_gate_lps.append(gate_lps)
                    batch_bit_widths.append(bit_widths); batch_width_lps.append(width_lps)
                    batch_deliv_masks.append(mask_matrix)
                    batch_rewards.append(reward); batch_step_costs.append(step_cost)
                    batch_dones.append(1.0 if t == 24 else 0.0)
                    obs, state = next_obs, next_state
            trainer_fc.train_step({
                'obs': batch_obs, 'state': batch_states, 'actions': batch_actions, 'action_lps': batch_action_lps,
                'gates': batch_gates, 'gate_lps': batch_gate_lps, 'bit_widths': batch_bit_widths, 'width_lps': batch_width_lps,
                'deliv_masks': batch_deliv_masks, 'rewards': batch_rewards, 'step_costs': batch_step_costs, 'dones': batch_dones
            })
        ret_fc = eval_policy(env, channel, trainer_fc, test_seeds)
        print(f"Seed {seed}: No-Comm = {ret_no:.2f}, Full-Comm = {ret_fc:.2f}, Diff = {ret_fc - ret_no:+.2f}")

if __name__ == "__main__":
    main()
