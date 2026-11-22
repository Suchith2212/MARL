import numpy as np
import torch
import time

from test_info_exchange import InfoExchangeSpreadEnv
from src.algos.mappo_trainer import MAPPOTrainer
from scripts.run_phase1_screen import set_seed

def main():
    test_seeds = [1000 + i * 17 for i in range(20)]
    env = InfoExchangeSpreadEnv(full_obs=True)
    
    set_seed(42)
    trainer = MAPPOTrainer(env.obs_dim, env.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=5e-4, critic_lr=1e-3)
    
    t0 = time.time()
    for update in range(40):
        batch_obs, batch_states, batch_actions, batch_action_lps = [], [], [], []
        batch_gates, batch_gate_lps, batch_bit_widths, batch_width_lps = [], [], [], []
        batch_deliv_masks, batch_rewards, batch_step_costs, batch_dones = [], [], [], []
        for ep in range(6):
            obs, state = env.reset(seed=42 * 1000 + update * 10 + ep)
            for t in range(env.max_cycles):
                h_list, raw_msgs, gates, gate_lps, bit_widths, width_lps = trainer.act(obs, deterministic=False)
                actions, action_lps = trainer.get_actions(h_list, {0:[], 1:[], 2:[]}, deterministic=False)
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
        trainer.train_step({
            'obs': batch_obs, 'state': batch_states, 'actions': batch_actions, 'action_lps': batch_action_lps,
            'gates': batch_gates, 'gate_lps': batch_gate_lps, 'bit_widths': batch_bit_widths, 'width_lps': batch_width_lps,
            'deliv_masks': batch_deliv_masks, 'rewards': batch_rewards, 'step_costs': batch_step_costs, 'dones': batch_dones
        })
        if (update + 1) % 10 == 0:
            rets = []
            for s in test_seeds:
                obs, state = env.reset(seed=s)
                ep_r = 0.0
                for t in range(env.max_cycles):
                    h_list, _, _, _, _, _ = trainer.act(obs, deterministic=True)
                    actions, _ = trainer.get_actions(h_list, {0:[], 1:[], 2:[]}, deterministic=True)
                    obs, state, r, done, _ = env.step(actions)
                    ep_r += r
                rets.append(ep_r)
            print(f"Full-Obs Update {update+1} ({time.time()-t0:.1f}s): Mean Return = {np.mean(rets):.2f}")

if __name__ == "__main__":
    main()
