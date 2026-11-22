import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.envs.po_mpe import PartiallyObservableSpreadEnv
from scripts.run_phase1_screen import train_seed

def main():
    torch.set_num_threads(1)
    seed = 42
    test_seeds = [1000 + i * 17 for i in range(100)]
    
    print("Training no_comm agent on old Phase 1 clue-nav task (seed 42)...")
    # We train the exact same no_comm policy
    env = PartiallyObservableSpreadEnv(n_agents=3, max_cycles=25, full_obs=False)
    from src.algos.mappo_trainer import MAPPOTrainer
    trainer = MAPPOTrainer(
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_agents=3,
        action_dim=env.action_dim,
        mode='no_comm',
        fixed_bit_width=0,
        actor_lr=5e-4,
        critic_lr=1e-3,
        ppo_epochs=4,
        entropy_coef=0.01,
        device='cpu'
    )
    
    # Train for 25 updates
    n_updates = 25
    eps_per_update = 12
    for update in range(n_updates):
        batch_obs, batch_states, batch_actions = [], [], []
        batch_action_lps, batch_gates, batch_gate_lps = [], [], []
        batch_bit_widths, batch_width_lps, batch_deliv_masks = [], [], []
        batch_rewards, batch_step_costs, batch_dones = [], [], []
        
        for ep in range(eps_per_update):
            obs, state = env.reset(seed=seed * 1000 + update * 12 + ep)
            for t in range(env.max_cycles):
                h_list, raw_msgs, gates, gate_lps, bit_widths, width_lps = trainer.act(obs, deterministic=False)
                actions, action_lps = trainer.get_actions(h_list, {i: [] for i in range(env.n_agents)}, deterministic=False)
                mask_matrix = [[0 for _ in range(env.n_agents)] for _ in range(env.n_agents)]
                g_train = [0, 0, 0]
                b_train = [0, 0, 0]
                step_cost = 0.0
                
                next_obs, next_state, reward, done, _ = env.step(actions)
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
        buf = {
            'obs': batch_obs, 'state': batch_states, 'actions': batch_actions,
            'action_lps': batch_action_lps, 'gates': batch_gates, 'gate_lps': batch_gate_lps,
            'bit_widths': batch_bit_widths, 'width_lps': batch_width_lps,
            'deliv_masks': batch_deliv_masks, 'rewards': batch_rewards, 'step_costs': batch_step_costs,
            'dones': batch_dones
        }
        trainer.train_step(buf)

    print("Evaluating final positions across 100 episodes...")
    pos_t0_r1, pos_t0_r2 = [], []
    pos_t1_r1, pos_t1_r2 = [], []
    
    for s in test_seeds:
        obs, state = env.reset(seed=s)
        for t in range(env.max_cycles):
            h_list, _, _, _, _, _ = trainer.act(obs, deterministic=True)
            actions, _ = trainer.get_actions(h_list, {i: [] for i in range(env.n_agents)}, deterministic=True)
            next_obs, next_state, reward, done, _ = env.step(actions)
            obs = next_obs
            state = next_state
            if done: break
        raw = env.env.unwrapped
        p1 = raw.world.agents[1].state.p_pos.copy()
        p2 = raw.world.agents[2].state.p_pos.copy()
        if env.target_idx == 0:
            pos_t0_r1.append(p1)
            pos_t0_r2.append(p2)
        else:
            pos_t1_r1.append(p1)
            pos_t1_r2.append(p2)
            
    pos_t0_r1 = np.array(pos_t0_r1)
    pos_t0_r2 = np.array(pos_t0_r2)
    pos_t1_r1 = np.array(pos_t1_r1)
    pos_t1_r2 = np.array(pos_t1_r2)
    
    print(f"Target 0 (L0 at -0.6, 0.0) Count: {len(pos_t0_r1)}")
    print(f"  Rover 1 Mean Final Pos: {np.mean(pos_t0_r1, axis=0)}")
    print(f"  Rover 2 Mean Final Pos: {np.mean(pos_t0_r2, axis=0)}")
    print(f"Target 1 (L1 at +0.6, 0.0) Count: {len(pos_t1_r1)}")
    print(f"  Rover 1 Mean Final Pos: {np.mean(pos_t1_r1, axis=0)}")
    print(f"  Rover 2 Mean Final Pos: {np.mean(pos_t1_r2, axis=0)}")
    
    # Also evaluate Random policy positions
    rand_pos_r1, rand_pos_r2 = [], []
    for s in test_seeds:
        env.reset(seed=s)
        for t in range(env.max_cycles):
            acts = {f"agent_{i}": np.random.randint(0, 5) for i in range(3)}
            _, _, _, done, _ = env.step(acts)
            if done: break
        raw = env.env.unwrapped
        rand_pos_r1.append(raw.world.agents[1].state.p_pos.copy())
        rand_pos_r2.append(raw.world.agents[2].state.p_pos.copy())
    rand_pos_r1 = np.array(rand_pos_r1)
    rand_pos_r2 = np.array(rand_pos_r2)
    print(f"Random Policy Rover 1 Mean Final Pos: {np.mean(rand_pos_r1, axis=0)}")
    print(f"Random Policy Rover 2 Mean Final Pos: {np.mean(rand_pos_r2, axis=0)}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    L0 = np.array([-0.6, 0.0])
    L1 = np.array([0.6, 0.0])
    
    for ax, p_r1, p_r2, title in [(axes[0], pos_t0_r1, pos_t0_r2, "Target is L0 (-0.6, 0.0)"),
                                   (axes[1], pos_t1_r1, pos_t1_r2, "Target is L1 (+0.6, 0.0)")]:
        ax.scatter(p_r1[:, 0], p_r1[:, 1], color='blue', alpha=0.6, label='Rover 1 Final Pos')
        ax.scatter(p_r2[:, 0], p_r2[:, 1], color='cyan', alpha=0.6, label='Rover 2 Final Pos')
        ax.scatter(L0[0], L0[1], color='red', marker='X', s=120, label='L0 (-0.6, 0)')
        ax.scatter(L1[0], L1[1], color='green', marker='X', s=120, label='L1 (+0.6, 0)')
        ax.scatter([0.0], [0.0], color='black', marker='+', s=100, label='Centroid (0,0)')
        ax.scatter([-0.1, 0.1], [-0.4, -0.4], color='orange', marker='o', s=80, label='Start Pos')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"No-Comm Trained Policy: {title}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=8)
        
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/no_comm_final_positions_split.png", dpi=300)
    plt.close()
    print("Saved plot to: results/figures/no_comm_final_positions_split.png")

if __name__ == '__main__':
    main()
