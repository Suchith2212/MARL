import os
import sys
import numpy as np
import torch
import time

from mpe2 import simple_spread_v3
from src.channel.broadcast_channel import BroadcastChannel
from src.algos.mappo_trainer import MAPPOTrainer
from src.utils.metrics import compute_paired_ci_95
from scripts.run_phase1_screen import set_seed

class ScoutWorkerTargetEnv:
    """
    Partially observable cooperative target coverage task with distributed observation.
    N = 3 agents:
    - Agent 0 is the Observer / Scout: observes the target coordinates (x*, y*).
    - Agents 1 and 2 are Workers: blind to the target coordinates (r_obs = 0.15) unless communicating.
    Shared team reward: all agents converging to the target coordinate.
    When full_obs=True, all agents directly observe the target coordinate.
    """
    def __init__(self, n_agents=3, max_cycles=25, full_obs=False):
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.full_obs = full_obs
        self.env = simple_spread_v3.parallel_env(N=n_agents, continuous_actions=False, max_cycles=max_cycles)
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.action_dim = 5
        
        # Obs: self vel (2) + self pos (2) + target rel_pos (2) + target visible mask (1) + peer rel_pos & masks (2*3=6) = 13 floats
        # Pad to 16 for message dimension alignment:
        # [vel_x, vel_y, pos_x, pos_y, rel_target_x, rel_target_y, target_mask, is_scout, peer1_x, peer1_y, mask1, peer2_x, peer2_y, mask2, 0, 0] = 16 floats
        self.obs_dim = 16
        # State: all agents vel & pos (12) + target pos (2) + target active (1) = 15 floats (pad to 18)
        self.state_dim = 18
        
        # Empirical endpoints (set after measuring random and full_obs)
        self.R_MIN = -55.0
        self.R_MAX = 35.0

    def normalize_score(self, return_val):
        score = (return_val - self.R_MIN) / (self.R_MAX - self.R_MIN)
        return float(np.clip(score, 0.0, 1.0))

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        raw = self.env.unwrapped
        rng = np.random.RandomState(seed)
        
        # Target coordinate in arena [-0.7, 0.7]^2
        self.target_pos = rng.uniform(-0.65, 0.65, size=2)
        # Position landmark 0 at target_pos
        raw.world.landmarks[0].state.p_pos = self.target_pos.copy()
        
        # Spawn agents in random locations
        for i, a in enumerate(raw.world.agents):
            a.state.p_pos = rng.uniform(-0.8, 0.8, size=2)
            a.state.p_vel = np.zeros(2)
            
        obs = self._extract_obs()
        state = self._get_state()
        return obs, state

    def _extract_obs(self):
        raw = self.env.unwrapped
        obs_dict = {}
        for i, a in enumerate(raw.world.agents):
            agent_obs = []
            agent_obs.extend(a.state.p_vel) # 2
            agent_obs.extend(a.state.p_pos) # 2
            
            # Target visibility:
            # Agent 0 (Scout) always observes target!
            # If full_obs, all agents observe target!
            # Otherwise, Agent 1 & 2 only observe target if within distance 0.15!
            dist_to_target = np.linalg.norm(self.target_pos - a.state.p_pos)
            rel_target = self.target_pos - a.state.p_pos
            
            if self.full_obs or (i == 0) or (dist_to_target <= 0.15):
                agent_obs.extend([float(rel_target[0]), float(rel_target[1]), 1.0])
            else:
                agent_obs.extend([0.0, 0.0, 0.0]) # Blind to target
                
            agent_obs.append(1.0 if i == 0 else 0.0) # is_scout flag (1)
            
            # Peer relative positions (2 peers * 3 = 6)
            for j, other in enumerate(raw.world.agents):
                if i == j: continue
                rel_peer = other.state.p_pos - a.state.p_pos
                agent_obs.extend([float(rel_peer[0]), float(rel_peer[1]), 1.0])
                
            agent_obs.extend([0.0, 0.0]) # Pad 2 floats -> total 16
            obs_dict[f"agent_{i}"] = np.array(agent_obs, dtype=np.float32)
        return obs_dict

    def _get_state(self):
        raw = self.env.unwrapped
        state = []
        for a in raw.world.agents:
            state.extend(a.state.p_vel); state.extend(a.state.p_pos) # 12
        state.extend(self.target_pos) # 2
        state.extend([1.0, 0.0, 0.0, 0.0]) # Pad 4 -> total 18
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        self.env.step(actions)
        raw = self.env.unwrapped
        
        # Shared team reward: all agents reaching the target coordinate
        r = 0.0
        for i in range(self.n_agents):
            a_pos = raw.world.agents[i].state.p_pos
            dist = np.linalg.norm(a_pos - self.target_pos)
            if dist <= 0.18:
                r += 1.5
            r -= 0.5 * dist
            
        # Collision penalty
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(raw.world.agents[i].state.p_pos - raw.world.agents[j].state.p_pos) < 0.25:
                    r -= 0.10
                    
        obs = self._extract_obs()
        state = self._get_state()
        return obs, state, float(r), False, {}

def eval_policy(env, channel, trainer, test_seeds):
    rets = []
    for s in test_seeds:
        obs, state = env.reset(seed=s)
        if channel: channel.reset()
        ep_r = 0.0
        for t in range(env.max_cycles):
            h_list, raw_msgs, gates, _, bit_widths, _ = trainer.act(obs, deterministic=True)
            if channel:
                q_msgs = [trainer.quantiser(raw_msgs[i:i+1], bit_width=bit_widths[i]).squeeze(0) if gates[i]==1 and bit_widths[i]>0 else torch.zeros(16) for i in range(3)]
                delivered, _ = channel.transmit_and_deliver(q_msgs, gates, bit_widths)
                actions, _ = trainer.get_actions(h_list, delivered, deterministic=True)
            else:
                actions, _ = trainer.get_actions(h_list, {0:[], 1:[], 2:[]}, deterministic=True)
            obs, state, r, done, _ = env.step(actions)
            ep_r += r
        rets.append(ep_r)
    return float(np.mean(rets))

if __name__ == "__main__":
    test_seeds = [1000 + i * 17 for i in range(20)]
    env = ScoutWorkerTargetEnv()
    
    # 1. Random policy lower bound
    random_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(25):
            acts = {a: np.random.randint(0, 5) for a in env.agents}
            _, _, r, _, _ = env.step(acts)
            ep_r += r
        random_rets.append(ep_r)
    mean_random = float(np.mean(random_rets))
    print(f"Random Lower Bound: Mean Return = {mean_random:.2f}")

    # 2. Test No-Comm vs Full-Comm on Seed 42 and 43
    channel = BroadcastChannel(n_agents=3, message_dim=16)
    for seed in [42, 43]:
        # No-comm
        set_seed(seed)
        trainer_no = MAPPOTrainer(env.obs_dim, env.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=5e-4, critic_lr=1e-3)
        for update in range(40):
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
        ret_no = eval_policy(env, None, trainer_no, test_seeds)

        # Full-comm b=8
        set_seed(seed)
        trainer_fc = MAPPOTrainer(env.obs_dim, env.state_dim, mode='full_comm_fixed', fixed_bit_width=8, actor_lr=5e-4, critic_lr=1e-3)
        for update in range(40):
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
