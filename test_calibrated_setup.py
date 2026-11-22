import os
import sys
import numpy as np
import torch
import time

from mpe2 import simple_spread_v3
from src.channel.broadcast_channel import BroadcastChannel
from src.algos.mappo_trainer import MAPPOTrainer
from scripts.run_phase1_screen import set_seed

class CalibratedInfoExchangeEnv:
    """
    Calibrated partially observable multi-agent target exchange benchmark.
    - N = 3 agents, N = 3 landmarks.
    - Observation radius r_obs = 0.35.
    - Agent bases: [-0.35, -0.35], [0.35, -0.35], [0.0, 0.40].
    - Target landmark i is placed near teammate (i+1)%3 (dist ~ 0.15 <= r_obs from teammate, dist ~ 0.70 > r_obs from agent i).
    - Teammate (i+1)%3 observes target i, agent i cannot observe target i.
    - When full_obs=True, radius masking is removed (Upper Bound).
    """
    def __init__(self, n_agents=3, r_obs=0.35, max_cycles=25, full_obs=False):
        self.n_agents = n_agents
        self.r_obs = r_obs
        self.max_cycles = max_cycles
        self.full_obs = full_obs
        self.env = simple_spread_v3.parallel_env(N=n_agents, continuous_actions=False, max_cycles=max_cycles)
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.action_dim = 5
        self.obs_dim = 4 + 3 * 3 + 2 * 3 # 19 floats
        self.state_dim = n_agents * 4 + n_agents * 2 # 18 floats
        
        # Empirical endpoints (set based on random and full-obs benchmarks)
        self.R_MIN = -45.0
        self.R_MAX = 40.0

    def normalize_score(self, return_val):
        score = (return_val - self.R_MIN) / (self.R_MAX - self.R_MIN)
        return float(np.clip(score, 0.0, 1.0))

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        raw = self.env.unwrapped
        rng = np.random.RandomState(seed)
        
        # Agent initial locations
        agent_bases = [[-0.35, -0.35], [0.35, -0.35], [0.0, 0.40]]
        for i, a in enumerate(raw.world.agents):
            jitter = rng.uniform(-0.04, 0.04, size=2)
            a.state.p_pos = np.array(agent_bases[i]) + jitter
            a.state.p_vel = np.zeros(2)
            
        # Target landmark i placed near teammate (i+1)%3
        for i, lm in enumerate(raw.world.landmarks):
            teammate_idx = (i + 1) % self.n_agents
            teammate_pos = raw.world.agents[teammate_idx].state.p_pos
            offset = rng.uniform(-0.10, 0.10, size=2)
            lm.state.p_pos = teammate_pos + offset
            
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
            
            for lm in raw.world.landmarks:
                rel = lm.state.p_pos - a.state.p_pos
                dist = np.linalg.norm(rel)
                if self.full_obs or (dist <= self.r_obs):
                    agent_obs.extend([float(rel[0]), float(rel[1]), 1.0])
                else:
                    agent_obs.extend([0.0, 0.0, 0.0])
                    
            for j, other in enumerate(raw.world.agents):
                if i == j: continue
                rel = other.state.p_pos - a.state.p_pos
                dist = np.linalg.norm(rel)
                if self.full_obs or (dist <= self.r_obs):
                    agent_obs.extend([float(rel[0]), float(rel[1]), 1.0])
                else:
                    agent_obs.extend([0.0, 0.0, 0.0])
                    
            obs_dict[f"agent_{i}"] = np.array(agent_obs, dtype=np.float32)
        return obs_dict

    def _get_state(self):
        raw = self.env.unwrapped
        state = []
        for a in raw.world.agents:
            state.extend(a.state.p_vel); state.extend(a.state.p_pos)
        for lm in raw.world.landmarks:
            state.extend(lm.state.p_pos)
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        self.env.step(actions)
        raw = self.env.unwrapped
        
        # Reward: each agent i reaching target landmark i
        r = 0.0
        for i in range(self.n_agents):
            a_pos = raw.world.agents[i].state.p_pos
            lm_pos = raw.world.landmarks[i].state.p_pos
            dist = np.linalg.norm(a_pos - lm_pos)
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

def eval_policy(env, channel, trainer, test_seeds, full_obs=False):
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
    env = CalibratedInfoExchangeEnv()
    
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
    
    # 2. Test Full-Obs Upper Bound (25 updates on seed 42)
    env_fo = CalibratedInfoExchangeEnv(full_obs=True)
    set_seed(42)
    trainer_fo = MAPPOTrainer(env_fo.obs_dim, env_fo.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=5e-4, critic_lr=1e-3)
    for update in range(25):
        batch_obs, batch_states, batch_actions, batch_action_lps = [], [], [], []
        batch_gates, batch_gate_lps, batch_bit_widths, batch_width_lps = [], [], [], []
        batch_deliv_masks, batch_rewards, batch_step_costs, batch_dones = [], [], [], []
        for ep in range(6):
            obs, state = env_fo.reset(seed=42 * 1000 + update * 10 + ep)
            for t in range(env_fo.max_cycles):
                h_list, raw_msgs, gates, gate_lps, bit_widths, width_lps = trainer_fo.act(obs, deterministic=False)
                actions, action_lps = trainer_fo.get_actions(h_list, {0:[], 1:[], 2:[]}, deterministic=False)
                next_obs, next_state, reward, done, _ = env_fo.step(actions)
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
        trainer_fo.train_step({
            'obs': batch_obs, 'state': batch_states, 'actions': batch_actions, 'action_lps': batch_action_lps,
            'gates': batch_gates, 'gate_lps': batch_gate_lps, 'bit_widths': batch_bit_widths, 'width_lps': batch_width_lps,
            'deliv_masks': batch_deliv_masks, 'rewards': batch_rewards, 'step_costs': batch_step_costs, 'dones': batch_dones
        })
    ret_fo = eval_policy(env_fo, None, trainer_fo, test_seeds)
    print(f"Full-Obs Upper Bound (25 updates): Mean Return = {ret_fo:.2f}")
