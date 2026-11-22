import os
import sys
import numpy as np
import torch
import time

from mpe2 import simple_spread_v3
from src.channel.broadcast_channel import BroadcastChannel
from src.models.comm_mappo_net import CommActor, CentralizedCritic
from src.quantiser.uniform_quantiser import UniformQuantiser
from src.algos.mappo_trainer import MAPPOTrainer

class InfoExchangeSpreadEnv:
    """
    Partially observable cooperative navigation benchmark requiring information exchange.
    N = 3 agents, N = 3 landmarks.
    Each agent i must navigate to target Landmark i.
    However, Landmark i is placed in the sector of teammate (i+1)%3, outside agent i's observation
    radius (r_obs = 0.40). Teammate (i+1)%3 observes Landmark i within its local radius.
    To reach its target, agent i requires communication from teammate (i+1)%3.
    """
    def __init__(self, n_agents=3, r_obs=0.40, max_cycles=25, full_obs=False):
        self.n_agents = n_agents
        self.r_obs = r_obs
        self.max_cycles = max_cycles
        self.full_obs = full_obs
        self.env = simple_spread_v3.parallel_env(N=n_agents, continuous_actions=False, max_cycles=max_cycles)
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.action_dim = 5
        
        # Self velocity (2) + Self pos (2) + 3 Landmarks rel_pos & mask (3*3=9) + 2 Peers rel_pos & mask (2*3=6) = 19
        self.obs_dim = 4 + 3 * 3 + 2 * 3
        # State: all agents vel & pos (3*4=12) + all landmarks pos (3*2=6) = 18
        self.state_dim = n_agents * 4 + n_agents * 2
        
        # Placeholders for endpoints (will be set empirically from random and full-obs baselines)
        self.R_MIN = -35.0
        self.R_MAX = 20.0

    def normalize_score(self, return_val):
        score = (return_val - self.R_MIN) / (self.R_MAX - self.R_MIN)
        return float(np.clip(score, 0.0, 1.0))

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        raw = self.env.unwrapped
        rng = np.random.RandomState(seed)
        
        # Fixed asymmetric initial sectors:
        # Agent 0 at (-0.6, -0.6), Agent 1 at (0.6, -0.6), Agent 2 at (0.0, 0.7)
        agent_bases = [[-0.6, -0.6], [0.6, -0.6], [0.0, 0.7]]
        for i, a in enumerate(raw.world.agents):
            jitter = rng.uniform(-0.05, 0.05, size=2)
            a.state.p_pos = np.array(agent_bases[i]) + jitter
            a.state.p_vel = np.zeros(2)
            
        # Target landmark i placed near teammate (i+1)%3:
        # Distance to teammate ~ 0.20 (within r_obs=0.40 of teammate)
        # Distance to assigned agent i ~ 1.30 (way outside r_obs=0.40 of agent i)
        for i, lm in enumerate(raw.world.landmarks):
            teammate_idx = (i + 1) % self.n_agents
            teammate_pos = raw.world.agents[teammate_idx].state.p_pos
            offset = rng.uniform(-0.15, 0.15, size=2)
            lm.state.p_pos = teammate_pos + offset
            
        obs = self._extract_obs()
        state = self._get_state()
        return obs, state

    def _extract_obs(self):
        raw = self.env.unwrapped
        obs_dict = {}
        for i, a in enumerate(raw.world.agents):
            agent_obs = []
            agent_obs.extend(a.state.p_vel)
            agent_obs.extend(a.state.p_pos)
            
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
        
        # Shared cooperative reward: each agent reaching its assigned target landmark i
        r = 0.0
        for i in range(self.n_agents):
            a_pos = raw.world.agents[i].state.p_pos
            lm_pos = raw.world.landmarks[i].state.p_pos
            dist = np.linalg.norm(a_pos - lm_pos)
            if dist <= 0.20:
                r += 1.0
            r -= 0.5 * dist
            
        # Collision penalty
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(raw.world.agents[i].state.p_pos - raw.world.agents[j].state.p_pos) < 0.30:
                    r -= 0.10
                    
        obs = self._extract_obs()
        state = self._get_state()
        return obs, state, float(r), False, {}

if __name__ == "__main__":
    env = InfoExchangeSpreadEnv()
    obs, state = env.reset(seed=42)
    print("Environment created successfully. Obs shape:", obs['agent_0'].shape, "State shape:", state.shape)
    
    # Evaluate pure random policy over 20 test seeds
    test_seeds = [1000 + i * 17 for i in range(20)]
    random_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(25):
            acts = {a: np.random.randint(0, 5) for a in env.agents}
            _, _, r, _, _ = env.step(acts)
            ep_r += r
        random_rets.append(ep_r)
    print(f"Random Policy Lower Bound (20 episodes): Mean Return = {np.mean(random_rets):.2f} (std = {np.std(random_rets):.2f})")
