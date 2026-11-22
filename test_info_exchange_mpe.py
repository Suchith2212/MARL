import os
import sys
import numpy as np
import torch
import time

from mpe2 import simple_spread_v3
from src.channel.broadcast_channel import BroadcastChannel
from src.algos.mappo_trainer import MAPPOTrainer
from scripts.run_phase1_screen import set_seed

class InfoExchangeMPEEnv:
    """
    Partially observable cooperative MPE navigation with distributed target clues.
    N = 3 agents, N = 3 landmarks.
    Agent i has target Landmark i.
    However, the clue for target i (which landmark agent i must navigate to) is observed ONLY
    by teammate (i+1)%3.
    Without communication, agent i does not know its target and cannot solve the task.
    With communication, teammate (i+1)%3 transmits the target information, allowing agent i
    to reach its target and achieve maximum reward.
    """
    def __init__(self, n_agents=3, max_cycles=25, full_obs=False):
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.full_obs = full_obs
        self.env = simple_spread_v3.parallel_env(N=n_agents, continuous_actions=False, max_cycles=max_cycles)
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.action_dim = 5
        
        # Self vel (2) + Self pos (2) + 3 Landmarks rel_pos (3*2=6) + Teammate target clue one-hot (3) + Own target clue (3 if full_obs else 0)
        # To maintain uniform obs_dim:
        # Obs: [vel (2), pos (2), rel_lm0 (2), rel_lm1 (2), rel_lm2 (2), observed_clue (3), own_target_clue (3 if full_obs else zeros)] = 17 floats
        self.obs_dim = 2 + 2 + 6 + 3 + 3 # 16 floats + 1 padding or 16
        self.obs_dim = 16
        # State: all agents vel & pos (12) + all landmarks pos (6) + all target assignments (3) = 21 floats
        self.state_dim = n_agents * 4 + n_agents * 2 + n_agents
        
        self.R_MIN = -45.0
        self.R_MAX = 35.0

    def normalize_score(self, return_val):
        score = (return_val - self.R_MIN) / (self.R_MAX - self.R_MIN)
        return float(np.clip(score, 0.0, 1.0))

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        raw = self.env.unwrapped
        rng = np.random.RandomState(seed)
        
        # Random permutation of target landmarks
        # targets[i] is the landmark index assigned to Agent i
        self.targets = rng.permutation(self.n_agents)
        
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
            
            # Relative vectors to all 3 landmarks
            for lm in raw.world.landmarks:
                rel = lm.state.p_pos - a.state.p_pos
                agent_obs.extend([float(rel[0]), float(rel[1])]) # 3 * 2 = 6
                
            # Teammate target clue observed by Agent i:
            # Agent i observes the target of teammate (i-1)%3!
            teammate_target = self.targets[(i - 1) % self.n_agents]
            clue_one_hot = [1.0 if k == teammate_target else 0.0 for k in range(self.n_agents)]
            agent_obs.extend(clue_one_hot) # 3
            
            # Own target clue:
            # Under full_obs, agent i directly knows its own target!
            # Under partial obs, own target clue is masked to [0, 0, 0]!
            if self.full_obs:
                own_target = self.targets[i]
                own_clue = [1.0 if k == own_target else 0.0 for k in range(self.n_agents)]
            else:
                own_clue = [0.0, 0.0, 0.0]
            agent_obs.extend(own_clue) # 3
            
            obs_dict[f"agent_{i}"] = np.array(agent_obs, dtype=np.float32)
        return obs_dict

    def _get_state(self):
        raw = self.env.unwrapped
        state = []
        for a in raw.world.agents:
            state.extend(a.state.p_vel); state.extend(a.state.p_pos)
        for lm in raw.world.landmarks:
            state.extend(lm.state.p_pos)
        for t in self.targets:
            state.append(float(t))
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        self.env.step(actions)
        raw = self.env.unwrapped
        
        # Reward: each agent i reaching its assigned target landmark targets[i]
        r = 0.0
        for i in range(self.n_agents):
            target_lm_idx = self.targets[i]
            a_pos = raw.world.agents[i].state.p_pos
            lm_pos = raw.world.landmarks[target_lm_idx].state.p_pos
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
    env = InfoExchangeMPEEnv()
    print("Obs dim:", env.obs_dim, "State dim:", env.state_dim)
    
    # 1. Random Policy Lower Bound
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
    
    # 2. Full-Obs Upper Bound (30 updates on seed 42)
    env_fo = InfoExchangeMPEEnv(full_obs=True)
    set_seed(42)
    trainer_fo = MAPPOTrainer(env_fo.obs_dim, env_fo.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=5e-4, critic_lr=1e-3)
    for update in range(30):
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
    print(f"Full-Obs Upper Bound (30 updates): Mean Return = {ret_fo:.2f}")

    # 3. No-Comm Baseline (30 updates on seed 42)
    env_po = InfoExchangeMPEEnv(full_obs=False)
    channel = BroadcastChannel(n_agents=3, message_dim=16)
    set_seed(42)
    trainer_no = MAPPOTrainer(env_po.obs_dim, env_po.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=5e-4, critic_lr=1e-3)
    for update in range(30):
        batch_obs, batch_states, batch_actions, batch_action_lps = [], [], [], []
        batch_gates, batch_gate_lps, batch_bit_widths, batch_width_lps = [], [], [], []
        batch_deliv_masks, batch_rewards, batch_step_costs, batch_dones = [], [], [], []
        for ep in range(6):
            obs, state = env_po.reset(seed=42 * 1000 + update * 10 + ep)
            for t in range(env_po.max_cycles):
                h_list, raw_msgs, gates, gate_lps, bit_widths, width_lps = trainer_no.act(obs, deterministic=False)
                actions, action_lps = trainer_no.get_actions(h_list, {0:[], 1:[], 2:[]}, deterministic=False)
                next_obs, next_state, reward, done, _ = env_po.step(actions)
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
    ret_no = eval_policy(env_po, channel, trainer_no, test_seeds)
    print(f"No-Comm Return (30 updates): Mean Return = {ret_no:.2f}")

    # 4. Full-Comm Baseline (30 updates on seed 42, b=8)
    set_seed(42)
    trainer_fc = MAPPOTrainer(env_po.obs_dim, env_po.state_dim, mode='full_comm_fixed', fixed_bit_width=8, actor_lr=5e-4, critic_lr=1e-3)
    for update in range(30):
        batch_obs, batch_states, batch_actions, batch_action_lps = [], [], [], []
        batch_gates, batch_gate_lps, batch_bit_widths, batch_width_lps = [], [], [], []
        batch_deliv_masks, batch_rewards, batch_step_costs, batch_dones = [], [], [], []
        for ep in range(6):
            obs, state = env_po.reset(seed=42 * 1000 + update * 10 + ep)
            channel.reset()
            for t in range(env_po.max_cycles):
                h_list, raw_msgs, gates, gate_lps, bit_widths, width_lps = trainer_fc.act(obs, deterministic=False)
                q_msgs = [trainer_fc.quantiser(raw_msgs[i:i+1], bit_width=8).squeeze(0) for i in range(3)]
                delivered, step_cost = channel.transmit_and_deliver(q_msgs, gates, bit_widths)
                actions, action_lps = trainer_fc.get_actions(h_list, delivered, deterministic=False)
                mask_matrix = [[0]*3 for _ in range(3)]
                for r_idx, items in delivered.items():
                    for it in items: mask_matrix[it['sender']][r_idx] = 1
                next_obs, next_state, reward, done, _ = env_po.step(actions)
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
    ret_fc = eval_policy(env_po, channel, trainer_fc, test_seeds)
    print(f"Full-Comm Return (30 updates): Mean Return = {ret_fc:.2f}")
    print(f"Full-Comm vs No-Comm Difference: {ret_fc - ret_no:+.2f}")
