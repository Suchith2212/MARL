import numpy as np
import torch
from mpe2 import simple_spread_v3
from src.channel.broadcast_channel import BroadcastChannel
from src.algos.mappo_trainer import MAPPOTrainer
from src.utils.metrics import compute_paired_ci_95
from scripts.run_phase1_screen import set_seed

class QuadrantTargetExchangeEnv:
    """
    Decentralised Partially Observable Target Guidance Task.
    N = 3 agents:
    - Agent 0: Guide at (0, 0).
    - Agents 1 & 2: Rovers starting near (0, 0).
    - In each episode, Target 1 is in quadrant Q_1 in {0, 1, 2, 3} at distance 0.65.
    - Target 2 is in quadrant Q_2 in {0, 1, 2, 3} \ {Q_1} at distance 0.65.
    - Rovers have r_obs = 0.15 (blind to targets at dist 0.65).
    - Agent 0 observes the target coordinates [t1_x, t1_y, t2_x, t2_y].
    - Under full_obs, Rovers directly observe their target coordinates.
    """
    def __init__(self, n_agents=3, max_cycles=25, full_obs=False):
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.full_obs = full_obs
        self.env = simple_spread_v3.parallel_env(N=n_agents, continuous_actions=False, max_cycles=max_cycles)
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.action_dim = 5
        self.obs_dim = 16
        self.state_dim = 18

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        raw = self.env.unwrapped
        rng = np.random.RandomState(seed)
        
        # 4 quadrant positions
        quadrants = [
            np.array([-0.55,  0.55]), # 0: Top-Left
            np.array([ 0.55,  0.55]), # 1: Top-Right
            np.array([-0.55, -0.55]), # 2: Bottom-Left
            np.array([ 0.55, -0.55]), # 3: Bottom-Right
        ]
        q_indices = rng.choice(4, size=2, replace=False)
        self.t1 = quadrants[q_indices[0]] + rng.uniform(-0.04, 0.04, size=2)
        self.t2 = quadrants[q_indices[1]] + rng.uniform(-0.04, 0.04, size=2)
        
        raw.world.landmarks[0].state.p_pos = self.t1.copy()
        raw.world.landmarks[1].state.p_pos = self.t2.copy()
        raw.world.landmarks[2].state.p_pos = np.array([2.5, 2.5]) # outside arena
        
        # Guide at center, rovers separated slightly
        raw.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        raw.world.agents[1].state.p_pos = np.array([-0.12, 0.0])
        raw.world.agents[2].state.p_pos = np.array([ 0.12, 0.0])
        for a in raw.world.agents:
            a.state.p_vel = np.zeros(2)
            
        return self._extract_obs(), self._get_state()

    def _extract_obs(self):
        raw = self.env.unwrapped
        obs_dict = {}
        for i, a in enumerate(raw.world.agents):
            o = []
            o.extend(a.state.p_vel) # 2
            o.extend(a.state.p_pos) # 2
            if i == 0:
                # Guide observes both targets
                o.extend([float(self.t1[0]), float(self.t1[1]), float(self.t2[0]), float(self.t2[1])]) # 4
                o.extend([0.0] * 8) # 8 -> 16
            elif i == 1:
                # Rover 1
                dist = np.linalg.norm(self.t1 - a.state.p_pos)
                if self.full_obs or dist <= 0.15:
                    rel = self.t1 - a.state.p_pos
                    o.extend([float(rel[0]), float(rel[1])])
                else:
                    o.extend([0.0, 0.0])
                o.extend([0.0] * 10) # 10 -> 16
            else:
                # Rover 2
                dist = np.linalg.norm(self.t2 - a.state.p_pos)
                if self.full_obs or dist <= 0.15:
                    rel = self.t2 - a.state.p_pos
                    o.extend([float(rel[0]), float(rel[1])])
                else:
                    o.extend([0.0, 0.0])
                o.extend([0.0] * 10) # 10 -> 16
            obs_dict[f"agent_{i}"] = np.array(o, dtype=np.float32)
        return obs_dict

    def _get_state(self):
        raw = self.env.unwrapped
        s = []
        for a in raw.world.agents:
            s.extend(a.state.p_pos); s.extend(a.state.p_vel)
        s.extend(self.t1); s.extend(self.t2)
        s.extend([0.0, 0.0])
        return np.array(s, dtype=np.float32)

    def step(self, actions):
        acts = dict(actions)
        acts['agent_0'] = 0 # guide stationary
        self.env.step(acts)
        raw = self.env.unwrapped
        raw.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        raw.world.agents[0].state.p_vel = np.zeros(2)
        
        p1 = raw.world.agents[1].state.p_pos
        p2 = raw.world.agents[2].state.p_pos
        d1 = np.linalg.norm(p1 - self.t1)
        d2 = np.linalg.norm(p2 - self.t2)
        
        r = -0.5 * (d1 + d2)
        if d1 <= 0.20:
            r += 1.0
        if d2 <= 0.20:
            r += 1.0
        if np.linalg.norm(p1 - p2) < 0.22:
            r -= 0.10
        return self._extract_obs(), self._get_state(), float(r), False, {}

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
    env = QuadrantTargetExchangeEnv()
    
    # 1. Random policy
    r_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(25):
            acts = {a: np.random.randint(0, 5) for a in env.agents}
            _, _, r, _, _ = env.step(acts)
            ep_r += r
        r_rets.append(ep_r)
    mean_random = float(np.mean(r_rets))
    print(f"Random Lower Bound: {mean_random:.2f}", flush=True)

    # 2. Test Seed 42 and 43 No-Comm vs Full-Comm
    channel = BroadcastChannel(n_agents=3, message_dim=16)
    for seed in [42, 43]:
        # No-Comm
        set_seed(seed)
        tr_no = MAPPOTrainer(env.obs_dim, env.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=1e-3, critic_lr=2e-3)
        for update in range(30):
            b_obs, b_states, b_acts, b_act_lps = [], [], [], []
            b_gates, b_glps, b_bws, b_wlps = [], [], [], []
            b_masks, b_rews, b_costs, b_dones = [], [], [], []
            for ep in range(6):
                obs, state = env.reset(seed=seed * 1000 + update * 10 + ep)
                for t in range(25):
                    h, m, g, glp, b, blp = tr_no.act(obs, deterministic=False)
                    acts, alps = tr_no.get_actions(h, {0:[], 1:[], 2:[]}, deterministic=False)
                    nobs, nstate, r, d, _ = env.step(acts)
                    b_obs.append([obs[f'agent_{i}'] for i in range(3)])
                    b_states.append(state)
                    b_acts.append([acts[f'agent_{i}'] for i in range(3)])
                    b_act_lps.append([alps[f'agent_{i}'] for i in range(3)])
                    b_gates.append(g); b_glps.append(glp)
                    b_bws.append(b); b_wlps.append(blp)
                    b_masks.append([[0]*3 for _ in range(3)])
                    b_rews.append(r); b_costs.append(0.0)
                    b_dones.append(1.0 if t==24 else 0.0)
                    obs, state = nobs, nstate
            tr_no.train_step({
                'obs': b_obs, 'state': b_states, 'actions': b_acts, 'action_lps': b_act_lps,
                'gates': b_gates, 'gate_lps': b_glps, 'bit_widths': b_bws, 'width_lps': b_wlps,
                'deliv_masks': b_masks, 'rewards': b_rews, 'step_costs': b_costs, 'dones': b_dones
            })
        ret_no = eval_policy(env, None, tr_no, test_seeds)

        # Full-Comm
        set_seed(seed)
        tr_fc = MAPPOTrainer(env.obs_dim, env.state_dim, mode='full_comm_fixed', fixed_bit_width=8, actor_lr=1e-3, critic_lr=2e-3)
        for update in range(30):
            b_obs, b_states, b_acts, b_act_lps = [], [], [], []
            b_gates, b_glps, b_bws, b_wlps = [], [], [], []
            b_masks, b_rews, b_costs, b_dones = [], [], [], []
            for ep in range(6):
                obs, state = env.reset(seed=seed * 1000 + update * 10 + ep)
                channel.reset()
                for t in range(25):
                    h, m, g, glp, b, blp = tr_fc.act(obs, deterministic=False)
                    q_msgs = [tr_fc.quantiser(m[i:i+1], bit_width=8).squeeze(0) for i in range(3)]
                    delivered, step_cost = channel.transmit_and_deliver(q_msgs, g, b)
                    acts, alps = tr_fc.get_actions(h, delivered, deterministic=False)
                    mask_matrix = [[0]*3 for _ in range(3)]
                    for r_idx, items in delivered.items():
                        for it in items: mask_matrix[it['sender']][r_idx] = 1
                    nobs, nstate, r, d, _ = env.step(acts)
                    b_obs.append([obs[f'agent_{i}'] for i in range(3)])
                    b_states.append(state)
                    b_acts.append([acts[f'agent_{i}'] for i in range(3)])
                    b_act_lps.append([alps[f'agent_{i}'] for i in range(3)])
                    b_gates.append(g); b_glps.append(glp)
                    b_bws.append(b); b_wlps.append(blp)
                    b_masks.append(mask_matrix)
                    b_rews.append(r); b_costs.append(step_cost)
                    b_dones.append(1.0 if t==24 else 0.0)
                    obs, state = nobs, nstate
            tr_fc.train_step({
                'obs': b_obs, 'state': b_states, 'actions': b_acts, 'action_lps': b_act_lps,
                'gates': b_gates, 'gate_lps': b_glps, 'bit_widths': b_bws, 'width_lps': b_wlps,
                'deliv_masks': b_masks, 'rewards': b_rews, 'step_costs': b_costs, 'dones': b_dones
            })
        ret_fc = eval_policy(env, channel, tr_fc, test_seeds)
        print(f"Seed {seed}: No-Comm = {ret_no:.2f}, Full-Comm = {ret_fc:.2f}, Diff = {ret_fc - ret_no:+.2f}", flush=True)
