import numpy as np
import torch
from mpe2 import simple_spread_v3
from src.channel.broadcast_channel import BroadcastChannel
from src.algos.mappo_trainer import MAPPOTrainer
from src.utils.metrics import compute_paired_ci_95
from scripts.run_phase1_screen import set_seed

class ActiveTargetMPEEnv:
    """
    Decentralised Partially Observable Target Guidance Task.
    - N = 3 agents.
    - 3 distinct physical landmarks at fixed locations:
      L0 = (-0.6, -0.4), L1 = (0.6, -0.4), L2 = (0.0, 0.6).
    - In each episode, one landmark L_{target} is chosen at random as the active goal.
    - Agent 0 is the scout at (0, 0) who observes the active target index.
    - Agents 1 and 2 are field rovers starting near center (0, 0).
      Under partial observation: rovers see all 3 physical landmarks, but the active target
      clue is masked to [0, 0, 0].
    - Under full observation: rovers directly observe the active target clue [1, 0, 0].
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
        
        self.lm_pos = [
            np.array([-0.6, -0.4]),
            np.array([ 0.6, -0.4]),
            np.array([ 0.0,  0.6])
        ]
        for k in range(3):
            raw.world.landmarks[k].state.p_pos = self.lm_pos[k].copy()
            
        # Secret active target
        self.target_idx = rng.randint(0, 3)
        self.target_pos = self.lm_pos[self.target_idx]
        
        raw.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        raw.world.agents[1].state.p_pos = np.array([-0.1, 0.0])
        raw.world.agents[2].state.p_pos = np.array([ 0.1, 0.0])
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
            for k in range(3):
                rel = self.lm_pos[k] - a.state.p_pos
                o.extend([float(rel[0]), float(rel[1])]) # 6
            if self.full_obs or (i == 0):
                clue = [1.0 if k == self.target_idx else 0.0 for k in range(3)]
            else:
                clue = [0.0, 0.0, 0.0]
            o.extend(clue) # 3
            o.extend([0.0, 0.0, 0.0]) # pad to 16
            obs_dict[f"agent_{i}"] = np.array(o, dtype=np.float32)
        return obs_dict

    def _get_state(self):
        raw = self.env.unwrapped
        s = []
        for a in raw.world.agents:
            s.extend(a.state.p_pos); s.extend(a.state.p_vel)
        s.extend(self.target_pos)
        s.extend([float(self.target_idx), 0.0, 0.0, 0.0])
        return np.array(s, dtype=np.float32)

    def step(self, actions):
        acts = dict(actions)
        acts['agent_0'] = 0
        self.env.step(acts)
        raw = self.env.unwrapped
        raw.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        raw.world.agents[0].state.p_vel = np.zeros(2)
        
        p1 = raw.world.agents[1].state.p_pos
        p2 = raw.world.agents[2].state.p_pos
        d1 = np.linalg.norm(p1 - self.target_pos)
        d2 = np.linalg.norm(p2 - self.target_pos)
        
        r = -0.4 * (d1 + d2)
        if d1 <= 0.22: r += 1.0
        if d2 <= 0.22: r += 1.0
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
    env = ActiveTargetMPEEnv()
    channel = BroadcastChannel(n_agents=3, message_dim=16)

    # 1. Random
    r_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(25):
            _, _, r, _, _ = env.step({a: np.random.randint(0, 5) for a in env.agents})
            ep_r += r
        r_rets.append(ep_r)
    print(f"Random: {np.mean(r_rets):.2f}", flush=True)

    # 2. Full-Obs Upper Bound
    env_fo = ActiveTargetMPEEnv(full_obs=True)
    set_seed(42)
    tr_fo = MAPPOTrainer(env_fo.obs_dim, env_fo.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=1e-3, critic_lr=2e-3)
    for update in range(30):
        b_obs, b_states, b_acts, b_act_lps = [], [], [], []
        b_gates, b_glps, b_bws, b_wlps = [], [], [], []
        b_masks, b_rews, b_costs, b_dones = [], [], [], []
        for ep in range(6):
            obs, state = env_fo.reset(seed=42 * 1000 + update * 10 + ep)
            for t in range(25):
                h, m, g, glp, b, blp = tr_fo.act(obs, deterministic=False)
                acts, alps = tr_fo.get_actions(h, {0:[], 1:[], 2:[]}, deterministic=False)
                nobs, nstate, r, d, _ = env_fo.step(acts)
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
        tr_fo.train_step({
            'obs': b_obs, 'state': b_states, 'actions': b_acts, 'action_lps': b_act_lps,
            'gates': b_gates, 'gate_lps': b_glps, 'bit_widths': b_bws, 'width_lps': b_wlps,
            'deliv_masks': b_masks, 'rewards': b_rews, 'step_costs': b_costs, 'dones': b_dones
        })
    ret_fo = eval_policy(env_fo, None, tr_fo, test_seeds)
    print(f"Full-Obs: {ret_fo:.2f}", flush=True)

    # 3. No-Comm Baseline
    set_seed(42)
    tr_no = MAPPOTrainer(env.obs_dim, env.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=1e-3, critic_lr=2e-3)
    for update in range(30):
        b_obs, b_states, b_acts, b_act_lps = [], [], [], []
        b_gates, b_glps, b_bws, b_wlps = [], [], [], []
        b_masks, b_rews, b_costs, b_dones = [], [], [], []
        for ep in range(6):
            obs, state = env.reset(seed=42 * 1000 + update * 10 + ep)
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
    print(f"No-Comm: {ret_no:.2f}", flush=True)

    # 4. Full-Comm Baseline
    set_seed(42)
    tr_fc = MAPPOTrainer(env.obs_dim, env.state_dim, mode='full_comm_fixed', fixed_bit_width=8, actor_lr=1e-3, critic_lr=2e-3)
    for update in range(30):
        b_obs, b_states, b_acts, b_act_lps = [], [], [], []
        b_gates, b_glps, b_bws, b_wlps = [], [], [], []
        b_masks, b_rews, b_costs, b_dones = [], [], [], []
        for ep in range(6):
            obs, state = env.reset(seed=42 * 1000 + update * 10 + ep)
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
    print(f"Full-Comm: {ret_fc:.2f}", flush=True)
    print(f"Diff (Full - No): {ret_fc - ret_no:+.2f}", flush=True)
