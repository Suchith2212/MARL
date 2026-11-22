import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpe2 import simple_spread_v3
from src.channel.broadcast_channel import BroadcastChannel
from src.quantiser.uniform_quantiser import UniformQuantiser
from src.utils.metrics import compute_paired_ci_95

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CleanGuideRoverEnv:
    def __init__(self, full_obs=False):
        self.n_agents = 3
        self.max_cycles = 25
        self.full_obs = full_obs
        self.env = simple_spread_v3.parallel_env(N=3, continuous_actions=False, max_cycles=25)
        self.agents = [f'agent_{i}' for i in range(3)]
        self.action_dim = 5
        self.obs_dim = 16
        self.state_dim = 18

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        raw = self.env.unwrapped
        rng = np.random.RandomState(seed)
        for a in raw.world.agents:
            a.collide = False
            a.max_speed = 1.0
            
        self.c1 = rng.choice([0, 1])
        self.c2 = rng.choice([0, 1])
        t1_opts = [np.array([-0.6, 0.5]), np.array([0.6, -0.5])]
        t2_opts = [np.array([0.6, 0.5]), np.array([-0.6, -0.5])]
        self.t1 = t1_opts[self.c1]
        self.t2 = t2_opts[self.c2]
        
        raw.world.landmarks[0].state.p_pos = self.t1.copy()
        raw.world.landmarks[1].state.p_pos = self.t2.copy()
        raw.world.landmarks[2].state.p_pos = np.array([2.5, 2.5]) # outside arena
        
        raw.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        raw.world.agents[1].state.p_pos = np.array([-0.05, 0.0])
        raw.world.agents[2].state.p_pos = np.array([ 0.05, 0.0])
        for a in raw.world.agents: a.state.p_vel = np.zeros(2)
        return self._extract_obs(), self._get_state()

    def _extract_obs(self):
        raw = self.env.unwrapped
        obs_dict = {}
        for i, a in enumerate(raw.world.agents):
            o = []
            o.extend(a.state.p_vel); o.extend(a.state.p_pos) # 4
            if i == 0:
                s1 = 1.0 if self.c1 == 1 else -1.0
                s2 = 1.0 if self.c2 == 1 else -1.0
                o.extend([s1, s2, float(self.t1[0]), float(self.t1[1]), float(self.t2[0]), float(self.t2[1])]) # 6
                o.extend([0.0] * 6) # 16
            elif i == 1:
                if self.full_obs:
                    s1 = 1.0 if self.c1 == 1 else -1.0
                    o.extend([s1, 0.0, float(self.t1[0]), float(self.t1[1]), 0.0, 0.0])
                else:
                    o.extend([0.0] * 6)
                o.extend([0.0] * 6)
            else:
                if self.full_obs:
                    s2 = 1.0 if self.c2 == 1 else -1.0
                    o.extend([0.0, s2, 0.0, 0.0, float(self.t2[0]), float(self.t2[1])])
                else:
                    o.extend([0.0] * 6)
                o.extend([0.0] * 6)
            obs_dict[f'agent_{i}'] = np.array(o, dtype=np.float32)
        return obs_dict

    def _get_state(self):
        raw = self.env.unwrapped
        s = []
        for a in raw.world.agents: s.extend(a.state.p_pos); s.extend(a.state.p_vel)
        s.extend(self.t1); s.extend(self.t2); s.extend([0.0, 0.0])
        return np.array(s, dtype=np.float32)

    def step(self, actions):
        acts = dict(actions)
        acts['agent_0'] = 0 # Guide is stationary observer
        self.env.step(acts)
        raw = self.env.unwrapped
        raw.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        raw.world.agents[0].state.p_vel = np.zeros(2)
        for a in raw.world.agents:
            a.state.p_pos = np.clip(a.state.p_pos, -1.2, 1.2)
            
        p1 = raw.world.agents[1].state.p_pos
        p2 = raw.world.agents[2].state.p_pos
        d1 = np.linalg.norm(p1 - self.t1)
        d2 = np.linalg.norm(p2 - self.t2)
        r = -0.4 * (d1 + d2)
        if d1 <= 0.22: r += 1.0
        if d2 <= 0.22: r += 1.0
        return self._extract_obs(), self._get_state(), float(r), False, {}

def eval_policy(env, channel, trainer, test_seeds, mode='full_comm'):
    rets = []
    for s in test_seeds:
        obs, state = env.reset(seed=s)
        if channel: channel.reset()
        ep_r = 0.0
        for t in range(env.max_cycles):
            h, m, g, _, b, _ = trainer.act(obs, deterministic=True)
            if mode == 'full_comm':
                # In full_comm, Agent 0 broadcasts the guide message
                g_eval = [1, 0, 0]
                b_eval = [8, 0, 0]
                q_msgs = [trainer.quantiser(m[i:i+1], bit_width=b_eval[i]).squeeze(0) if g_eval[i]==1 else torch.zeros(16) for i in range(3)]
                delivered, _ = channel.transmit_and_deliver(q_msgs, g_eval, b_eval)
                acts, _ = trainer.get_actions(h, delivered, deterministic=True)
            else:
                acts, _ = trainer.get_actions(h, {0:[], 1:[], 2:[]}, deterministic=True)
            obs, state, r, _, _ = env.step(acts)
            ep_r += r
        rets.append(ep_r)
    return float(np.mean(rets))

if __name__ == '__main__':
    from src.algos.mappo_trainer import MAPPOTrainer
    
    test_seeds = [1000 + i * 17 for i in range(20)]
    env = CleanGuideRoverEnv()
    channel = BroadcastChannel(n_agents=3, message_dim=16)

    # 1. Random Policy Baseline
    r_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(25):
            acts = {a: np.random.randint(0, 5) for a in env.agents}
            _, _, r, _, _ = env.step(acts)
            ep_r += r
        r_rets.append(ep_r)
    R_MIN = float(np.mean(r_rets))
    print(f"Random Lower Bound (R_MIN): {R_MIN:.2f}", flush=True)

    # 2. Full-Observation Upper Bound Baseline across 5 seeds
    fo_rets = []
    env_fo = CleanGuideRoverEnv(full_obs=True)
    for seed in [42, 43, 44, 45, 46]:
        set_seed(seed)
        tr_fo = MAPPOTrainer(env_fo.obs_dim, env_fo.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=5e-4, critic_lr=1e-3)
        for update in range(30):
            b_obs, b_states, b_acts, b_act_lps = [], [], [], []
            b_gates, b_glps, b_bws, b_wlps = [], [], [], []
            b_masks, b_rews, b_costs, b_dones = [], [], [], []
            for ep in range(12):
                obs, state = env_fo.reset(seed=seed * 1000 + update * 12 + ep)
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
        ret_fo = eval_policy(env_fo, None, tr_fo, test_seeds, mode='no_comm')
        fo_rets.append(ret_fo)
    R_MAX = float(np.mean(fo_rets))
    print(f"Full-Obs Upper Bound (R_MAX): {R_MAX:.2f} (std: {np.std(fo_rets):.2f})", flush=True)

    # 3. Train No-Comm vs Full-Comm across 5 seeds
    seeds = [42, 43, 44, 45, 46]
    no_rets = []
    fc_rets = []
    
    print("\n--- Training No-Comm and Full-Comm Across 5 Seeds ---", flush=True)
    for seed in seeds:
        # A. No-Comm
        set_seed(seed)
        tr_no = MAPPOTrainer(env.obs_dim, env.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=5e-4, critic_lr=1e-3)
        for update in range(30):
            b_obs, b_states, b_acts, b_act_lps = [], [], [], []
            b_gates, b_glps, b_bws, b_wlps = [], [], [], []
            b_masks, b_rews, b_costs, b_dones = [], [], [], []
            for ep in range(12):
                obs, state = env.reset(seed=seed * 1000 + update * 12 + ep)
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
        ret_no = eval_policy(env, None, tr_no, test_seeds, mode='no_comm')
        no_rets.append(ret_no)

        # B. Full-Comm (b=8)
        set_seed(seed)
        tr_fc = MAPPOTrainer(env.obs_dim, env.state_dim, mode='full_comm_fixed', fixed_bit_width=8, actor_lr=5e-4, critic_lr=1e-3)
        for update in range(30):
            b_obs, b_states, b_acts, b_act_lps = [], [], [], []
            b_gates, b_glps, b_bws, b_wlps = [], [], [], []
            b_masks, b_rews, b_costs, b_dones = [], [], [], []
            for ep in range(12):
                obs, state = env.reset(seed=seed * 1000 + update * 12 + ep)
                channel.reset()
                for t in range(25):
                    h, m, g, glp, b, blp = tr_fc.act(obs, deterministic=False)
                    g_train = [1, 0, 0]
                    b_train = [8, 0, 0]
                    q_msgs = [tr_fc.quantiser(m[i:i+1], bit_width=b_train[i]).squeeze(0) if g_train[i]==1 else torch.zeros(16) for i in range(3)]
                    delivered, step_cost = channel.transmit_and_deliver(q_msgs, g_train, b_train)
                    acts, alps = tr_fc.get_actions(h, delivered, deterministic=False)
                    mask_matrix = [[0]*3 for _ in range(3)]
                    for r_idx, items in delivered.items():
                        for it in items: mask_matrix[it['sender']][r_idx] = 1
                    nobs, nstate, r, d, _ = env.step(acts)
                    b_obs.append([obs[f'agent_{i}'] for i in range(3)])
                    b_states.append(state)
                    b_acts.append([acts[f'agent_{i}'] for i in range(3)])
                    b_act_lps.append([alps[f'agent_{i}'] for i in range(3)])
                    b_gates.append(g_train); b_glps.append([0.0]*3)
                    b_bws.append(b_train); b_wlps.append([0.0]*3)
                    b_masks.append(mask_matrix)
                    b_rews.append(r); b_costs.append(step_cost)
                    b_dones.append(1.0 if t==24 else 0.0)
                    obs, state = nobs, nstate
            tr_fc.train_step({
                'obs': b_obs, 'state': b_states, 'actions': b_acts, 'action_lps': b_act_lps,
                'gates': b_gates, 'gate_lps': b_glps, 'bit_widths': b_bws, 'width_lps': b_wlps,
                'deliv_masks': b_masks, 'rewards': b_rews, 'step_costs': b_costs, 'dones': b_dones
            })
        ret_fc = eval_policy(env, channel, tr_fc, test_seeds, mode='full_comm')
        fc_rets.append(ret_fc)
        print(f"Seed {seed}: No-Comm = {ret_no:.2f}, Full-Comm = {ret_fc:.2f}, Diff = {ret_fc - ret_no:+.2f}", flush=True)

    # Compute Normalised Scores and Paired CI
    denom = max(1e-6, R_MAX - R_MIN)
    norm_no = [(r - R_MIN) / denom for r in no_rets]
    norm_fc = [(r - R_MIN) / denom for r in fc_rets]
    diffs = [norm_fc[i] - norm_no[i] for i in range(5)]
    m_diff, margin, ci_low, ci_high = compute_paired_ci_95(diffs)
    
    print("\n--- PHASE 1 CHECK 1 SUMMARY ---", flush=True)
    print(f"R_MIN (Random): {R_MIN:.2f}, R_MAX (Full-Obs): {R_MAX:.2f}", flush=True)
    print(f"No-Comm Returns: {[round(x, 2) for x in no_rets]} -> Mean: {np.mean(no_rets):.2f}", flush=True)
    print(f"Full-Comm Returns: {[round(x, 2) for x in fc_rets]} -> Mean: {np.mean(fc_rets):.2f}", flush=True)
    print(f"Normalised Scores (No-Comm): {[round(x, 4) for x in norm_no]} -> Mean: {np.mean(norm_no):.4f}", flush=True)
    print(f"Normalised Scores (Full-Comm): {[round(x, 4) for x in norm_fc]} -> Mean: {np.mean(norm_fc):.4f}", flush=True)
    print(f"Paired Normalised Differences: {[round(x, 4) for x in diffs]}", flush=True)
    print(f"Mean Difference: {m_diff:.4f}, 95% Paired CI: [{ci_low:.4f}, {ci_high:.4f}]", flush=True)
    if m_diff >= 0.10 and ci_low > 0.0:
        print("CHECK 1 RESULT: PASSED (Mean diff >= 0.10 and CI excludes zero)", flush=True)
    else:
        print("CHECK 1 RESULT: FAILED", flush=True)
