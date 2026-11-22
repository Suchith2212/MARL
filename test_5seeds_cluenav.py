import numpy as np
import torch
import torch.nn as nn
from mpe2 import simple_spread_v3
from src.channel.broadcast_channel import BroadcastChannel
from src.algos.mappo_trainer import MAPPOTrainer
from src.utils.metrics import compute_paired_ci_95

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PartiallyObservableSpreadEnv:
    """
    Decentralised Partially Observable Target Navigation (Dec-POMDP with Information Exchange).
    - 3 cooperative agents in 2D continuous space.
    - Candidate landmarks: L0 = (-0.6, 0.0), L1 = (0.6, 0.0), L2 = (0.0, 0.6).
    - In each episode, an active goal is selected uniformly at random: target_idx in {0, 1}.
    - Agent 0 is the Scout/Sensor at (0, 0): observes the active goal cue (+1.0 for L1, -1.0 for L0).
    - Agent 1 and Agent 2 start at (-0.1, -0.4) and (0.1, -0.4).
      They observe relative vectors to candidate goals, but under partial observability (full_obs=False),
      their target cue is masked to [0.0, 0.0].
      Without communication, rovers do not know which landmark is active.
      With broadcast communication, Agent 0 transmits the target clue, allowing rovers to reach the goal.
    - Under full observability (full_obs=True), rovers directly observe the active target cue.
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
        
        # Documented endpoints for normalised score
        self.R_MIN = -20.0
        self.R_MAX = 0.0

    def set_endpoints(self, r_min, r_max):
        self.R_MIN = float(r_min)
        self.R_MAX = float(r_max)

    def normalize_score(self, return_val):
        denom = max(1e-6, self.R_MAX - self.R_MIN)
        score = (return_val - self.R_MIN) / denom
        return float(np.clip(score, 0.0, 1.0))

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        raw = self.env.unwrapped
        rng = np.random.RandomState(seed)
        for a in raw.world.agents:
            a.collide = False
            a.max_speed = 1.0
            
        self.target_idx = rng.choice([0, 1])
        self.lm_pos = [np.array([-0.6, 0.0]), np.array([0.6, 0.0]), np.array([0.0, 0.6])]
        for k in range(3):
            raw.world.landmarks[k].state.p_pos = self.lm_pos[k].copy()
            
        raw.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        raw.world.agents[1].state.p_pos = np.array([-0.1, -0.4])
        raw.world.agents[2].state.p_pos = np.array([ 0.1, -0.4])
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
            rel0 = self.lm_pos[0] - a.state.p_pos # 2
            rel1 = self.lm_pos[1] - a.state.p_pos # 2
            o.extend([float(rel0[0]), float(rel0[1]), float(rel1[0]), float(rel1[1])])
            if i == 0:
                cue = 1.0 if self.target_idx == 1 else -1.0
                o.extend([cue, 0.0]) # 2
                o.extend([0.0] * 6)  # pad to 16
            else:
                if self.full_obs:
                    cue = 1.0 if self.target_idx == 1 else -1.0
                    o.extend([cue, 0.0])
                else:
                    o.extend([0.0, 0.0])
                o.extend([0.0] * 6)
            obs_dict[f"agent_{i}"] = np.array(o, dtype=np.float32)
        return obs_dict

    def _get_state(self):
        raw = self.env.unwrapped
        s = []
        for a in raw.world.agents:
            s.extend(a.state.p_pos); s.extend(a.state.p_vel) # 12
        s.append(float(self.target_idx)) # 1
        s.extend([0.0] * 5) # pad to 18
        return np.array(s, dtype=np.float32)

    def step(self, actions):
        acts = dict(actions)
        acts['agent_0'] = 0 # scout is stationary
        self.env.step(acts)
        raw = self.env.unwrapped
        raw.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        raw.world.agents[0].state.p_vel = np.zeros(2)
        
        target = self.lm_pos[self.target_idx]
        p1 = raw.world.agents[1].state.p_pos
        p2 = raw.world.agents[2].state.p_pos
        d1 = np.linalg.norm(p1 - target)
        d2 = np.linalg.norm(p2 - target)
        
        r = -0.4 * (d1 + d2)
        if d1 <= 0.22: r += 1.0
        if d2 <= 0.22: r += 1.0
        return self._extract_obs(), self._get_state(), float(r), False, {}

def eval_policy(env, channel, trainer, test_seeds, mode='full_comm'):
    rets = []
    c_disc_list = []
    c_bar_list = []
    for s in test_seeds:
        obs, state = env.reset(seed=s)
        if channel: channel.reset()
        ep_r = 0.0
        for t in range(env.max_cycles):
            h, m, g, _, b, _ = trainer.act(obs, deterministic=True)
            if mode == 'full_comm':
                # Scout broadcasts the target cue; rovers listen
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
        if channel and mode == 'full_comm':
            cd, cb = channel.get_episode_metrics()
            c_disc_list.append(cd)
            c_bar_list.append(cb)
    return {
        'mean_return': float(np.mean(rets)),
        'std_return': float(np.std(rets)),
        'mean_c_disc': float(np.mean(c_disc_list)) if c_disc_list else 0.0,
        'mean_c_bar': float(np.mean(c_bar_list)) if c_bar_list else 0.0,
        'returns': rets
    }

if __name__ == '__main__':
    test_seeds = [1000 + i * 17 for i in range(20)]
    env = PartiallyObservableSpreadEnv()
    channel = BroadcastChannel(n_agents=3, message_dim=16)

    # 1. Random Lower Bound
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
    print(f"Random Lower Bound (R_MIN): {R_MIN:.2f} (std: {np.std(r_rets):.2f})", flush=True)

    # 2. Full-Observation Upper Bound across 5 seeds
    fo_rets = []
    env_fo = PartiallyObservableSpreadEnv(full_obs=True)
    for seed in [42, 43, 44, 45, 46]:
        set_seed(seed)
        tr_fo = MAPPOTrainer(env_fo.obs_dim, env_fo.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=5e-4, critic_lr=1e-3)
        for update in range(25):
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
        res_fo = eval_policy(env_fo, None, tr_fo, test_seeds, mode='no_comm')
        fo_rets.append(res_fo['mean_return'])
    R_MAX = float(np.mean(fo_rets))
    print(f"Full-Obs Upper Bound (R_MAX): {R_MAX:.2f} (std: {np.std(fo_rets):.2f})", flush=True)

    env.set_endpoints(R_MIN, R_MAX)
    print(f"\nDocumented Score Endpoints: [R_MIN={R_MIN:.2f}, R_MAX={R_MAX:.2f}]", flush=True)

    # 3. Train No-Comm vs Full-Comm across 5 seeds
    seeds = [42, 43, 44, 45, 46]
    no_results = []
    fc_results = []
    
    print("\n--- Training No-Comm and Full-Comm Across 5 Seeds ---", flush=True)
    for seed in seeds:
        # A. No-Comm
        set_seed(seed)
        tr_no = MAPPOTrainer(env.obs_dim, env.state_dim, mode='no_comm', fixed_bit_width=0, actor_lr=5e-4, critic_lr=1e-3)
        for update in range(25):
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
        res_no = eval_policy(env, None, tr_no, test_seeds, mode='no_comm')
        res_no['norm_score'] = env.normalize_score(res_no['mean_return'])
        no_results.append(res_no)

        # B. Full-Comm (b=8)
        set_seed(seed)
        tr_fc = MAPPOTrainer(env.obs_dim, env.state_dim, mode='full_comm_fixed', fixed_bit_width=8, actor_lr=5e-4, critic_lr=1e-3)
        for update in range(25):
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
        res_fc = eval_policy(env, channel, tr_fc, test_seeds, mode='full_comm')
        res_fc['norm_score'] = env.normalize_score(res_fc['mean_return'])
        fc_results.append(res_fc)
        
        diff_raw = res_fc['mean_return'] - res_no['mean_return']
        diff_norm = res_fc['norm_score'] - res_no['norm_score']
        print(f"Seed {seed}: No-Comm Return = {res_no['mean_return']:.2f} (score: {res_no['norm_score']:.3f}), "
              f"Full-Comm Return = {res_fc['mean_return']:.2f} (score: {res_fc['norm_score']:.3f}), "
              f"Score Diff = {diff_norm:+.3f} (raw diff: {diff_raw:+.2f})", flush=True)

    # 4. Statistical Analysis: Paired 95% Confidence Interval
    diffs_norm = [fc_results[i]['norm_score'] - no_results[i]['norm_score'] for i in range(5)]
    m_diff, margin, ci_low, ci_high = compute_paired_ci_95(diffs_norm)

    print("\n=======================================================", flush=True)
    print("PHASE 1 SCREENING CHECK 1: FINAL STATISTICAL REPORT", flush=True)
    print("=======================================================", flush=True)
    print(f"Evaluation Test Suite: 20 fixed held-out seeds ({test_seeds[:3]} ... {test_seeds[-1]})")
    print(f"Independent Training Seeds (n=5): {seeds}")
    print(f"Documented Endpoints: R_MIN = {R_MIN:.4f}, R_MAX = {R_MAX:.4f}")
    print(f"No-Comm Returns: {[round(r['mean_return'], 2) for r in no_results]} -> Mean: {np.mean([r['mean_return'] for r in no_results]):.2f}")
    print(f"Full-Comm Returns: {[round(r['mean_return'], 2) for r in fc_results]} -> Mean: {np.mean([r['mean_return'] for r in fc_results]):.2f}")
    print(f"No-Comm Normalised Scores: {[round(r['norm_score'], 4) for r in no_results]} -> Mean: {np.mean([r['norm_score'] for r in no_results]):.4f}")
    print(f"Full-Comm Normalised Scores: {[round(r['norm_score'], 4) for r in fc_results]} -> Mean: {np.mean([r['norm_score'] for r in fc_results]):.4f}")
    print(f"Paired Normalised Differences (Delta_s): {[round(x, 4) for x in diffs_norm]}")
    print(f"Mean Difference (Delta_bar): {m_diff:.4f}")
    print(f"Paired 95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Pass Criterion: Mean Diff >= 0.10 AND CI lower bound > 0.0")
    if m_diff >= 0.10 and ci_low > 0.0:
        print(">>> CHECK 1 STATUS: PASSED <<<")
    else:
        print(">>> CHECK 1 STATUS: FAILED <<<")
