import numpy as np
from mpe2 import simple_spread_v3

class ContinuousTrackingEnv:
    """
    Decentralised POMDP with Continuous, Time-Varying Target Tracking.
    - Scout (Agent 0) observes continuous target pos/vel.
    - Rovers (Agents 1, 2) must track the dynamic target under partial observability.
    """
    def __init__(self, n_agents=3, max_cycles=25, full_obs=False):
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.full_obs = full_obs
        self.env = simple_spread_v3.parallel_env(
            N=n_agents,
            local_ratio=0.5,
            max_cycles=max_cycles,
            continuous_actions=False
        )
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.action_dim = 5
        self.obs_dim = 16
        self.state_dim = 18
        
        self.R_MIN = -25.0
        self.R_MAX = 0.0
        self.t = 0
        
    def set_endpoints(self, r_min, r_max):
        self.R_MIN = float(r_min)
        self.R_MAX = float(r_max)
        
    def normalize_score(self, return_val):
        denom = max(1e-6, self.R_MAX - self.R_MIN)
        score = (return_val - self.R_MIN) / denom
        return float(np.clip(score, 0.0, 1.0))

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.t = 0
        raw = self.env.unwrapped
        rng = np.random.RandomState(seed)
        for a in raw.world.agents:
            a.collide = False
            a.max_speed = 1.0
            
        # Target trajectory parameters
        self.target_pos = rng.uniform(-0.5, 0.5, size=2)
        self.theta = rng.uniform(0, 2 * np.pi)
        self.speed = 0.035
        self.target_vel = self.speed * np.array([np.cos(self.theta), np.sin(self.theta)])
        
        # Place landmarks
        raw.world.landmarks[0].state.p_pos = self.target_pos.copy()
        raw.world.landmarks[1].state.p_pos = np.array([2.5, 2.5])
        raw.world.landmarks[2].state.p_pos = np.array([2.5, 2.5])
        
        # Agents initial positions
        raw.world.agents[0].state.p_pos = np.array([0.0, 0.0]) # Scout at origin
        raw.world.agents[1].state.p_pos = np.array([-0.2, -0.2])
        raw.world.agents[2].state.p_pos = np.array([ 0.2, -0.2])
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
                # Scout observes exact continuous target pos and vel
                o.extend(self.target_pos) # 2
                o.extend(self.target_vel) # 2
                o.extend([0.0] * 8)      # pad to 16
            else:
                if self.full_obs:
                    rel_target = self.target_pos - a.state.p_pos
                    o.extend(rel_target)      # 2
                    o.extend(self.target_vel) # 2
                else:
                    o.extend([0.0, 0.0, 0.0, 0.0])
                # other agent relative pos
                other_idx = 2 if i == 1 else 1
                rel_other = raw.world.agents[other_idx].state.p_pos - a.state.p_pos
                o.extend(rel_other) # 2
                o.extend([0.0] * 6) # pad to 16
            obs_dict[f"agent_{i}"] = np.array(o, dtype=np.float32)
        return obs_dict

    def _get_state(self):
        raw = self.env.unwrapped
        s = []
        for a in raw.world.agents:
            s.extend(a.state.p_pos); s.extend(a.state.p_vel) # 12
        s.extend(self.target_pos) # 2
        s.extend(self.target_vel) # 2
        s.extend([0.0] * 2)       # pad to 18
        return np.array(s, dtype=np.float32)

    def step(self, actions):
        self.t += 1
        acts = dict(actions)
        acts['agent_0'] = 0 # scout is stationary sensor
        self.env.step(acts)
        raw = self.env.unwrapped
        raw.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        raw.world.agents[0].state.p_vel = np.zeros(2)
        
        # Update dynamic target position
        rng = np.random.RandomState(int(self.target_pos[0] * 10000 + self.t) % 2**31)
        self.theta += rng.normal(0.0, 0.2)
        self.target_vel = self.speed * np.array([np.cos(self.theta), np.sin(self.theta)])
        self.target_pos = np.clip(self.target_pos + self.target_vel, -0.85, 0.85)
        raw.world.landmarks[0].state.p_pos = self.target_pos.copy()
        
        p1 = raw.world.agents[1].state.p_pos
        p2 = raw.world.agents[2].state.p_pos
        d1 = float(np.linalg.norm(p1 - self.target_pos))
        d2 = float(np.linalg.norm(p2 - self.target_pos))
        
        r = -0.5 * (d1 + d2)
        if d1 <= 0.15: r += 0.5
        if d2 <= 0.15: r += 0.5
        return self._extract_obs(), self._get_state(), float(r), self.t >= self.max_cycles, {}

def get_proportional_action(agent_pos, target_pos):
    """
    Computes discrete action (0: noop, 1: left, 2: right, 3: down, 4: up)
    to move from agent_pos toward target_pos.
    """
    diff = target_pos - agent_pos
    dx, dy = diff[0], diff[1]
    if np.linalg.norm(diff) < 0.05:
        return 0 # noop
    if abs(dx) > abs(dy):
        return 1 if dx < 0 else 2 # 1: left, 2: right
    else:
        return 3 if dy < 0 else 4 # 3: down, 4: up

def evaluate_baselines():
    test_seeds = [1000 + i * 17 for i in range(20)]
    env = ContinuousTrackingEnv(n_agents=3, max_cycles=25)
    
    # 1. Random Policy
    rand_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(env.max_cycles):
            acts = {f"agent_{i}": np.random.randint(0, 5) for i in range(3)}
            _, _, r, done, _ = env.step(acts)
            ep_r += r
            if done: break
        rand_rets.append(ep_r)
        
    # 2. Scripted No-Information Heuristic: Moves to (0, 0) and stays there
    heur_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(env.max_cycles):
            raw = env.env.unwrapped
            acts = {'agent_0': 0}
            for i in [1, 2]:
                pos = raw.world.agents[i].state.p_pos
                acts[f"agent_{i}"] = get_proportional_action(pos, np.array([0.0, 0.0]))
            _, _, r, done, _ = env.step(acts)
            ep_r += r
            if done: break
        heur_rets.append(ep_r)

    # 3. Scripted Oracle: Has access to exact target_pos at each step
    oracle_rets = []
    for s in test_seeds:
        env.reset(seed=s)
        ep_r = 0.0
        for _ in range(env.max_cycles):
            raw = env.env.unwrapped
            acts = {'agent_0': 0}
            for i in [1, 2]:
                pos = raw.world.agents[i].state.p_pos
                acts[f"agent_{i}"] = get_proportional_action(pos, env.target_pos)
            _, _, r, done, _ = env.step(acts)
            ep_r += r
            if done: break
        oracle_rets.append(ep_r)
        
    print(f"Random Policy Return      : {np.mean(rand_rets):.2f} +- {np.std(rand_rets):.2f}")
    print(f"Scripted Heuristic (Blind): {np.mean(heur_rets):.2f} +- {np.std(heur_rets):.2f}")
    print(f"Scripted Oracle (Full-Obs): {np.mean(oracle_rets):.2f} +- {np.std(oracle_rets):.2f}")

if __name__ == '__main__':
    evaluate_baselines()
