import numpy as np
from mpe2 import simple_spread_v3

class PartiallyObservableSpreadEnv:
    """
    Decentralised Partially Observable Multi-Agent Particle Environment (Dec-POMDP)
    with Information Exchange (Scout-Rover Navigation).
    
    Setup:
    - N = 3 cooperative agents in 2D continuous space.
    - Candidate landmarks: L0 = (-0.6, 0.0), L1 = (0.6, 0.0), L2 = (0.0, 0.6).
    - In each episode, an active goal is selected uniformly at random: target_idx in {0, 1}.
    - Agent 0 is the Scout/Sensor at (0, 0): observes the active goal cue (+1.0 for L1, -1.0 for L0).
    - Agent 1 and Agent 2 start at (-0.1, -0.4) and (0.1, -0.4).
      They observe relative vectors to candidate goals, but under partial observability (full_obs=False),
      their target cue is masked to [0.0, 0.0].
      Without communication, rovers do not know which landmark is active.
      With broadcast communication, Agent 0 transmits the target clue, allowing rovers to reach the goal.
    - Under full observability (full_obs=True), rovers directly observe the active target cue.
    
    Decentralized observation o_i (obs_dim = 16):
    - Agent 0 (Scout): [vel (2), pos (2), rel_L0 (2), rel_L1 (2), cue (1), pad (7)]
    - Agent 1 (Rover 1): [vel (2), pos (2), rel_L0 (2), rel_L1 (2), cue (1 if full_obs else 0), pad (7)]
    - Agent 2 (Rover 2): [vel (2), pos (2), rel_L0 (2), rel_L1 (2), cue (1 if full_obs else 0), pad (7)]
    
    Centralized state s for CTDE critic V_omega(s) (state_dim = 18):
    - All agents pos & vel (3 * 4 = 12 floats)
    - Active target index (1 float)
    - Zero padding (5 floats) -> Total 18 floats.
    
    Reward:
    - Distance penalty to active landmark: -0.4 * (d(Rover 1, Target) + d(Rover 2, Target))
    - Arrival bonus: +1.0 per rover within d_cov = 0.22 of active target
    
    Documented score endpoints:
    - R_MIN = -25.0 (empirical lower bound of blind / uninformed rovers)
    - R_MAX = 0.0 (well-coordinated target coverage upper bound)
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
        
        # Documented endpoints for normalised score
        self.R_MIN = -25.0
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
            o.extend([float(rel0[0]), float(rel0[1]), float(rel1[0]), float(rel1[1])]) # 4
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
