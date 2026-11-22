import torch
import numpy as np

class BatchedTrackingEnv:
    """
    Pure PyTorch Batched Vectorised Continuous Tracking Environment.
    Operates simultaneously across B parallel environments with zero Python loops over envs.
    
    Agents:
      - Agent 0: Scout / Sensor (stationary at origin (0, 0))
      - Agent 1: Rover 1 (mobile particle)
      - Agent 2: Rover 2 (mobile particle)
      
    Stages:
      - 'static' : Target position p_target ~ U([-0.8, 0.8]^2) is stationary for the episode.
      - 'dynamic': Target drifts continuously with smooth velocity.
    """
    def __init__(self, num_envs=64, max_cycles=25, stage='static', full_obs=False, device='cpu'):
        self.B = num_envs
        self.T_max = max_cycles
        self.stage = stage
        self.full_obs = full_obs
        self.device = torch.device(device)
        self.n_agents = 3
        
        # Physics constants
        self.dt = 0.1
        self.damping = 0.25
        self.max_speed = 1.0
        self.max_force = 1.0
        self.arena_bound = 1.0
        self.capture_radius = 0.20
        
        # Low-level PD controller gains
        self.Kp = 4.0
        self.Kd = 1.0
        
        # Observations:
        # Scout (Agent 0): [target_pos (2), target_vel (2), own_pos (2), own_vel (2), padding (8)] -> 16
        # Rovers (Agent 1, 2): [own_pos (2), own_vel (2), other_rover_pos (2), padding (10)] -> 16
        # In full_obs: Rovers also get [target_pos (2), target_vel (2)] -> 16
        self.obs_dim = 16
        self.state_dim = 18
        
        # State tensors [B, n_agents, 2]
        self.pos = torch.zeros(self.B, self.n_agents, 2, dtype=torch.float32, device=self.device)
        self.vel = torch.zeros(self.B, self.n_agents, 2, dtype=torch.float32, device=self.device)
        self.target_pos = torch.zeros(self.B, 2, dtype=torch.float32, device=self.device)
        self.target_vel = torch.zeros(self.B, 2, dtype=torch.float32, device=self.device)
        self.theta = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.step_count = torch.zeros(self.B, dtype=torch.int64, device=self.device)
        
        # Frozen endpoints
        self.R_MIN = 0.0
        self.R_MAX = 1.0

    def set_endpoints(self, r_min, r_max):
        self.R_MIN = float(r_min)
        self.R_MAX = float(r_max)

    def normalize_score_unclipped(self, r):
        denom = max(1e-6, self.R_MAX - self.R_MIN)
        return (r - self.R_MIN) / denom

    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.step_count.zero_()
        
        # Target positions in [-0.8, 0.8]^2
        self.target_pos = (torch.rand(self.B, 2, device=self.device) * 1.6) - 0.8
        
        if self.stage == 'dynamic':
            self.theta = torch.rand(self.B, device=self.device) * 2 * np.pi
            speed = 0.03
            self.target_vel = speed * torch.stack([torch.cos(self.theta), torch.sin(self.theta)], dim=-1)
        else:
            self.target_vel.zero_()
            
        # Agent positions:
        # Agent 0 at (0, 0)
        self.pos[:, 0, :] = 0.0
        # Rover 1 at (-0.2, -0.2), Rover 2 at (0.2, -0.2)
        self.pos[:, 1, 0] = -0.2; self.pos[:, 1, 1] = -0.2
        self.pos[:, 2, 0] =  0.2; self.pos[:, 2, 1] = -0.2
        self.vel.zero_()
        
        return self._get_obs(), self._get_state()

    def _get_obs(self):
        """
        Returns obs dict: {agent_0: [B, 16], agent_1: [B, 16], agent_2: [B, 16]}
        """
        obs = {}
        # Scout (Agent 0): sees continuous target position and velocity
        o0 = torch.zeros(self.B, self.obs_dim, device=self.device)
        o0[:, 0:2] = self.target_pos
        o0[:, 2:4] = self.target_vel
        o0[:, 4:6] = self.pos[:, 0]
        o0[:, 6:8] = self.vel[:, 0]
        obs['agent_0'] = o0
        
        # Rovers
        for i in [1, 2]:
            oi = torch.zeros(self.B, self.obs_dim, device=self.device)
            oi[:, 0:2] = self.pos[:, i]
            oi[:, 2:4] = self.vel[:, i]
            other = 2 if i == 1 else 1
            oi[:, 4:6] = self.pos[:, other] - self.pos[:, i]
            if self.full_obs:
                oi[:, 6:8] = self.target_pos
                oi[:, 8:10] = self.target_vel
            obs[f'agent_{i}'] = oi
            
        return obs

    def _get_state(self):
        """
        Centralized state for critic: [B, 18]
        All agents pos & vel (3 * 4 = 12) + target_pos (2) + target_vel (2) + pad (2)
        """
        s = torch.zeros(self.B, self.state_dim, device=self.device)
        s[:, 0:6] = self.pos.reshape(self.B, 6)
        s[:, 6:12] = self.vel.reshape(self.B, 6)
        s[:, 12:14] = self.target_pos
        s[:, 14:16] = self.target_vel
        return s

    def step_with_target_estimates(self, target_estimates):
        """
        Takes target_estimates: tensor [B, n_agents, 2] of estimated target positions p_hat.
        Low-level controller computes closed-form PD motor forces directly:
          F_i = clip(Kp * (p_hat_i - p_i) - Kd * v_i, -F_max, F_max)
        """
        forces = torch.zeros(self.B, self.n_agents, 2, device=self.device)
        for i in [1, 2]:
            p_hat = target_estimates[:, i] # [B, 2]
            p_i = self.pos[:, i]
            v_i = self.vel[:, i]
            f_i = self.Kp * (p_hat - p_i) - self.Kd * v_i
            forces[:, i] = torch.clamp(f_i, -self.max_force, self.max_force)
            
        return self.step_physics(forces)

    def step_physics(self, forces):
        """
        Batched vectorized 2D kinematics update across all B environments.
        """
        self.step_count += 1
        
        # Update velocities and positions for mobile rovers
        for i in [1, 2]:
            f = forces[:, i]
            new_v = (self.vel[:, i] + f * self.dt) * (1.0 - self.damping)
            self.vel[:, i] = torch.clamp(new_v, -self.max_speed, self.max_speed)
            new_p = self.pos[:, i] + self.vel[:, i] * self.dt
            self.pos[:, i] = torch.clamp(new_p, -self.arena_bound, self.arena_bound)
            
        # If dynamic, update target trajectory
        if self.stage == 'dynamic':
            noise = torch.randn(self.B, device=self.device) * 0.15
            self.theta += noise
            speed = 0.03
            self.target_vel = speed * torch.stack([torch.cos(self.theta), torch.sin(self.theta)], dim=-1)
            self.target_pos = torch.clamp(self.target_pos + self.target_vel, -0.85, 0.85)
            
        # Compute rewards vectorized across B:
        # d1: [B], d2: [B]
        d1 = torch.norm(self.pos[:, 1] - self.target_pos, dim=-1)
        d2 = torch.norm(self.pos[:, 2] - self.target_pos, dim=-1)
        
        # Step reward: distance penalty + arrival bonuses
        r = -0.5 * (d1 + d2)
        r = r + 0.5 * (d1 <= self.capture_radius).float()
        r = r + 0.5 * (d2 <= self.capture_radius).float()
        
        done = (self.step_count >= self.T_max)
        
        return self._get_obs(), self._get_state(), r, done, {}
