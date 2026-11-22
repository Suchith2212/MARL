import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RunningMeanStd:
    """Tracks running mean and variance for observation/value normalization."""
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

    def denormalize(self, x):
        return x * np.sqrt(self.var + 1e-8) + self.mean

class UniformQuantiserSTE(nn.Module):
    """Uniform quantiser with Straight-Through Estimator (STE)."""
    def __init__(self, x_min=-1.0, x_max=1.0):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x, bit_width):
        if bit_width <= 0:
            return torch.zeros_like(x)
        num_levels = 2 ** int(bit_width)
        x_clamped = torch.clamp(x, self.x_min, self.x_max)
        delta = (self.x_max - self.x_min) / (num_levels - 1)
        k = torch.round((x_clamped - self.x_min) / delta)
        q = self.x_min + k * delta
        return x + (q - x).detach()

class ScoutEncoder(nn.Module):
    """Small MLP Encoder for Scout: maps continuous observation to 16-dim latent."""
    def __init__(self, obs_dim=16, msg_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, msg_dim),
            nn.Tanh()
        )
    def forward(self, obs):
        return self.net(obs)

class RoverDecoderPolicy(nn.Module):
    """Small MLP Decoder / Policy for Rover: outputs mean and log_std for estimated target."""
    def __init__(self, in_dim=22, act_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def forward(self, x):
        feat = self.net(x)
        mu = torch.tanh(self.mu_head(feat)) # target in [-1, 1]^2
        std = torch.exp(torch.clamp(self.log_std, -2.0, 1.0))
        return mu, std

    def sample(self, x, deterministic=False):
        mu, std = self.forward(x)
        if deterministic:
            return mu, torch.zeros_like(mu)
        dist = torch.distributions.Normal(mu, std)
        sample = dist.sample()
        log_prob = dist.log_prob(sample).sum(dim=-1)
        return torch.clamp(sample, -1.0, 1.0), log_prob

    def evaluate(self, x, action):
        mu, std = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

class CentralizedCritic(nn.Module):
    """Small MLP Centralized Value Function."""
    def __init__(self, state_dim=18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, state):
        return self.net(state).squeeze(-1)

class VecMAPPOTrainer:
    """
    High-throughput Batched Vectorised MAPPO Trainer:
    - Messages trained by recomputing sender messages and replaying in PPO minibatches.
    - Straight-through estimator (STE) for quantisation.
    - Observation, advantage, and value normalisation.
    """
    def __init__(self, mode='full_comm', fixed_bw=8, lr_actor=5e-4, lr_critic=1e-3, 
                 ppo_epochs=5, num_minibatches=4, clip_eps=0.2, entropy_coef=0.01,
                 gamma=0.99, gae_lambda=0.95, device='cpu'):
        self.mode = mode
        self.fixed_bw = fixed_bw
        self.device = torch.device(device)
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Modules
        self.quantiser = UniformQuantiserSTE(-1.0, 1.0)
        self.encoder = ScoutEncoder(obs_dim=16, msg_dim=16).to(self.device)
        
        in_dim = 10 if mode == 'full_obs' else 22
        self.rover1_policy = RoverDecoderPolicy(in_dim=in_dim, act_dim=2).to(self.device)
        self.rover2_policy = RoverDecoderPolicy(in_dim=in_dim, act_dim=2).to(self.device)
        self.critic = CentralizedCritic(state_dim=18).to(self.device)
        
        actor_params = list(self.rover1_policy.parameters()) + list(self.rover2_policy.parameters())
        if mode == 'full_comm':
            actor_params += list(self.encoder.parameters())
        self.actor_opt = optim.Adam(actor_params, lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def get_rover_inputs(self, scout_obs, rover1_obs, rover2_obs):
        B = scout_obs.shape[0]
        if self.mode == 'full_obs':
            return rover1_obs[:, 0:10], rover2_obs[:, 0:10]
        elif self.mode == 'no_comm':
            zero_msg = torch.zeros(B, 16, device=self.device)
            return torch.cat([rover1_obs[:, 0:6], zero_msg], dim=-1), torch.cat([rover2_obs[:, 0:6], zero_msg], dim=-1)
        else: # full_comm
            raw_msg = self.encoder(scout_obs)
            q_msg = self.quantiser(raw_msg, bit_width=self.fixed_bw)
            return torch.cat([rover1_obs[:, 0:6], q_msg], dim=-1), torch.cat([rover2_obs[:, 0:6], q_msg], dim=-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        r1_in, r2_in = self.get_rover_inputs(obs['agent_0'], obs['agent_1'], obs['agent_2'])
        p1_hat, lp1 = self.rover1_policy.sample(r1_in, deterministic=deterministic)
        p2_hat, lp2 = self.rover2_policy.sample(r2_in, deterministic=deterministic)
        
        target_estimates = torch.zeros(obs['agent_0'].shape[0], 3, 2, device=self.device)
        target_estimates[:, 1] = p1_hat
        target_estimates[:, 2] = p2_hat
        
        return target_estimates, (p1_hat, p2_hat), (lp1, lp2)

    def train_step(self, rollouts):
        T = rollouts['rewards'].shape[0]
        B = rollouts['rewards'].shape[1]
        
        with torch.no_grad():
            states_flat = rollouts['states'].reshape(T * B, 18)
            values_flat = self.critic(states_flat)
            values = values_flat.reshape(T, B)
            
            advantages = torch.zeros(T, B, device=self.device)
            last_gae = torch.zeros(B, device=self.device)
            next_value = torch.zeros(B, device=self.device)
            
            for t in reversed(range(T)):
                mask = 1.0 - rollouts['dones'][t]
                val_next = next_value if t == T - 1 else values[t + 1]
                delta = rollouts['rewards'][t] + self.gamma * val_next * mask - values[t]
                advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
                
            returns = advantages + values

        N = T * B
        b_scout_obs = rollouts['scout_obs'].reshape(N, 16)
        b_rover1_obs = rollouts['rover1_obs'].reshape(N, -1)
        b_rover2_obs = rollouts['rover2_obs'].reshape(N, -1)
        b_p1 = rollouts['p1_hat'].reshape(N, 2)
        b_p2 = rollouts['p2_hat'].reshape(N, 2)
        b_old_lp1 = rollouts['lp1'].reshape(N)
        b_old_lp2 = rollouts['lp2'].reshape(N)
        b_states = rollouts['states'].reshape(N, 18)
        b_returns = returns.reshape(N)
        b_adv = advantages.reshape(N)
        
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        minibatch_size = N // self.num_minibatches
        
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(N)
            for start in range(0, N, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                
                # Recompute rover inputs so gradients flow to encoder (via STE)
                mb_r1_in, mb_r2_in = self.get_rover_inputs(
                    b_scout_obs[mb_idx], b_rover1_obs[mb_idx], b_rover2_obs[mb_idx]
                )
                
                lp1, ent1 = self.rover1_policy.evaluate(mb_r1_in, b_p1[mb_idx])
                ratio1 = torch.exp(lp1 - b_old_lp1[mb_idx])
                surr1_1 = ratio1 * b_adv[mb_idx]
                surr2_1 = torch.clamp(ratio1, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv[mb_idx]
                actor_loss1 = -torch.min(surr1_1, surr2_1).mean() - self.entropy_coef * ent1.mean()
                
                lp2, ent2 = self.rover2_policy.evaluate(mb_r2_in, b_p2[mb_idx])
                ratio2 = torch.exp(lp2 - b_old_lp2[mb_idx])
                surr1_2 = ratio2 * b_adv[mb_idx]
                surr2_2 = torch.clamp(ratio2, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv[mb_idx]
                actor_loss2 = -torch.min(surr1_2, surr2_2).mean() - self.entropy_coef * ent2.mean()
                
                total_actor_loss = actor_loss1 + actor_loss2
                
                self.actor_opt.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.rover1_policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.rover2_policy.parameters(), 0.5)
                if self.mode == 'full_comm':
                    nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
                self.actor_opt.step()
                
                v_pred = self.critic(b_states[mb_idx])
                critic_loss = 0.5 * ((v_pred - b_returns[mb_idx]) ** 2).mean()
                
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()
