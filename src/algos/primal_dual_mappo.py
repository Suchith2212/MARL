"""
Primal-Dual Gated MAPPO Trainer for Constrained Communication in MARL.

- Fixed-width precision b in {1, 2, 4, 8} (or dynamic b in Phase 3)
- Gating policy pi_gate(g | u) outputs Bernoulli logit for gate g in {0, 1}
- When g = 0: masked message (rovers receive zeros)
- When g = 1: quantized message q(m, b) sent
- Cost per step: c_t = d * b * g_t (d = 16, no headers)
- Constraint: E[bar_C] <= B bits/step (B in {8, 16, 32, 64})
- Dual multiplier lambda updated via subgradient ascent:
    lambda <- max(0, lambda + alpha_lambda * (batch_bar_C - B))
- Step reward augmented with Lagrangian penalty:
    r'_t = r_t - lambda * (c_t - B) / T
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.algos.vec_mappo import UniformQuantiserSTE, ScoutEncoder, RoverDecoderPolicy, CentralizedCritic

class ScoutGatingHead(nn.Module):
    """Outputs gating probability p_t = sigmoid(logit_t)."""
    def __init__(self, obs_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, obs):
        return self.net(obs).squeeze(-1) # logits [B]

class PrimalDualGatedMAPPOTrainer:
    def __init__(self, fixed_bw=8, budget=16, d=16, T=25,
                 lr_actor=5e-4, lr_critic=1e-3, lr_gate=5e-4, alpha_lambda=0.01,
                 ppo_epochs=5, num_minibatches=4, clip_eps=0.2, entropy_coef=0.01,
                 gate_entropy_coef=0.02, gamma=0.99, gae_lambda=0.95, device='cpu'):
        self.fixed_bw = fixed_bw
        self.budget = float(budget)
        self.d = d
        self.T = T
        self.device = torch.device(device)
        self.alpha_lambda = alpha_lambda
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.gate_entropy_coef = gate_entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Dual multiplier lambda >= 0
        self.dual_lambda = 0.0
        self.lambda_history = []
        self.cost_history = []

        # Modules
        self.quantiser = UniformQuantiserSTE(-1.0, 1.0)
        self.encoder = ScoutEncoder(obs_dim=16, msg_dim=d).to(self.device)
        self.gate_head = ScoutGatingHead(obs_dim=16).to(self.device)
        
        # Rover policy takes rover_obs (6) + received_msg (d=16) = 22
        self.rover1_policy = RoverDecoderPolicy(in_dim=6 + d, act_dim=2).to(self.device)
        self.rover2_policy = RoverDecoderPolicy(in_dim=6 + d, act_dim=2).to(self.device)
        self.critic = CentralizedCritic(state_dim=18).to(self.device)
        
        actor_params = (list(self.rover1_policy.parameters()) +
                        list(self.rover2_policy.parameters()) +
                        list(self.encoder.parameters()) +
                        list(self.gate_head.parameters()))
        self.actor_opt = optim.Adam(actor_params, lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def get_gate_dist(self, scout_obs):
        logits = self.gate_head(scout_obs)
        return torch.distributions.Bernoulli(logits=logits), logits

    def get_rover_inputs(self, scout_obs, rover1_obs, rover2_obs, gates):
        """
        gates: [B] in {0, 1}
        When gate == 0, message is masked to zeros.
        """
        B = scout_obs.shape[0]
        raw_msg = self.encoder(scout_obs) # [B, d]
        q_msg = self.quantiser(raw_msg, bit_width=self.fixed_bw) # [B, d]
        
        # Mask message with gate
        gate_mask = gates.unsqueeze(-1).float() # [B, 1]
        masked_msg = q_msg * gate_mask # [B, d]
        
        r1_in = torch.cat([rover1_obs[:, 0:6], masked_msg], dim=-1)
        r2_in = torch.cat([rover2_obs[:, 0:6], masked_msg], dim=-1)
        return r1_in, r2_in

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        scout_obs = obs['agent_0']
        B = scout_obs.shape[0]
        gate_dist, logits = self.get_gate_dist(scout_obs)
        
        if deterministic:
            gates = (torch.sigmoid(logits) >= 0.5).float()
            gate_lp = torch.zeros(B, device=self.device)
        else:
            gates = gate_dist.sample()
            gate_lp = gate_dist.log_prob(gates)
            
        r1_in, r2_in = self.get_rover_inputs(scout_obs, obs['agent_1'], obs['agent_2'], gates)
        p1_hat, lp1 = self.rover1_policy.sample(r1_in, deterministic=deterministic)
        p2_hat, lp2 = self.rover2_policy.sample(r2_in, deterministic=deterministic)
        
        target_estimates = torch.zeros(B, 3, 2, device=self.device)
        target_estimates[:, 1] = p1_hat
        target_estimates[:, 2] = p2_hat
        
        step_costs = gates * (self.d * self.fixed_bw) # bits
        return target_estimates, gates, gate_lp, (p1_hat, p2_hat), (lp1, lp2), step_costs

    def train_step(self, rollouts):
        """
        rollouts contains:
          states: [T, B, 18]
          scout_obs: [T, B, 16]
          rover1_obs: [T, B, 10]
          rover2_obs: [T, B, 10]
          gates: [T, B]
          gate_lps: [T, B]
          p1_hat, p2_hat: [T, B, 2]
          lp1, lp2: [T, B]
          raw_rewards: [T, B]
          step_costs: [T, B]
          dones: [T, B]
        """
        T = rollouts['raw_rewards'].shape[0]
        B = rollouts['raw_rewards'].shape[1]
        
        # Compute episode-averaged cost per environment
        # bar_C = sum_{t=1}^T c_t / T
        ep_costs = rollouts['step_costs'].sum(dim=0) / T # [B]
        batch_bar_C = float(ep_costs.mean().item())
        self.cost_history.append(batch_bar_C)
        
        # Dual update: lambda <- max(0, lambda + alpha_lambda * (bar_C - B))
        self.dual_lambda = max(0.0, self.dual_lambda + self.alpha_lambda * (batch_bar_C - self.budget))
        self.lambda_history.append(self.dual_lambda)
        
        # Augmented Lagrangian step rewards:
        # r'_t = r_t - lambda * (c_t - B) / T
        penalties = self.dual_lambda * (rollouts['step_costs'] - self.budget) / float(T)
        aug_rewards = rollouts['raw_rewards'] - penalties
        
        # GAE with augmented rewards
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
                delta = aug_rewards[t] + self.gamma * val_next * mask - values[t]
                advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
                
            returns = advantages + values

        N = T * B
        b_scout_obs = rollouts['scout_obs'].reshape(N, 16)
        b_rover1_obs = rollouts['rover1_obs'].reshape(N, -1)
        b_rover2_obs = rollouts['rover2_obs'].reshape(N, -1)
        b_gates = rollouts['gates'].reshape(N)
        b_old_gate_lp = rollouts['gate_lps'].reshape(N)
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
                
                # 1. Gate policy loss
                gate_dist, _ = self.get_gate_dist(b_scout_obs[mb_idx])
                new_gate_lp = gate_dist.log_prob(b_gates[mb_idx])
                gate_ratio = torch.exp(new_gate_lp - b_old_gate_lp[mb_idx])
                gate_surr1 = gate_ratio * b_adv[mb_idx]
                gate_surr2 = torch.clamp(gate_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv[mb_idx]
                gate_loss = -torch.min(gate_surr1, gate_surr2).mean() - self.gate_entropy_coef * gate_dist.entropy().mean()
                
                # 2. Rover policies loss with recomputed messages (STE gradient flows through encoder)
                mb_r1_in, mb_r2_in = self.get_rover_inputs(
                    b_scout_obs[mb_idx], b_rover1_obs[mb_idx], b_rover2_obs[mb_idx], b_gates[mb_idx]
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
                
                total_actor_loss = actor_loss1 + actor_loss2 + gate_loss
                
                self.actor_opt.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.rover1_policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.rover2_policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.gate_head.parameters(), 0.5)
                self.actor_opt.step()
                
                # Critic update
                v_pred = self.critic(b_states[mb_idx])
                critic_loss = 0.5 * ((v_pred - b_returns[mb_idx]) ** 2).mean()
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()
                
        return {'batch_cost': batch_bar_C, 'dual_lambda': self.dual_lambda}
