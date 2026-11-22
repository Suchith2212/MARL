"""
Primal-Dual MAPPO with Zero-Order Hold (ZOH) Memory and PID Lagrangian Controller.

Key features:
1. Zero-Order Hold (ZOH): Rovers receive last received message (d=16) and normalized
   time since reception tau in [0, 1]. Total rover input = 6 + 16 + 1 = 23 dims.
2. PID Lagrangian Controller:
     e_k = bar_C_k - B
     I_k = max(0, I_{k-1} + alpha_I * e_k)
     lambda_k = max(0, K_P * e_k + I_k)
3. Lambda = 0 warm-up phase (first warmup_updates).
4. Gate entropy regularization to prevent gate collapse.
5. Support for:
   - learned gating (Primal-Dual)
   - periodic gating baseline (fixed period P = round(16*b / B))
   - random gating baseline (Bernoulli with p = min(1.0, B / (16*b)))
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


class ZOHMAPPOTrainer:
    def __init__(self, mode='primal_dual', fixed_bw=8, budget=16, d=16, T=25,
                 lr_actor=5e-4, lr_critic=1e-3, K_p=0.005, alpha_I=0.01,
                 warmup_updates=25, ppo_epochs=5, num_minibatches=4, clip_eps=0.2,
                 entropy_coef=0.01, gate_entropy_coef=0.05, gamma=0.99, gae_lambda=0.95,
                 device='cpu'):
        self.mode = mode # 'primal_dual', 'periodic', 'random'
        self.fixed_bw = fixed_bw
        self.budget = float(budget)
        self.d = d
        self.T = T
        self.device = torch.device(device)
        self.K_p = K_p
        self.alpha_I = alpha_I
        self.warmup_updates = warmup_updates
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.gate_entropy_coef = gate_entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # PID Lagrangian State
        self.integral_I = 0.0
        self.dual_lambda = 0.0
        self.update_count = 0
        self.lambda_history = []
        self.cost_history = []
        
        # Baseline scheduling parameters
        pkt_cost = float(self.d * self.fixed_bw)
        self.freq_star = min(1.0, self.budget / pkt_cost)
        self.period = max(1, int(round(1.0 / max(1e-4, self.freq_star))))

        # Modules
        self.quantiser = UniformQuantiserSTE(-1.0, 1.0)
        self.encoder = ScoutEncoder(obs_dim=16, msg_dim=d).to(self.device)
        self.gate_head = ScoutGatingHead(obs_dim=16).to(self.device)
        
        # Rover policy takes rover_obs (6) + last_msg (16) + tau (1) = 23 dims
        self.rover1_policy = RoverDecoderPolicy(in_dim=6 + d + 1, act_dim=2).to(self.device)
        self.rover2_policy = RoverDecoderPolicy(in_dim=6 + d + 1, act_dim=2).to(self.device)
        self.critic = CentralizedCritic(state_dim=18).to(self.device)
        
        actor_params = (list(self.rover1_policy.parameters()) +
                        list(self.rover2_policy.parameters()) +
                        list(self.encoder.parameters()))
        if self.mode == 'primal_dual':
            actor_params += list(self.gate_head.parameters())
            
        self.actor_opt = optim.Adam(actor_params, lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def get_gate_dist(self, scout_obs):
        logits = self.gate_head(scout_obs)
        return torch.distributions.Bernoulli(logits=logits), logits

    def compute_step_gate(self, scout_obs, t_step, acc=None, deterministic=False):
        B = scout_obs.shape[0]
        if acc is None:
            acc = torch.zeros(B, device=self.device)
            
        if self.mode == 'periodic':
            # Deterministic transmission every P steps
            open_gate = (t_step % self.period == 0)
            gates = torch.full((B,), float(open_gate), device=self.device)
            gate_lp = torch.zeros(B, device=self.device)
            return gates, gate_lp, acc
        elif self.mode == 'random':
            # Bernoulli sampling with rate freq_star so test cost matches budget
            gates = torch.bernoulli(torch.full((B,), self.freq_star, device=self.device))
            gate_lp = torch.zeros(B, device=self.device)
            return gates, gate_lp, acc
        else: # primal_dual learned gating
            gate_dist, logits = self.get_gate_dist(scout_obs)
            if deterministic:
                # Integrate-and-fire deterministic gating
                p = torch.sigmoid(logits)
                acc = acc + p
                gates = (acc >= 1.0).float()
                acc = acc - gates
                gate_lp = torch.zeros(B, device=self.device)
            else:
                gates = gate_dist.sample()
                gate_lp = gate_dist.log_prob(gates)
            return gates, gate_lp, acc

    def update_zoh_buffer(self, last_msg, tau, q_msg, gates):
        """
        ZOH Memory Buffer Update:
        If gate == 1: last_msg <- q_msg, tau <- 0
        If gate == 0: last_msg <- last_msg, tau <- tau + 1/T
        """
        gate_m = gates.unsqueeze(-1) # [B, 1]
        new_last_msg = gate_m * q_msg + (1.0 - gate_m) * last_msg
        new_tau = (1.0 - gate_m) * (tau + 1.0 / float(self.T))
        return new_last_msg, new_tau

    @torch.no_grad()
    def act(self, obs, last_msg, tau, t_step, acc=None, deterministic=False):
        scout_obs = obs['agent_0']
        B = scout_obs.shape[0]
        
        # 1. Compute gate
        gates, gate_lp, new_acc = self.compute_step_gate(scout_obs, t_step, acc=acc, deterministic=deterministic)
        
        # 2. Encode and quantize message
        raw_msg = self.encoder(scout_obs)
        q_msg = self.quantiser(raw_msg, bit_width=self.fixed_bw)
        
        # 3. Update ZOH buffer
        new_last_msg, new_tau = self.update_zoh_buffer(last_msg, tau, q_msg, gates)
        
        # 4. Form rover inputs: rover_obs (6) + new_last_msg (16) + new_tau (1) = 23
        r1_in = torch.cat([obs['agent_1'][:, 0:6], new_last_msg, new_tau], dim=-1)
        r2_in = torch.cat([obs['agent_2'][:, 0:6], new_last_msg, new_tau], dim=-1)
        
        p1_hat, lp1 = self.rover1_policy.sample(r1_in, deterministic=deterministic)
        p2_hat, lp2 = self.rover2_policy.sample(r2_in, deterministic=deterministic)
        
        target_estimates = torch.zeros(B, 3, 2, device=self.device)
        target_estimates[:, 1] = p1_hat
        target_estimates[:, 2] = p2_hat
        
        step_costs = gates * (self.d * self.fixed_bw) # bits
        return target_estimates, gates, gate_lp, (p1_hat, p2_hat), (lp1, lp2), step_costs, new_last_msg, new_tau, new_acc

    def train_step(self, rollouts):
        self.update_count += 1
        T = rollouts['raw_rewards'].shape[0]
        B = rollouts['raw_rewards'].shape[1]
        
        # Episode-averaged cost per environment
        ep_costs = rollouts['step_costs'].sum(dim=0) / T # [B]
        batch_bar_C = float(ep_costs.mean().item())
        self.cost_history.append(batch_bar_C)
        
        # PID Lagrangian Update
        err = batch_bar_C - self.budget
        if self.update_count <= self.warmup_updates or self.mode != 'primal_dual':
            self.dual_lambda = 0.0
            self.integral_I = 0.0
        else:
            self.integral_I = max(0.0, self.integral_I + self.alpha_I * err)
            self.dual_lambda = max(0.0, self.K_p * err + self.integral_I)
            
        self.lambda_history.append(self.dual_lambda)
        
        # Augmented Lagrangian rewards
        penalties = self.dual_lambda * (rollouts['step_costs'] - self.budget) / float(T)
        aug_rewards = rollouts['raw_rewards'] - penalties
        
        # GAE
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
        b_held_msg = rollouts['held_msg'].reshape(N, 16)
        b_tau = rollouts['tau'].reshape(N, 1)
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
                
                total_loss = 0.0
                
                # 1. Gate loss (if primal_dual)
                if self.mode == 'primal_dual':
                    gate_dist, _ = self.get_gate_dist(b_scout_obs[mb_idx])
                    new_gate_lp = gate_dist.log_prob(b_gates[mb_idx])
                    gate_ratio = torch.exp(new_gate_lp - b_old_gate_lp[mb_idx])
                    gate_surr1 = gate_ratio * b_adv[mb_idx]
                    gate_surr2 = torch.clamp(gate_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv[mb_idx]
                    gate_loss = -torch.min(gate_surr1, gate_surr2).mean() - self.gate_entropy_coef * gate_dist.entropy().mean()
                    total_loss = total_loss + gate_loss
                    
                # 2. Recompute rover inputs with gradients flowing through encoder & STE when gate == 1
                mb_raw_msg = self.encoder(b_scout_obs[mb_idx])
                mb_q_msg = self.quantiser(mb_raw_msg, bit_width=self.fixed_bw)
                gate_m = b_gates[mb_idx].unsqueeze(-1)
                mb_msg = gate_m * mb_q_msg + (1.0 - gate_m) * b_held_msg[mb_idx]
                
                mb_r1_in = torch.cat([b_rover1_obs[mb_idx, 0:6], mb_msg, b_tau[mb_idx]], dim=-1)
                mb_r2_in = torch.cat([b_rover2_obs[mb_idx, 0:6], mb_msg, b_tau[mb_idx]], dim=-1)
                
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
                
                total_loss = total_loss + actor_loss1 + actor_loss2
                
                self.actor_opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.rover1_policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.rover2_policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
                if self.mode == 'primal_dual':
                    nn.utils.clip_grad_norm_(self.gate_head.parameters(), 0.5)
                self.actor_opt.step()
                
                # Critic loss
                v_pred = self.critic(b_states[mb_idx])
                critic_loss = 0.5 * ((v_pred - b_returns[mb_idx]) ** 2).mean()
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()
                
        return {'batch_cost': batch_bar_C, 'dual_lambda': self.dual_lambda}
