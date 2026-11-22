import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.comm_mappo_net import CommActor, CentralizedCritic
from src.quantiser.uniform_quantiser import UniformQuantiser
from src.channel.broadcast_channel import BroadcastChannel

class MAPPOTrainer:
    """
    Vectorized MAPPO Trainer with learned gating, learned precision,
    and primal-dual budget control.
    
    Modes:
      - 'no_comm': No communication
      - 'full_comm_fixed': Fixed transmission (g=1) with fixed bit-width b in {1, 2, 4, 8}
      - 'learned_gate_fixed': Learned gate g in {0, 1} with fixed bit-width b
      - 'learned_precision': Learned gate g and learned bit-width b in {1, 2, 4, 8}
      - 'soft_penalty': Fixed Lagrangian soft penalty
    """
    def __init__(
        self,
        obs_dim,
        state_dim,
        n_agents=3,
        action_dim=5,
        hidden_dim=64,
        message_dim=16,
        context_dim=32,
        mode='full_comm_fixed',
        fixed_bit_width=8,
        budget_B=48.0,
        dual_lr=0.002,
        actor_lr=1e-3,
        critic_lr=2e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        ppo_epochs=5,
        entropy_coef=0.01,
        value_coef=0.5,
        device='cpu'
    ):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.mode = mode
        self.fixed_bit_width = fixed_bit_width
        self.budget_B = float(budget_B)
        self.dual_lr = dual_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)
        
        self.allowed_widths = (1, 2, 4, 8)
        self.width_to_idx = {w: i for i, w in enumerate(self.allowed_widths)}
        self.idx_to_width = {i: w for i, w in enumerate(self.allowed_widths)}
        
        # Networks
        self.actors = nn.ModuleList([
            CommActor(obs_dim, action_dim, hidden_dim, message_dim, context_dim, self.allowed_widths)
            for _ in range(n_agents)
        ]).to(self.device)
        
        self.critic = CentralizedCritic(state_dim, hidden_dim).to(self.device)
        self.quantiser = UniformQuantiser(message_dim=message_dim, allowed_widths=self.allowed_widths)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actors.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Primal-dual parameters
        self.lambda_cost = 0.0
        self.moving_cost_C = 0.0
        self.cost_ema_beta = 0.1

    def act(self, obs_dict, deterministic=False):
        """Rollout communication & feature selection."""
        obs_tensor = [torch.tensor(obs_dict[f"agent_{i}"], dtype=torch.float32, device=self.device) for i in range(self.n_agents)]
        
        gates = []
        gate_lps = []
        bit_widths = []
        width_lps = []
        raw_msgs = []
        h_list = []
        
        for i in range(self.n_agents):
            h_i, m_i = self.actors[i].get_features_and_message(obs_tensor[i].unsqueeze(0))
            h_list.append(h_i)
            raw_msgs.append(m_i.squeeze(0))
            
            gate_dist, width_dist = self.actors[i].forward_comm_heads(h_i)
            
            # Mode gating
            if self.mode == 'no_comm':
                g_val = 0
                g_lp = 0.0
            elif self.mode == 'full_comm_fixed':
                g_val = 1
                g_lp = 0.0
            else:
                if deterministic:
                    g_val = int(torch.argmax(gate_dist.probs, dim=-1).item())
                else:
                    g_val = int(gate_dist.sample().item())
                g_lp = float(gate_dist.log_prob(torch.tensor([g_val], device=self.device)).item())
                
            # Mode bit-width
            if self.mode in ['no_comm', 'full_comm_fixed', 'learned_gate_fixed']:
                b_val = self.fixed_bit_width if g_val == 1 else 0
                w_lp = 0.0
            else:  # learned_precision
                if g_val == 0:
                    b_val = 0
                    w_lp = 0.0
                else:
                    if deterministic:
                        w_idx = int(torch.argmax(width_dist.probs, dim=-1).item())
                    else:
                        w_idx = int(width_dist.sample().item())
                    b_val = self.idx_to_width[w_idx]
                    w_lp = float(width_dist.log_prob(torch.tensor([w_idx], device=self.device)).item())
                    
            gates.append(g_val)
            gate_lps.append(g_lp)
            bit_widths.append(b_val)
            width_lps.append(w_lp)
            
        raw_msgs_tensor = torch.stack(raw_msgs)
        return h_list, raw_msgs_tensor, gates, gate_lps, bit_widths, width_lps

    def get_actions(self, h_list, received_dict, deterministic=False):
        """Action selection after receiving channel-delivered messages."""
        actions = {}
        action_lps = {}
        
        for i in range(self.n_agents):
            rec_items = received_dict[i]
            if len(rec_items) > 0:
                peer_msgs = torch.stack([item['message'] for item in rec_items]).unsqueeze(0)
                peer_masks = torch.ones((1, len(rec_items)), device=self.device)
            else:
                peer_msgs = torch.zeros((1, 1, self.actors[i].message_dim), device=self.device)
                peer_masks = torch.zeros((1, 1), device=self.device)
                
            u_i = self.actors[i].aggregator(h_list[i], peer_msgs, peer_masks)
            action_dist = self.actors[i].forward_action_head(h_list[i], u_i)
            
            if deterministic:
                a_val = int(torch.argmax(action_dist.probs, dim=-1).item())
            else:
                a_val = int(action_dist.sample().item())
            a_lp = float(action_dist.log_prob(torch.tensor([a_val], device=self.device)).item())
            
            actions[f"agent_{i}"] = a_val
            action_lps[f"agent_{i}"] = a_lp
            
        return actions, action_lps

    def train_step(self, rollout_buffer):
        """
        Vectorized PPO update over all T steps.
        Recomputes sender messages from history, replays channel masks,
        applies STE quantisation, masks unused comm choices, and updates lambda.
        """
        obs_traj = torch.tensor(np.array(rollout_buffer['obs']), dtype=torch.float32, device=self.device) # [T, n_agents, obs_dim]
        states = torch.tensor(np.array(rollout_buffer['state']), dtype=torch.float32, device=self.device)     # [T, state_dim]
        actions = torch.tensor(np.array(rollout_buffer['actions']), dtype=torch.long, device=self.device)    # [T, n_agents]
        old_action_lps = torch.tensor(np.array(rollout_buffer['action_lps']), dtype=torch.float32, device=self.device)
        gates = torch.tensor(np.array(rollout_buffer['gates']), dtype=torch.long, device=self.device)        # [T, n_agents]
        old_gate_lps = torch.tensor(np.array(rollout_buffer['gate_lps']), dtype=torch.float32, device=self.device)
        bit_widths = torch.tensor(np.array(rollout_buffer['bit_widths']), dtype=torch.long, device=self.device)
        old_width_lps = torch.tensor(np.array(rollout_buffer['width_lps']), dtype=torch.float32, device=self.device)
        deliv_masks = torch.tensor(np.array(rollout_buffer['deliv_masks']), dtype=torch.float32, device=self.device) # [T, n_agents, n_agents]
        raw_rewards = np.array(rollout_buffer['rewards'], dtype=np.float32)
        step_costs = np.array(rollout_buffer['step_costs'], dtype=np.float32)
        dones = np.array(rollout_buffer.get('dones', [1 if (i + 1) % 25 == 0 else 0 for i in range(len(raw_rewards))]), dtype=np.float32)
        T = len(raw_rewards)
        
        # Penalised rewards: r_t^lambda = r_t - lambda * (1 / Z_T) * cost_t
        weights = np.array([self.gamma ** (t % 25) for t in range(T)], dtype=np.float32)
        Z_T = float(np.sum([self.gamma ** t for t in range(25)]))
        penalized_rewards = raw_rewards - self.lambda_cost * (step_costs / Z_T)
        
        with torch.no_grad():
            values = self.critic(states).squeeze(-1).cpu().numpy()
            
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < T else 0.0
            non_terminal = 1.0 - dones[t]
            delta = penalized_rewards[t] + self.gamma * next_val * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
            
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        adv_norm = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Width index mapping
        width_idx_map = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2, 3], dtype=torch.long, device=self.device)
        
        for _ in range(self.ppo_epochs):
            # 1. Critic Update
            curr_values = self.critic(states).squeeze(-1)
            critic_loss = self.value_coef * F.mse_loss(curr_values, ret_tensor)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            # 2. Vectorized Sender Pass & Quantisation (STE)
            h_all = []
            qm_all = []
            gate_dists = []
            width_dists = []
            
            for i in range(self.n_agents):
                h_i, m_i = self.actors[i].get_features_and_message(obs_traj[:, i, :])  # [T, hidden], [T, d]
                h_all.append(h_i)
                g_dist, w_dist = self.actors[i].forward_comm_heads(h_i)
                gate_dists.append(g_dist)
                width_dists.append(w_dist)
                
                # Quantise with STE
                bw_i = torch.clamp(bit_widths[:, i], min=1)
                qm_i = self.quantiser(m_i, bw_i)
                # Gate mask
                g_float = gates[:, i:i+1].float()
                qm_i = qm_i * g_float
                qm_all.append(qm_i)
                
            # 3. Vectorized Receiver Pass, Attention & Loss
            total_actor_loss = 0.0
            for i in range(self.n_agents):
                peer_msgs_list = []
                peer_masks_list = []
                for j in range(self.n_agents):
                    if i == j:
                        continue
                    # Delivery mask from j to i
                    mask_ji = deliv_masks[:, j, i] * gates[:, j].float()
                    peer_msgs_list.append(qm_all[j])
                    peer_masks_list.append(mask_ji)
                    
                peer_msgs_t = torch.stack(peer_msgs_list, dim=1)   # [T, num_peers, d]
                peer_masks_t = torch.stack(peer_masks_list, dim=1)  # [T, num_peers]
                
                u_i = self.actors[i].aggregator(h_all[i], peer_msgs_t, peer_masks_t) # [T, context_dim]
                a_dist = self.actors[i].forward_action_head(h_all[i], u_i)
                
                a_lp = a_dist.log_prob(actions[:, i])
                old_a_lp = old_action_lps[:, i]
                
                # Masked communication log-prob
                g_lp = gate_dists[i].log_prob(gates[:, i])
                old_g_lp = old_gate_lps[:, i]
                
                w_indices = width_idx_map[bit_widths[:, i]]
                w_lp = width_dists[i].log_prob(w_indices)
                old_w_lp = old_width_lps[:, i]
                
                g_mask = gates[:, i].float()
                comm_lp = g_lp + g_mask * w_lp
                comm_old_lp = old_g_lp + g_mask * old_w_lp
                
                if self.mode in ['learned_gate_fixed']:
                    total_lp = a_lp + g_lp
                    total_old_lp = old_a_lp + old_g_lp
                elif self.mode == 'learned_precision':
                    total_lp = a_lp + comm_lp
                    total_old_lp = old_a_lp + comm_old_lp
                else:  # no_comm or full_comm_fixed
                    total_lp = a_lp
                    total_old_lp = old_a_lp
                    
                ratio = torch.exp(total_lp - total_old_lp)
                surr1 = ratio * adv_norm
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_norm
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy
                ent = a_dist.entropy().mean()
                if self.mode in ['learned_gate_fixed', 'learned_precision']:
                    ent = ent + gate_dists[i].entropy().mean()
                if self.mode == 'learned_precision':
                    ent = ent + (g_mask * width_dists[i].entropy()).mean()
                actor_loss = actor_loss - self.entropy_coef * ent
                
                total_actor_loss = total_actor_loss + actor_loss
                
            total_actor_loss = total_actor_loss / self.n_agents
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors.parameters(), 0.5)
            self.actor_optimizer.step()

        # 4. Primal-Dual Lambda update
        num_episodes = max(1, int(np.sum(dones)))
        c_rollout = float(np.sum(weights * step_costs) / (num_episodes * Z_T))
        self.moving_cost_C = (1.0 - self.cost_ema_beta) * self.moving_cost_C + self.cost_ema_beta * c_rollout
        if self.mode in ['learned_gate_fixed', 'learned_precision']:
            self.lambda_cost = max(0.0, self.lambda_cost + self.dual_lr * (self.moving_cost_C - self.budget_B))
            
        return float(c_rollout), float(np.mean(step_costs)), self.lambda_cost
