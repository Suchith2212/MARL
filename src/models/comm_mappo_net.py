import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class AttentionAggregator(nn.Module):
    """
    Computes message context u_i^t = Attn(h_i^t, {m_{j -> i}}_{j != i}).
    Query comes from agent i's local hidden state h_i^t.
    Keys and Values come from received messages.
    """
    def __init__(self, hidden_dim=64, message_dim=16, context_dim=32):
        super().__init__()
        self.context_dim = context_dim
        self.q_proj = nn.Linear(hidden_dim, context_dim)
        self.k_proj = nn.Linear(message_dim, context_dim)
        self.v_proj = nn.Linear(message_dim, context_dim)
        self.out_proj = nn.Linear(context_dim, context_dim)

    def forward(self, h_i, peer_messages, peer_masks):
        """
        h_i: (batch, hidden_dim)
        peer_messages: (batch, num_peers, message_dim)
        peer_masks: (batch, num_peers) - binary float, 1 if valid received message, 0 if empty/dropped
        """
        batch_size, num_peers, _ = peer_messages.shape
        
        # Query: (batch, 1, context_dim)
        Q = self.q_proj(h_i).unsqueeze(1)
        # Keys & Values: (batch, num_peers, context_dim)
        K = self.k_proj(peer_messages)
        V = self.v_proj(peer_messages)
        
        # Scaled dot-product attention
        scale = 1.0 / (self.context_dim ** 0.5)
        scores = torch.bmm(Q, K.transpose(1, 2)) * scale  # (batch, 1, num_peers)
        
        # Apply mask: if peer_mask == 0, score = -1e9
        mask_expanded = peer_masks.unsqueeze(1)  # (batch, 1, num_peers)
        scores = scores.masked_fill(mask_expanded == 0, -1e9)
        
        # If all peers are masked out for an agent, fallback safely to zeros
        all_masked = (peer_masks.sum(dim=-1, keepdim=True) == 0)  # (batch, 1)
        
        weights = F.softmax(scores, dim=-1)  # (batch, 1, num_peers)
        attn_out = torch.bmm(weights, V).squeeze(1)  # (batch, context_dim)
        
        # Mask out agents with no received messages at all
        attn_out = torch.where(all_masked, torch.zeros_like(attn_out), attn_out)
        return self.out_proj(attn_out)


class CommActor(nn.Module):
    """
    Decentralised Actor for Agent i:
    - Feature encoder for local observation/history h_i^t
    - Message generator m_i^t = f_phi(h_i^t) in [-1, 1]^d
    - Categorical gate head pi_g(g_i^t | h_i^t)
    - Categorical bit-width head pi_b(b_i^t | h_i^t) over W = {1, 2, 4, 8}
    - Action head pi_theta(a_i^t | h_i^t, u_i^t)
    """
    def __init__(self, obs_dim, action_dim=5, hidden_dim=64, message_dim=16, context_dim=32, allowed_widths=(1, 2, 4, 8)):
        super().__init__()
        self.message_dim = message_dim
        self.allowed_widths = allowed_widths
        self.num_widths = len(allowed_widths)
        
        # Local feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Message head: f_phi(h_i) bounded in [-1, 1]^d
        self.msg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh()
        )
        
        # Communication discrete decision heads (Categorical PPO, NO Gumbel-Softmax)
        self.gate_logits = nn.Linear(hidden_dim, 2)  # {0: silent, 1: transmit}
        self.width_logits = nn.Linear(hidden_dim, self.num_widths)  # indices into allowed_widths
        
        # Attention aggregation module
        self.aggregator = AttentionAggregator(hidden_dim, message_dim, context_dim)
        
        # Action head conditioned on both local history and aggregated message context u_i^t
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def get_features_and_message(self, obs):
        h = self.encoder(obs)
        m = self.msg_head(h)
        return h, m

    def forward_comm_heads(self, h):
        gate_dist = Categorical(logits=self.gate_logits(h))
        width_dist = Categorical(logits=self.width_logits(h))
        return gate_dist, width_dist

    def forward_action_head(self, h, u):
        combined = torch.cat([h, u], dim=-1)
        action_dist = Categorical(logits=self.action_head(combined))
        return action_dist


class CentralizedCritic(nn.Module):
    """
    Centralised Critic V_omega(s) for MAPPO.
    Takes full environment state s (or joint observations).
    """
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)
