import numpy as np
import torch
from collections import deque

class BroadcastChannel:
    """
    Simulated imperfect broadcast communication channel and exact payload-bit accountant.
    
    Specs:
    - Message dimension d = 16.
    - Bit widths W = {1, 2, 4, 8}.
    - Cost is counted ONCE per broadcast at sender, regardless of number of receivers.
    - No headers, retransmissions, or channel coding overhead.
    - Drops with probability p independently per receiver.
    - Delay tau steps.
    """
    def __init__(self, n_agents=3, message_dim=16, p_drop=0.0, delay_tau=0, gamma=0.99, horizon=25):
        self.n_agents = n_agents
        self.message_dim = message_dim
        self.p_drop = p_drop
        self.delay_tau = delay_tau
        self.gamma = gamma
        self.horizon = horizon
        
        # Precompute Z_T = sum_{t=1}^T gamma^{t-1}
        self.discount_weights = np.array([gamma ** t for t in range(horizon)], dtype=np.float32)
        self.Z_T = float(np.sum(self.discount_weights))
        
        self.reset()

    def reset(self):
        self.step_idx = 0
        # Buffer for delayed messages: deque of tuples (source_agent, dest_agent, delivery_step, message_tensor)
        self.delay_queue = []
        # Per-step payload tracking for episode accounting
        self.episode_step_costs = []
        self.episode_step_bits = []

    def transmit_and_deliver(self, messages, gates, bit_widths):
        """
        messages: Tensor or list of messages [n_agents, d]
        gates: list or Tensor of binary gates [n_agents], 1 if sent, 0 otherwise
        bit_widths: list or Tensor of integer bit-widths [n_agents]
        
        Returns:
            received_messages: dict mapping receiver_idx -> list of (sender_idx, message_tensor, mask_delivered)
            step_cost: total team payload bits transmitted at this step
        """
        step_cost = 0
        for i in range(self.n_agents):
            g_i = int(gates[i])
            b_i = int(bit_widths[i]) if g_i == 1 else 0
            # Payload cost counted ONCE per broadcast at sender
            if g_i == 1:
                cost_i = self.message_dim * b_i
                step_cost += cost_i
                
                # Broadcast to every peer j != i
                for j in range(self.n_agents):
                    if i == j:
                        continue
                    # Check drop
                    is_dropped = np.random.rand() < self.p_drop
                    delivery_time = self.step_idx + self.delay_tau
                    if not is_dropped:
                        self.delay_queue.append({
                            'sender': i,
                            'receiver': j,
                            'delivery_step': delivery_time,
                            'message': messages[i]
                        })

        self.episode_step_bits.append(step_cost)
        
        # Collect currently deliverable messages for each receiver j
        delivered = {j: [] for j in range(self.n_agents)}
        remaining_queue = []
        for item in self.delay_queue:
            if item['delivery_step'] <= self.step_idx:
                delivered[item['receiver']].append(item)
            else:
                remaining_queue.append(item)
        self.delay_queue = remaining_queue
        
        self.step_idx += 1
        return delivered, step_cost

    def get_episode_metrics(self):
        """
        Computes C (discounted-normalised) and C_bar (plain average payload bits per step).
        """
        costs = np.array(self.episode_step_bits, dtype=np.float32)
        T_actual = len(costs)
        if T_actual == 0:
            return 0.0, 0.0
        
        # C_bar: plain undiscounted average payload bits per step
        C_bar = float(np.mean(costs))
        
        # C: discounted-normalised payload cost using Z_T
        weights = self.discount_weights[:T_actual]
        Z = np.sum(weights) if T_actual != self.horizon else self.Z_T
        C_disc = float(np.sum(weights * costs) / Z)
        
        return C_disc, C_bar
