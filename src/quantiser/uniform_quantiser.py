import torch
import torch.nn as nn

class UniformQuantiser(nn.Module):
    """
    Uniform Quantiser Q_b for message vectors m in R^d.
    Clips entries to [-1, 1] and quantises to 2^b uniformly spaced levels.
    Allowed bit-widths W = {1, 2, 4, 8}. Message dimension d = 16.
    
    Uses Straight-Through Estimator (STE) only for the rounding operation.
    """
    def __init__(self, message_dim=16, allowed_widths=(1, 2, 4, 8)):
        super().__init__()
        self.message_dim = message_dim
        self.allowed_widths = allowed_widths
        self.width_to_idx = {w: i for i, w in enumerate(allowed_widths)}
        self.idx_to_width = {i: w for i, w in enumerate(allowed_widths)}

    def forward(self, m, bit_width):
        """
        m: Tensor of shape (..., d)
        bit_width: int, or Tensor of integers matching batch dimensions
        """
        # 1. Clip message entries to [-1, 1]
        m_clipped = torch.clamp(m, -1.0, 1.0)
        
        if isinstance(bit_width, int):
            num_levels = 2 ** bit_width
            levels_minus_one = float(num_levels - 1)
            # Map [-1, 1] to [0, num_levels - 1]
            scaled = (m_clipped + 1.0) * 0.5 * levels_minus_one
            rounded = torch.round(scaled)
            q = -1.0 + 2.0 * (rounded / levels_minus_one)
        else:
            # bit_width is a Tensor
            if bit_width.dim() < m.dim():
                # expand bit_width to have same trailing dimension as m
                bw_expanded = bit_width.unsqueeze(-1).expand_as(m)
            else:
                bw_expanded = bit_width
            num_levels = (2 ** bw_expanded).float()
            levels_minus_one = num_levels - 1.0
            scaled = (m_clipped + 1.0) * 0.5 * levels_minus_one
            rounded = torch.round(scaled)
            q = -1.0 + 2.0 * (rounded / levels_minus_one)

        # Straight-Through Estimator: forward is q, backward gradient is through m_clipped
        q_ste = m_clipped + (q - m_clipped).detach()
        return q_ste
