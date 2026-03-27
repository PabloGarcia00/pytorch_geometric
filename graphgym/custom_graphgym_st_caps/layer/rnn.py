import torch
import torch.nn as nn
from typing import Tuple

class PeepholeLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, use_peephole: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_peephole = use_peephole

        # Input gate
        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        self.W_ci = nn.Linear(hidden_size, hidden_size) if use_peephole else None
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        # Forget gate
        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)
        self.W_cf = nn.Linear(hidden_size, hidden_size) if use_peephole else None
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        # Cell gate
        self.W_ic = nn.Linear(input_size, hidden_size)
        self.W_hc = nn.Linear(hidden_size, hidden_size)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        # Output gate
        self.W_io = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)
        self.W_co = nn.Linear(hidden_size, hidden_size) if use_peephole else None
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        # Depth gate (connects to lower layer)
        self.W_id = nn.Linear(input_size, hidden_size)
        self.W_cd_prev = nn.Linear(hidden_size, hidden_size)
        self.W_cd_lower = nn.Linear(hidden_size, hidden_size)
        self.b_d = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor,
                c_lower: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input gate
        i = self.W_ii(x) + self.W_hi(h_prev)
        if self.use_peephole and self.W_ci is not None:
            i = i + self.W_ci(c_prev)
        i = torch.sigmoid(i + self.b_i)

        # Forget gate
        f = self.W_if(x) + self.W_hf(h_prev)
        if self.use_peephole and self.W_cf is not None:
            f = f + self.W_cf(c_prev)
        f = torch.sigmoid(f + self.b_f)

        # Cell candidate
        c_tilde = torch.tanh(self.W_ic(x) + self.W_hc(h_prev) + self.b_c)

        # Depth gate (if lower layer cell state provided)
        if c_lower is not None:
            d = torch.sigmoid(
                self.W_id(x) + self.W_cd_prev(c_prev) +
                self.W_cd_lower(c_lower) + self.b_d
            )
            c = d * c_lower + f * c_prev + i * c_tilde
        else:
            c = f * c_prev + i * c_tilde

        # Output gate
        o = self.W_io(x) + self.W_ho(h_prev)
        if self.use_peephole and self.W_co is not None:
            o = o + self.W_co(c)
        o = torch.sigmoid(o + self.b_o)

        # Hidden state
        h = o * torch.tanh(c)

        return h, c
