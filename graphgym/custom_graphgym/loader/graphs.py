import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphBuilder:
    def __init__(self, lambda_threshold: float = 0.25):
        self.lambda_threshold = lambda_threshold

    def build_graphs_all(
        self,
        signal: torch.Tensor,
        window: int = 10,
        lambda_threshold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Vectorized equivalent of calling `build_graph(signal[:, :t+1], window,
        lambda_threshold)` for every end-position t = 0..T-1 in one batched pass,
        instead of a Python loop issuing T separate correlation computations.

        Each position only ever looks at its own trailing window (<= `window`
        samples ending at t), so positions are independent and can be computed
        together via `unfold` + a validity mask (early positions have a shorter,
        left-padded window; the mask keeps the padding out of the mean/std/corr).

        Returns adjacency of shape [T, N, N]. Index 0 is degenerate (single-sample
        history, mirrors build_graph's own T<2 guard) - callers should special-case
        t=0 as a zero matrix, same as the original per-t loop did.
        """
        N, T = signal.shape
        device = signal.device
        threshold = self.lambda_threshold

        padded = F.pad(signal, (window - 1, 0))            # [N, T + window - 1]
        windows = padded.unfold(1, window, 1)               # [N, T, window]

        valid_len = torch.clamp(torch.arange(T, device=device) + 1, max=window)  # [T]
        slot_idx = torch.arange(window, device=device).unsqueeze(0)              # [1, window]
        valid_mask = (slot_idx >= (window - valid_len.unsqueeze(1))).float()      # [T, window]
        valid_mask_b = valid_mask.unsqueeze(0)                                    # [1, T, window]

        count = valid_len.clamp(min=1).float()              # [T]
        denom = (valid_len - 1).clamp(min=1).float()         # [T]

        masked = windows * valid_mask_b
        mean = masked.sum(dim=-1) / count                    # [N, T]
        centered = (windows - mean.unsqueeze(-1)) * valid_mask_b  # [N, T, window]
        var = (centered ** 2).sum(dim=-1) / denom            # [N, T]
        std = torch.sqrt(var)

        # Same rationale as build_graph: degenerate (near-constant) nodes have
        # std ~0, so dividing by it would blow up into spurious correlation.
        degenerate = std < 1e-6
        std_safe = std.clone()
        std_safe[degenerate] = 1.0
        norm_centered = (centered / std_safe.unsqueeze(-1)) * valid_mask_b
        norm_centered[degenerate] = 0.0

        nc = norm_centered.permute(1, 0, 2)                  # [T, N, window]
        corr = torch.bmm(nc, nc.transpose(1, 2)) / denom.view(-1, 1, 1)  # [T, N, N]
        abs_corr = torch.abs(corr)

        if lambda_threshold is not None:
            mask_thr = torch.sigmoid(100.0 * (abs_corr - lambda_threshold))
        else:
            mask_thr = (abs_corr > threshold).float()
        adjacency = torch.nan_to_num(torch.exp(abs_corr) * mask_thr)
        return adjacency

    def build_graph(
        self,
        net_demand: torch.Tensor,
        window: int = 10,
        lambda_threshold: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, T = net_demand.shape
        device = net_demand.device
        threshold = lambda_threshold if lambda_threshold is not None else self.lambda_threshold
        start_idx = max(0, T - window)
        if T - start_idx < 2:
            return torch.zeros(N, N, device=device), net_demand[:, -1]
        recent_data = net_demand[:, start_idx:]
        mean = recent_data.mean(dim=1, keepdim=True)
        centered = recent_data - mean
        std = centered.std(dim=1, unbiased=True)
        # Constant/inactive (zero-filled) nodes have std ~0. Dividing their
        # near-zero floating-point residual by a tiny epsilon amplifies noise
        # into spurious, unbounded "self-correlation" that overflows exp()
        # below — zero those rows outright instead of dividing into them.
        degenerate = std < 1e-6
        std_safe = std.clone()
        std_safe[degenerate] = 1.0
        norm_centered = centered / std_safe.unsqueeze(1)
        norm_centered[degenerate] = 0.0
        eff_T = recent_data.shape[1]
        corr = torch.matmul(norm_centered, norm_centered.T) / (eff_T - 1)
        abs_corr = torch.abs(corr)
        if lambda_threshold is not None:
            mask = torch.sigmoid(100.0 * (abs_corr - threshold))
        else:
            mask = (abs_corr > threshold).float()
        adjacency = torch.nan_to_num(torch.exp(abs_corr) * mask)
        return adjacency, net_demand[:, -1]


class LaplacianCalculator:
    @staticmethod
    def compute_normalized_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
        N = adjacency.shape[0]
        adjacency = adjacency + torch.eye(N, device=adjacency.device)
        degree = torch.sum(adjacency, dim=1)
        degree_sqrt_inv = torch.diag(torch.pow(degree + 1e-6, -0.5))
        adj_norm = degree_sqrt_inv @ adjacency @ degree_sqrt_inv
        return torch.eye(N, device=adjacency.device) - adj_norm

    @staticmethod
    def chebyshev_polynomial(laplacian: torch.Tensor, K: int = 1) -> list:
        N = laplacian.shape[0]
        device = laplacian.device
        laplacian_scaled = (2.0 / 2.0) * laplacian - torch.eye(N, device=device)
        polynomials = [torch.eye(N, device=device)]
        if K >= 1:
            polynomials.append(laplacian_scaled)
        for k in range(2, K + 1):
            poly = 2 * torch.matmul(laplacian_scaled, polynomials[-1]) - polynomials[-2]
            polynomials.append(poly)
        return polynomials
