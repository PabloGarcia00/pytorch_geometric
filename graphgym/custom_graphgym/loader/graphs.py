import torch
from typing import Optional, Tuple


class GraphBuilder:
    def __init__(self, lambda_threshold: float = 0.25):
        self.lambda_threshold = lambda_threshold

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
        std[std == 0] = 1e-9
        norm_centered = centered / std.unsqueeze(1)
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
