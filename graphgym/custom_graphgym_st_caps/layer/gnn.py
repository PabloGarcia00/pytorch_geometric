import torch
import torch.nn as nn
import torch.nn.functional as F
from ..loader.graphs import LaplacianCalculator

class SpectralGraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, K: int = 1):
        super().__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters for Chebyshev coefficients
        self.beta = nn.Parameter(torch.randn(K + 1, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.beta)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        device = x.device

        # Add self-loops and compute normalized adjacency
        adjacency = adjacency + torch.eye(N, device=device)
        degree = torch.sum(adjacency, dim=1)
        degree_sqrt_inv = torch.pow(degree + 1e-6, -0.5)
        degree_sqrt_inv = torch.diag(degree_sqrt_inv)

        adj_norm = torch.matmul(torch.matmul(degree_sqrt_inv, adjacency), degree_sqrt_inv)

        if self.K == 1:
            out = torch.matmul(adj_norm, x)
            out = torch.matmul(out, self.beta[0])
            return out + self.bias
        else:
            laplacian = LaplacianCalculator.compute_normalized_laplacian(adjacency)
            polynomials = LaplacianCalculator.chebyshev_polynomial(laplacian, self.K)

            out = torch.zeros((N, self.out_features), device=device)
            for k in range(self.K + 1):
                features = torch.matmul(polynomials[k], x)
                out = out + torch.matmul(features, self.beta[k])

            return out + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int = 3):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features

        self.W = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(n_heads)
        ])

        self.attention_kernels = nn.ModuleList([
            nn.Linear(2 * out_features, 1)
            for _ in range(n_heads)
        ])

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        device = x.device

        mask = (adjacency > 0).float()
        mask = mask + torch.eye(N, device=device)  # Include self

        outputs = []

        for r in range(self.n_heads):
            h = self.W[r](x)  # [N, out_features]

            h_i = h.unsqueeze(1).expand(N, N, self.out_features)
            h_j = h.unsqueeze(0).expand(N, N, self.out_features)
            concat = torch.cat([h_i, h_j], dim=-1)  # [N, N, 2*out_features]

            e = self.attention_kernels[r](concat).squeeze(-1)  # [N, N]
            e = F.leaky_relu(e, negative_slope=0.2)

            # Mask non-neighbors
            e = e * mask + (1 - mask) * (-1e9)
            alpha = F.softmax(e, dim=1)  # [N, N]

            out = torch.matmul(alpha, h)  # [N, out_features]
            outputs.append(out)

        output = torch.stack(outputs).mean(dim=0)
        return F.relu(output)
