from typing import Optional, Tuple

import torch

from torch_geometric.graphgym.config import cfg


class GraphBuilder:
    def __init__(
        self,
        lambda_threshold: float = 0.25,
        dual_read: bool = False,
        dual_agg: str = "max",  # "max" | "mean" | "concat"
    ):
        self.lambda_threshold = lambda_threshold
        self.dual_read = dual_read
        self.dual_agg = dual_agg

    def build_graph(
        self,
        net_demand: torch.Tensor,
        window: int = 10,
        lambda_threshold: Optional[torch.Tensor] = None,
        batch_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds a correlation-based adjacency matrix.
        If batch_vec is provided, uses torch.bmm for efficient batch-wise
        calculation, avoiding cross-graph correlations.
        """
        N, T = net_demand.shape
        device = net_demand.device
        threshold = (
            lambda_threshold
            if lambda_threshold is not None
            else self.lambda_threshold
        )
        start_idx = max(0, T - window)
        recent_data = net_demand[:, start_idx:]
        eff_T = recent_data.shape[1]

        if batch_vec is not None:
            batch_size = int(batch_vec.max().item() + 1)
            nodes_per_graph = N // batch_size
            data_b = recent_data.view(batch_size, nodes_per_graph, eff_T)

            mean = data_b.mean(dim=2, keepdim=True)
            centered = data_b - mean
            std = centered.std(dim=2, keepdim=True, unbiased=True)
            std[std == 0] = 1e-9
            norm_centered = centered / std

            corr = torch.bmm(norm_centered, norm_centered.transpose(1, 2)) / (
                eff_T - 1
            )
            abs_corr = torch.abs(corr)

            use_topk = cfg.earne_data.get("use_topk", False)
            if use_topk:
                k = min(cfg.earne_data.k_neighbors, nodes_per_graph - 1)
                _, topk_indices = torch.topk(abs_corr, k=k, dim=-1)

                b_range = torch.arange(batch_size, device=device).view(-1, 1, 1)
                r_range = torch.arange(nodes_per_graph, device=device).view(
                    1, -1, 1
                )
                b_idx = b_range.expand(batch_size, nodes_per_graph, k).reshape(
                    -1
                )
                r_idx = r_range.expand(batch_size, nodes_per_graph, k).reshape(
                    -1
                )
                c_idx = topk_indices.reshape(-1)

                self_loop_mask = r_idx != c_idx
                b_idx, r_idx, c_idx = (
                    b_idx[self_loop_mask],
                    r_idx[self_loop_mask],
                    c_idx[self_loop_mask],
                )
                edge_attr = abs_corr[b_idx, r_idx, c_idx].unsqueeze(-1)
            else:
                mask = (
                    torch.sigmoid(100.0 * (abs_corr - threshold))
                    if lambda_threshold is not None
                    else (abs_corr > threshold).float()
                )
                adj_b = torch.exp(abs_corr) * mask
                b_idx, r_idx, c_idx = adj_b.nonzero(as_tuple=True)

                self_loop_mask = r_idx != c_idx
                b_idx, r_idx, c_idx = (
                    b_idx[self_loop_mask],
                    r_idx[self_loop_mask],
                    c_idx[self_loop_mask],
                )
                edge_attr = adj_b[b_idx, r_idx, c_idx].unsqueeze(-1)

            edge_index = torch.stack(
                [
                    b_idx * nodes_per_graph + r_idx,
                    b_idx * nodes_per_graph + c_idx,
                ],
                dim=0,
            )
            return edge_index, edge_attr

        else:
            mean = recent_data.mean(dim=1, keepdim=True)
            centered = recent_data - mean
            std = centered.std(dim=1, unbiased=True)
            std[std == 0] = 1e-9
            norm_centered = centered / std.unsqueeze(1)

            corr = torch.matmul(norm_centered, norm_centered.T) / (eff_T - 1)
            abs_corr = torch.abs(corr)

            mask = (
                torch.sigmoid(100.0 * (abs_corr - threshold))
                if lambda_threshold is not None
                else (abs_corr > threshold).float()
            )
            adjacency = torch.exp(abs_corr) * mask
            edges = adjacency.nonzero()
            self_loop_mask = edges[:, 0] != edges[:, 1]
            edges = edges[self_loop_mask]
            edge_index = edges.t().contiguous()
            edge_attr = adjacency[edges[:, 0], edges[:, 1]].unsqueeze(-1)
            return edge_index, edge_attr

    def build_temporal_graphs(self, net_demand: torch.Tensor, m: int) -> list:
        graphs = []
        for t in range(m + 1):
            edge_index, edge_attr = self.build_graph(net_demand[:, : t + 1])
            graphs.append((edge_index, edge_attr))
        return graphs

    # ── Correlation helpers ───────────────────────────────────────────────────

    def compute_abs_corr(
        self,
        primary: torch.Tensor,  # [N, T] consumption or net_demand
        operational: Optional[torch.Tensor] = None,  # [N, T] float mask
        second_stream: Optional[
            torch.Tensor
        ] = None,  # [N, T] generation — dual only
    ) -> torch.Tensor:
        """
        Compute [N, N] abs correlation matrix.
        In dual_read mode, computes correlation for each stream separately
        and aggregates via self.dual_agg strategy.
        """
        abs_corr_primary = self._compute_single_abs_corr(primary, operational)

        if self.dual_read:
            if second_stream is None:
                raise ValueError(
                    "dual_read=True but second_stream was not provided"
                )
            abs_corr_secondary = self._compute_single_abs_corr(
                second_stream, operational
            )

            if self.dual_agg == "max":
                return torch.maximum(abs_corr_primary, abs_corr_secondary)

            elif self.dual_agg == "mean":
                return (abs_corr_primary + abs_corr_secondary) / 2.0

            elif self.dual_agg == "concat":
                # Concatenate along time dim — valid since each stream is
                # independently normalised before entering this method
                op_combined = (
                    torch.cat([operational, operational], dim=1)
                    if operational is not None
                    else None
                )
                return self._compute_single_abs_corr(
                    torch.cat([primary, second_stream], dim=1),
                    op_combined,
                )
            else:
                raise ValueError(
                    f"Unknown dual_agg {self.dual_agg!r}, choose max | mean | concat"
                )

        return abs_corr_primary

    def _compute_single_abs_corr(
        self,
        net_demand: torch.Tensor,  # [N, T]
        operational: Optional[torch.Tensor] = None,  # [N, T] float mask
    ) -> torch.Tensor:
        """Compute [N, N] abs Pearson correlation for a single stream.
        Inactive timesteps (zero-filled) are excluded via the operational mask
        so they do not bias the mean, std, or dot product.
        """
        N, T = net_demand.shape

        # Normalise by total active load so node magnitudes are comparable
        if operational is not None:
            active_sum = (
                (net_demand * operational)
                .sum(dim=1, keepdim=True)
                .clamp(min=1e-9)
            )
        else:
            active_sum = net_demand.sum(dim=1, keepdim=True).clamp(min=1e-9)

        normalised = (net_demand / active_sum) * (
            operational if operational is not None else 1.0
        )

        if operational is not None:
            active_counts = operational.sum(dim=1, keepdim=True).clamp(
                min=1.0
            )  # [N, 1]
            mean = normalised.sum(dim=1, keepdim=True) / active_counts  # [N, 1]
            centered = (normalised - mean) * operational  # zero inactive
            var = (centered**2).sum(dim=1, keepdim=True) / (
                active_counts - 1
            ).clamp(min=1.0)
            std = var.sqrt().clamp(min=1e-9)  # [N, 1]
            # [N, N] joint active counts — normalise dot product per pair
            joint_counts = torch.matmul(
                operational.float(), operational.float().t()
            ).clamp(min=1.0)
            norm_centered = centered / std  # [N, T]
            corr = torch.matmul(norm_centered, norm_centered.t()) / (
                joint_counts - 1
            ).clamp(min=1.0)
        else:
            mean = normalised.mean(dim=1, keepdim=True)
            centered = normalised - mean
            std = (
                centered.std(dim=1, unbiased=True).clamp(min=1e-9).unsqueeze(1)
            )
            norm_centered = centered / std
            corr = torch.matmul(norm_centered, norm_centered.t()) / (T - 1)

        return torch.abs(corr).clamp(max=1.0)

    def threshold_graph(
        self,
        abs_corr: torch.Tensor,
        lambda_threshold: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply (learnable) threshold to cached abs_corr. Call every forward."""
        threshold = (
            lambda_threshold
            if lambda_threshold is not None
            else self.lambda_threshold
        )

        mask = torch.sigmoid(100.0 * (abs_corr - threshold))
        adjacency = torch.exp(abs_corr) * mask

        edges = adjacency.nonzero()
        self_loop_mask = edges[:, 0] != edges[:, 1]
        edges = edges[self_loop_mask]
        edge_index = edges.t().contiguous()
        edge_attr = adjacency[edges[:, 0], edges[:, 1]].unsqueeze(-1)
        return edge_index, edge_attr


class LaplacianCalculator:
    @staticmethod
    def compute_normalized_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
        N = adjacency.shape[0]
        adjacency = adjacency + torch.eye(N, device=adjacency.device)
        degree = torch.sum(adjacency, dim=1)
        degree_sqrt_inv = torch.pow(degree + 1e-6, -0.5)
        degree_sqrt_inv = torch.diag(degree_sqrt_inv)
        laplacian = torch.eye(N, device=adjacency.device) - torch.matmul(
            torch.matmul(degree_sqrt_inv, adjacency), degree_sqrt_inv
        )
        return laplacian

    @staticmethod
    def chebyshev_polynomial(laplacian: torch.Tensor, K: int = 1) -> list:
        N = laplacian.shape[0]
        device = laplacian.device
        lambda_max = 2.0
        laplacian_scaled = (2.0 / lambda_max) * laplacian - torch.eye(
            N, device=device
        )
        polynomials = [torch.eye(N, device=device)]
        if K >= 1:
            polynomials.append(laplacian_scaled)
        for k in range(2, K + 1):
            poly = (
                2 * torch.matmul(laplacian_scaled, polynomials[-1])
                - polynomials[-2]
            )
            polynomials.append(poly)
        return polynomials
