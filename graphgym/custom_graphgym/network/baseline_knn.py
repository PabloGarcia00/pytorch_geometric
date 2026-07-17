import torch
import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from ..target_utils import active_targets
from ._baseline_common import flatten_window, last_step_calendar_features, stack_true


@register_network("baseline_knn")
class BaselineKNNNetwork(nn.Module):
    """
    Per-node KNN baseline: an "analog ensemble" lookup. For household i,
    finds the k most similar historical windows from that SAME household's
    own cached training windows (never another household's -- this is what
    makes it a per-node fit, not a pooled one) and returns the empirical
    quantiles of the outcomes that followed those windows.

    Non-parametric: has zero nn.Parameters. "Fitting" is a one-shot,
    no-grad pass over the training set (fit_cache(), called by
    custom_graphgym/train/knn_train.py before trainer.validate()/test() --
    trainer.fit() is never called for this network, so there is no
    configure_optimizers()/backward() concern at all). See the
    implementation plan for why this reconciles cleanly with GraphGym's
    Lightning-based training loop despite KNN having no gradient step.

    The neighbor bank is dense per node ([num_nodes, S_train, F]) rather
    than ragged, with a parallel validity mask -- this keeps the whole
    lookup a handful of vectorized ops (torch.cdist / topk / gather /
    quantile) with num_nodes as a leading batch dimension, not a Python loop
    over households.
    """

    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()

        self.num_nodes = cfg.share.num_nodes
        self.targets = active_targets()
        self.weather_mode = cfg.earne_data.weather_mode
        self.feature_mode = cfg.baseline.knn_feature_mode
        assert self.feature_mode in (
            "window",
            "window_calendar",
            "window_calendar_weather",
        ), f"Unknown knn_feature_mode: {self.feature_mode}"
        assert cfg.baseline.knn_distance == "euclidean", (
            f"Only 'euclidean' is implemented, got {cfg.baseline.knn_distance!r}"
        )

        quantiles = torch.tensor(cfg.model.quantiles, dtype=torch.float32)
        self.register_buffer("_quantiles_t", quantiles, persistent=False)

        self._fitted = False

        # Never read in forward() -- exists solely so self.parameters() is
        # non-empty. Both torch.optim.Adam([]) and GraphGym's stock
        # LoggerCallback (torch_geometric/graphgym/logger.py, which
        # unconditionally reads trainer.lr_scheduler_configs[0] on every
        # validation/test batch to log lr) require configure_optimizers()
        # to have produced a real optimizer/scheduler pair. knn_train.py
        # calls trainer.fit() with max_epoch=0 to trigger that setup without
        # any real training step, since this parameter is disconnected from
        # the prediction path entirely.
        self._unused_optim_placeholder = nn.Parameter(torch.zeros(1))

    def _features(self, batch):
        use_weather = self.weather_mode and self.feature_mode == "window_calendar_weather"
        window = flatten_window(batch, weather_mode=use_weather)
        if self.feature_mode == "window":
            return window
        calendar = last_step_calendar_features(batch, self.num_nodes)
        return torch.cat([window, calendar], dim=-1)

    @torch.no_grad()
    def fit_cache(self, train_loader):
        """Populate the per-node neighbor bank from the training split.
        Called once by knn_train.py, outside any Lightning step -- no
        gradient graph is ever built here.
        """
        device = torch.device(cfg.accelerator)
        y_attr = {"load": "y_load", "pv": "y_pv"}
        feats = []
        y_by_target = {t: [] for t in self.targets}
        valid = []

        for batch in train_loader:
            batch = batch.to(device)
            B = batch.num_graphs
            x = self._features(batch).view(B, self.num_nodes, -1)  # [B, N, F]
            feats.append(x)
            for t in self.targets:
                y_by_target[t].append(
                    getattr(batch, y_attr[t]).view(B, self.num_nodes)
                )
            valid.append(batch.mask.view(B, self.num_nodes).bool())

        # [N, S_train, F] / [N, S_train] -- dense per node, invalid targets
        # (mask == 0) are kept in place and excluded at query time via
        # bank_valid rather than compacted out, so every node's bank stays
        # the same size (S_train) and the whole lookup stays vectorized.
        # Only the selected target(s)' y-banks are cached -- halves the
        # (already sizeable) bank memory when only one target is active.
        bank_x = torch.cat(feats, dim=0).permute(1, 0, 2).contiguous()
        bank_valid = torch.cat(valid, dim=0).permute(1, 0).contiguous()

        cache_device = torch.device(cfg.baseline.knn_cache_device)
        self.register_buffer("bank_x", bank_x.to(cache_device), persistent=False)
        self.register_buffer(
            "bank_valid", bank_valid.to(cache_device), persistent=False
        )
        for t in self.targets:
            bank_y = torch.cat(y_by_target[t], dim=0).permute(1, 0).contiguous()
            self.register_buffer(
                f"bank_y_{t}", bank_y.to(cache_device), persistent=False
            )
        self._quantiles_t = self._quantiles_t.to(cache_device)
        self._fitted = True

    def _quantile_lookup(self, bank_y, idx, B):
        y = bank_y.unsqueeze(1).expand(-1, B, -1)  # [N, B, S_train]
        gathered = torch.gather(y, dim=-1, index=idx)  # [N, B, k]
        q_vals = torch.quantile(gathered, self._quantiles_t, dim=-1)  # [Q, N, B]
        return q_vals.permute(2, 1, 0).reshape(B * self.num_nodes, -1)  # [N_total, Q]

    def forward(self, batch):
        if not self._fitted:
            raise RuntimeError(
                "BaselineKNNNetwork.fit_cache() must be called before forward() "
                "-- see custom_graphgym/train/knn_train.py."
            )

        out_device = batch.x.device
        query = self._features(batch)
        B = batch.num_graphs
        q = (
            query.to(self.bank_x.device)
            .view(B, self.num_nodes, -1)
            .permute(1, 0, 2)
        )  # [N, B, F]

        dist = torch.cdist(q, self.bank_x)  # [N, B, S_train]
        dist = dist.masked_fill(~self.bank_valid.unsqueeze(1), float("inf"))
        k = min(cfg.baseline.knn_k, self.bank_x.shape[1])
        _, idx = torch.topk(dist, k, dim=-1, largest=False)  # [N, B, k]

        # One quantile block per selected target, in active_targets() order.
        q_blocks = [
            self._quantile_lookup(getattr(self, f"bank_y_{t}"), idx, B)
            for t in self.targets
        ]
        pred = torch.cat(q_blocks, dim=1).to(out_device)

        true = stack_true(batch)
        return pred, true
