from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from custom_graphgym.transform.transform import Transform
from scipy.spatial import distance_matrix

from torch_geometric.data import Data, Dataset
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader


# ==========================================
# 1. HELPERS
# ==========================================
def _calc_splits(timestamps, seq_len, n_raw, train_split, val_split):
    n_samples = n_raw - seq_len - 1
    train_end = int(n_samples * train_split)
    val_end = int(n_samples * val_split)
    indices = list(range(n_samples))
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _encode_temporal(timestamps):
    ts = pd.to_datetime(timestamps)
    return torch.tensor(
        np.stack(
            [
                np.sin(2 * np.pi * ts.month / 12),
                np.cos(2 * np.pi * ts.month / 12),
                np.sin(2 * np.pi * ts.dayofweek / 7),
                np.cos(2 * np.pi * ts.dayofweek / 7),
                np.sin(2 * np.pi * ts.hour / 24),
                np.cos(2 * np.pi * ts.hour / 24),
            ],
            axis=1,
        ),
        dtype=torch.float,
    )


def _pivot_raw(
    df: pl.DataFrame, col: str, master_ids: list[str]
) -> pl.DataFrame:
    """Pivot without filling — preserves nulls for mask and operational derivation."""
    return (
        df.pivot(values=col, index="timestamp", on="user_id")
        .sort("timestamp")
        .select(master_ids)
    )


def _pivot_filled(
    df: pl.DataFrame, col: str, master_ids: list[str]
) -> torch.Tensor:
    """Pivot and fill nulls/NaNs with 0 for use as model input only."""
    return torch.tensor(
        _pivot_raw(df, col, master_ids).fill_nan(0.0).fill_null(0.0).to_numpy(),
        dtype=torch.float,
    )


def _mask_from_pivot(pivot: pl.DataFrame) -> torch.Tensor:
    """Derive boolean [T, N] mask from a raw (unfilled) pivot. True = data present."""
    return (
        torch.tensor(
            pivot.to_numpy(),
            dtype=torch.float32,
        )
        .isnan()
        .logical_not()
    )


# ==========================================
# 2. DATASET
# ==========================================
class EARNeGraphDataset(Dataset):
    def __init__(self, root, seq_len=96, transform=None, pre_transform=None):
        self.seq_len = seq_len
        self.dual_read = cfg.model.dim_in == 2
        super().__init__(root, transform, pre_transform)

        # ── Load master bundle ────────────────────────────────────────────
        master = torch.load(self.processed_paths[0], weights_only=False)

        self.temporal_data = master["temporal_data"]
        self.timestamps = master.get("timestamps")
        master_ids = master["ids"]
        master_zips = master["zips"]
        master_pos = master["pos"]

        # ── Node filtering ────────────────────────────────────────────────
        self.node_mask = self._get_node_mask(
            master_ids, master_zips, master["mask"], self.timestamps
        )
        self.node_indices = torch.where(self.node_mask)[0]
        self.num_nodes = len(self.node_indices)

        # ── Weather data ──────────────────────────────────────────────────────
        self.weather_data = master["weather_data"]
        if self.weather_data is not None:
            self.weather_data = self.weather_data[:, self.node_indices]
            master_features = master["weather_features"]
            req_features = cfg.earne_data.get("weather_features", [])
            weather_idx = [
                master_features.index(f)
                for f in req_features
                if f in master_features
            ]
            self.weather_data = self.weather_data[:, :, weather_idx]

        # ── Energy tensors ────────────────────────────────────────────────
        self.load_scaled = master["load_scaled"][:, self.node_indices]
        self.pv_scaled = master["pv_scaled"][:, self.node_indices]
        self.mask_data = master["mask"][:, self.node_indices]
        self.operational = master["operational"][:, self.node_indices]

        if cfg.earne_data.get("mask_physics_impossible", False):
            phys = master.get("physics_mask")
            if phys is not None:
                self.mask_data = self.mask_data & ~phys[:, self.node_indices]

        if self.dual_read:
            self.consumption_scaled = master["consumption_scaled"][
                :, self.node_indices
            ]
            self.generation_scaled = master["generation_scaled"][
                :, self.node_indices
            ]
        self.net_scaled = master["net_scaled"][:, self.node_indices]

        # ── Transform (for inverse transforms at inference) ───────────────
        self.transform_obj = Transform.load(
            Path(root) / self._transform_filename
        )

        self.pos = master_pos[self.node_indices]
        self.active_ids = [master_ids[i] for i in self.node_indices]
        self.active_zips = [master_zips[i] for i in self.node_indices]

        # ── Topology ──────────────────────────────────────────────────────
        self.edge_index, self.edge_attr = self._generate_topology()

        # ── Static correlation matrix ─────────────────────────────────────
        if cfg.earne_data.graph_mode == "static_corr":
            from .graph_builder import GraphBuilder

            gb = GraphBuilder(
                lambda_threshold=cfg.earne_data.get("lambda_graph", 0.25),
                dual_read=self.dual_read,
                dual_agg=cfg.earne_data.get("dual_agg", "max"),
            )
            # Pass operational mask so inactive (zero-filled) regions are excluded
            # from correlation computation
            self.abs_corr = gb.compute_abs_corr(
                (
                    self.consumption_scaled.t()
                    if self.dual_read
                    else self.net_scaled.t()
                ),
                operational=self.operational.t(),
                second_stream=(
                    self.generation_scaled.t() if self.dual_read else None
                ),
            )
        else:
            self.abs_corr = None

        self._data = self.get(0)
        self.data = self._data

    @property
    def _transform_filename(self) -> str:
        mode = "dual" if self.dual_read else "single"
        return f"transform_{mode}.pt"

    @property
    def processed_file_names(self):
        return [f"data.pt"]

    def _get_node_mask(self, ids, zips, mask_data=None, timestamps=None):
        mask = torch.ones(len(ids), dtype=torch.bool)
        if cfg.earne_data.filter_zips:
            z_filter = torch.tensor(cfg.earne_data.filter_zips)
            zips_tensor = torch.tensor(zips)
            zip_mask = (zips_tensor.unsqueeze(1) == z_filter.unsqueeze(0)).any(
                dim=1
            )
            mask &= zip_mask
        if cfg.earne_data.filter_ids:
            mac_mask = torch.zeros_like(mask)
            for m in cfg.earne_data.filter_ids:
                if m in ids:
                    mac_mask[ids.index(m)] = True
            mask &= mac_mask
        # Optionally require nodes to be active in the first and last week of the dataset.
        # require_full_span=True (default) = optimal coverage, drops late-onboarded / early-offboarded nodes.
        # require_full_span=False = native observation window, all nodes included.
        if mask_data is not None and cfg.earne_data.get("require_full_span", True):
            mask_np = mask_data.numpy()
            T = mask_np.shape[0]
            # Must be active within the first week (672 steps of 15 min) and last week (672 steps)
            window = min(672, T // 2)
            first_active = np.argmax(mask_np, axis=0)
            last_active = T - 1 - np.argmax(mask_np[::-1], axis=0)
            no_on_offboarding = torch.tensor(
                (first_active < window) & (last_active > T - 1 - window),
                dtype=torch.bool,
            )
            mask &= no_on_offboarding
        max_nodes = cfg.earne_data.get("max_nodes", 0)
        if max_nodes > 0:
            idx = torch.where(mask)[0][:max_nodes]
            new_mask = torch.zeros_like(mask)
            new_mask[idx] = True
            mask = new_mask
        return mask

    def _generate_topology(self):
        mode = cfg.earne_data.graph_mode
        if mode == "full_graph":
            edges = list(permutations(range(self.num_nodes), 2))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            dist_mat = distance_matrix(self.pos, self.pos)
            src, dst = edge_index[0].numpy(), edge_index[1].numpy()
            distances = dist_mat[src, dst]
            edge_attr = torch.tensor(
                1.0 / (distances + 1e-6), dtype=torch.float
            ).unsqueeze(-1)
            return edge_index, edge_attr

        elif mode == "spatial_knn":
            k = min(cfg.earne_data.k_neighbors, self.num_nodes - 1)
            if k <= 0:
                return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                    (0, 1)
                )
            dist_mat = distance_matrix(self.pos, self.pos)
            nearest = np.argsort(dist_mat, axis=1)[:, 1 : k + 1]
            src = np.repeat(np.arange(self.num_nodes), k)
            dst = nearest.ravel()
            distances = dist_mat[src, dst]
            edge_attr = torch.tensor(
                1.0 / (distances + 1e-6), dtype=torch.float
            ).unsqueeze(-1)
            edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
            return edge_index, edge_attr

        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))

    def len(self):
        return self.net_scaled.shape[0] - self.seq_len - 1

    def get_split_indices(self):
        return _calc_splits(
            self.timestamps,
            self.seq_len,
            self.net_scaled.shape[0],
            cfg.train.train_split,
            cfg.train.val_split,
        )

    def get(self, idx):
        tgt = idx + self.seq_len

        if self.dual_read:
            # [T, N] -> [N, T, 2]
            x = torch.stack(
                [
                    self.consumption_scaled[idx:tgt].t(),
                    self.generation_scaled[idx:tgt].t(),
                ],
                dim=-1,
            )
        else:
            # [T, N] -> [N, T, 1]
            x = self.net_scaled[idx:tgt].t().unsqueeze(-1)

        w_step = (
            self.weather_data[idx:tgt].permute(1, 0, 2)
            if self.weather_data is not None
            else None
        )

        return Data(
            x=x,
            y_load=self.load_scaled[tgt],
            y_pv=self.pv_scaled[tgt],
            y_net_demand=self.net_scaled[tgt],
            mask=self.mask_data[tgt],
            weather=w_step,
            temporal=self.temporal_data[idx:tgt],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            operational=self.operational[idx:tgt].t().unsqueeze(-1),
            num_nodes=self.num_nodes,
            abs_corr=self.abs_corr,
        )

    # =========================================================================
    # PROCESS
    # =========================================================================

    def process(self):
        (
            df,
            master_ids,
            timestamps,
            mask_raw,
            operational_data,
            temporal_data,
            pos,
            master_zips,
            weather_raw,
            ordered_cols,
            train_end,
            physics_impossible_mask,
        ) = self._process_shared()

        t = Transform()
        if self.dual_read:
            scaled, t = self._process_dual_read(
                df, master_ids, mask_raw, train_end, t
            )
        else:
            scaled, t = self._process_single_read(
                df, master_ids, mask_raw, train_end, t
            )

        # ── Weather normalization — skipped if weather disabled ───────────
        if weather_raw is not None:
            w_mode_cfg = cfg.earne_data.weather_norm_mode
            weather_tensor = torch.tensor(weather_raw, dtype=torch.float)
            weather_scaled = torch.zeros_like(weather_tensor)
            for i, col in enumerate(ordered_cols):
                train_slice = weather_tensor[:train_end, :, i].ravel()
                t.fit(
                    w_mode_cfg, **{col: train_slice}
                )  # fit on train only — no mask needed, already sliced
                weather_scaled[:, :, i] = t.transform(
                    col, weather_tensor[:, :, i].ravel()
                ).reshape(weather_tensor.shape[0], weather_tensor.shape[1])
        else:
            weather_scaled = None

        t.save(Path(self.root) / self._transform_filename)

        torch.save(
            {
                **scaled,
                "mask": mask_raw,
                "physics_mask": physics_impossible_mask,
                "ids": master_ids,
                "zips": master_zips,
                "weather_data": weather_scaled,
                "weather_features": ordered_cols,
                "temporal_data": temporal_data,
                "pos": pos,
                "timestamps": timestamps,
                "operational": operational_data,
            },
            self.processed_paths[0],
        )

    def _process_shared(self):
        # ── Load & filter ─────────────────────────────────────────────────
        gold_data = cfg.earne_data.gold_data
        if not Path(gold_data).exists():
            # Saved configs often bake in the gold_data path of the machine
            # that produced them (e.g. a sagemaker path). Fall back to this
            # machine's copy, which lives at the pytorch_geometric project root.
            local_gold_data = Path(__file__).resolve().parents[3] / "fleet_gold_layer.parquet"
            if local_gold_data.exists():
                gold_data = str(local_gold_data)

        lf = pl.scan_parquet(gold_data).with_columns(
            pl.col("user_id").cast(pl.Utf8)
        )
        start_date = datetime.fromisoformat(cfg.earne_data.start_date).replace(
            tzinfo=timezone.utc
        )
        try:
            end_date = start_date.replace(year=start_date.year + 3)
        except ValueError:
            # Handle leap year Feb 29 fallback
            end_date = start_date.replace(year=start_date.year + 3, day=28)
        lf = lf.filter(
            pl.col("timestamp").is_between(start_date, end_date, closed="none")
        )
        lf = lf.filter(pl.col("zipcode").is_not_null()).with_columns(
            pl.col("zipcode").cast(pl.Int32, strict=False)
        )

        activity_cols = (
            ["consumption_w", "generation_w", "inverter_w", "load_w"]
            if self.dual_read
            else ["net_demand_w", "load_w", "inverter_w"]
        )
        last_ts = (
            lf.filter(pl.any_horizontal(pl.col(activity_cols).abs() > 0))
            .select(pl.col("timestamp").max())
            .collect()
            .item()
        )
        lf = lf.filter(pl.col("timestamp") <= last_ts)
        df = lf.sort(["timestamp", "user_id"]).collect()

        master_ids = sorted([str(i) for i in df["user_id"].unique().to_list()])
        timestamps = sorted(df["timestamp"].unique().to_list())

        # ── Mask from raw pivots — no filling ──────────────────────────────
        # A timestep is only supervised if every column its label is derived
        # from is present. If any activity column is NA the label can't be
        # computed, so the whole timestep must be excluded — not just the
        # column that happened to be missing.
        mask_raw = torch.stack(
            [_mask_from_pivot(_pivot_raw(df, col, master_ids)) for col in activity_cols]
        ).all(dim=0)  # [T, N] True = every activity column present

        # ── Operational timeline from mask ────────────────────────────────
        mask_np = mask_raw.numpy()
        T = mask_np.shape[0]
        first_active = np.argmax(mask_np, axis=0)
        last_active = T - 1 - np.argmax(mask_np[::-1], axis=0)
        t_range = np.arange(T)[:, None]
        operational = (
            (t_range >= first_active[None, :])
            & (t_range <= last_active[None, :])
        ).astype(np.float32)
        operational_data = torch.tensor(operational, dtype=torch.float32)

        # ── Temporal encoding ─────────────────────────────────────────────
        temporal_data = _encode_temporal(timestamps)

        # ── Coordinates ───────────────────────────────────────────────────
        mac_zip = (
            df.group_by("user_id")
            .agg(pl.col("zipcode").first())
            .join(
                pl.DataFrame({"user_id": master_ids}), on="user_id", how="right"
            )
            .with_columns(pl.col("zipcode").fill_null(0).cast(pl.Int64))
        )
        coords_df = pd.read_csv(cfg.earne_data.zipcode_coords)
        zip_to_latlon = coords_df.set_index("zipcode")[
            ["latitude", "longitude"]
        ]
        zips = mac_zip["zipcode"].to_pandas()
        latlon = zip_to_latlon.reindex(zips).fillna(0.0).to_numpy()
        pos = torch.tensor(latlon, dtype=torch.float)
        master_zips = zips.tolist()

        # ── Weather raw [T, N, W] ─────────────────────────────────────────
        weather_raw, ordered_cols = self._process_weather(df, master_ids)

        # ── Train end index ───────────────────────────────────────────────
        train_idx, _, _ = _calc_splits(
            timestamps,
            cfg.model.seq_len,
            mask_raw.shape[0],
            cfg.train.train_split,
            cfg.train.val_split,
        )
        train_end = len(train_idx) + cfg.model.seq_len

        # ── Physics impossibility mask ────────────────────────────────────
        # Flag (T, N) entries where generation_w > inverter_w — physically impossible.
        # Saved in the bundle so the flag is config-toggleable without reprocessing.
        if "generation_w" in df.columns and "inverter_w" in df.columns:
            gen_piv = _pivot_filled(df, "generation_w", master_ids)   # [T, N]
            inv_piv = _pivot_filled(df, "inverter_w", master_ids)     # [T, N]
            physics_impossible_mask = (gen_piv - inv_piv) > 0
        else:
            physics_impossible_mask = torch.zeros_like(mask_raw, dtype=torch.bool)

        return (
            df,
            master_ids,
            timestamps,
            mask_raw,
            operational_data,
            temporal_data,
            pos,
            master_zips,
            weather_raw,
            ordered_cols,
            train_end,
            physics_impossible_mask,
        )

    def _process_weather(
        self,
        df: pl.DataFrame,
        master_ids: list[str],
    ) -> tuple[np.ndarray | None, list[str]]:
        """
        Build raw weather array [T, N, W].
        Returns (None, []) if weather is disabled via config.
        """
        weather_cols = cfg.earne_data.get("weather_features", [])
        use_weather = cfg.earne_data.get("use_weather", True)

        if not use_weather or not weather_cols:
            return None, []

        mean_cols = [f for f in weather_cols if f.endswith("_mean")]
        std_cols = [f for f in weather_cols if f.endswith("_std")]
        # Flag-style columns (e.g. is_clear_sky_day) have no _mean/_std pair
        # and pass through unpaired, instead of being swept into mean_cols.
        other_cols = [
            f for f in weather_cols if f not in mean_cols and f not in std_cols
        ]

        if mean_cols and std_cols:
            assert len(mean_cols) == len(
                std_cols
            ), f"Mismatch: {len(mean_cols)} mean cols vs {len(std_cols)} std cols."
            assert set(f.replace("_mean", "_std") for f in mean_cols) == set(
                std_cols
            ), f"Mean/std pairs don't match.\nMeans: {mean_cols}\nStds: {std_cols}"

        ordered_cols = mean_cols + std_cols + other_cols

        weather_arrays = []
        for col in ordered_cols:
            arr = (
                df.pivot(index="timestamp", on="user_id", values=col)
                .sort("timestamp")
                .select(master_ids)
                .cast(pl.Float32)  # non-float cols (e.g. boolean flags) need
                # a numeric dtype before forward_fill/fill_null can run
                .select(pl.all().forward_fill().fill_null(0.0))
                .to_numpy()
            )
            weather_arrays.append(arr)

        return np.stack(weather_arrays, axis=-1), ordered_cols  # [T, N, W]

    def _process_single_read(
        self, df, master_ids, mask_raw, train_end, t: Transform
    ):
        net_raw = _pivot_filled(df, "net_demand_w", master_ids)
        load_raw = _pivot_filled(df, "load_w", master_ids)
        pv_raw = _pivot_filled(df, "inverter_w", master_ids)

        # fit on training data only — mask selects valid (non-NA) entries within train slice
        train_mask = mask_raw[:train_end].bool()
        t.fit(
            cfg.earne_data.energy_norm_mode,
            mask=train_mask,
            net_demand=net_raw[:train_end],
            load=load_raw[:train_end],
            pv=pv_raw[:train_end],
        )

        # transform full tensors after fitting
        return {
            "net_scaled": t.transform("net_demand", net_raw),
            "load_scaled": t.transform("load", load_raw),
            "pv_scaled": t.transform("pv", pv_raw),
        }, t

    def _process_dual_read(
        self, df, master_ids, mask_raw, train_end, t: Transform
    ):
        consumption_raw = _pivot_filled(df, "consumption_w", master_ids)
        generation_raw = _pivot_filled(df, "generation_w", master_ids)
        load_raw = _pivot_filled(df, "load_w", master_ids)
        pv_raw = _pivot_filled(df, "inverter_w", master_ids)
        net_raw = _pivot_filled(df, "net_demand_w", master_ids)

        train_mask = mask_raw[:train_end].bool()
        t.fit(
            cfg.earne_data.energy_norm_mode,
            mask=train_mask,
            consumption=consumption_raw[:train_end],
            generation=generation_raw[:train_end],
            load=load_raw[:train_end],
            pv=pv_raw[:train_end],
            net_demand=net_raw[:train_end],
        )

        return {
            "consumption_scaled": t.transform("consumption", consumption_raw),
            "generation_scaled": t.transform("generation", generation_raw),
            "load_scaled": t.transform("load", load_raw),
            "pv_scaled": t.transform("pv", pv_raw),
            "net_scaled": t.transform("net_demand", net_raw),
        }, t


@register_loader("earne_loader_new")
def load_earne_dataset(format, name, dataset_dir):
    dataset = EARNeGraphDataset(
        root=cfg.earne_data.processed_root, seq_len=cfg.model.seq_len
    )
    dataset.task = "graph"
    train_idx, val_idx, test_idx = dataset.get_split_indices()
    dataset.data.train_graph_index = torch.tensor(train_idx, dtype=torch.long)
    dataset.data.val_graph_index = torch.tensor(val_idx, dtype=torch.long)
    dataset.data.test_graph_index = torch.tensor(test_idx, dtype=torch.long)
    # Mirrors upstream GraphGym's cfg.share.dim_in/dim_out (set from the
    # dataset in torch_geometric/graphgym/loader.py) -- makes the fixed,
    # per-run household count available to per-node baseline networks
    # (custom_graphgym/network/baseline_*.py) at __init__ time, before
    # create_model() runs.
    cfg.share.num_nodes = dataset.num_nodes
    return dataset
