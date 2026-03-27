import torch
import pandas as pd
import numpy as np
import os
import duckdb
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data, Dataset
from torch_geometric.graphgym.register import register_loader
from torch_geometric.graphgym.config import cfg
from itertools import permutations
from scipy.spatial import distance_matrix

# ==========================================
# 0. NORMALIZATION UTILITIES
# ==========================================
def zero_preserved_log_stats(X):
    """Compute log-space mean/std on strictly positive values only."""
    Y = np.copy(X)
    Y[Y <= 0] = np.nan          # mask zeros AND negatives before log
    Y_log = np.log(Y)
    nonzero_mean = torch.tensor(np.nanmean(Y_log, axis=0, keepdims=True), dtype=torch.float)
    nonzero_std  = torch.tensor(np.nanstd( Y_log, axis=0, keepdims=True), dtype=torch.float)
    nonzero_mean[torch.isnan(nonzero_mean)] = 0
    nonzero_std[nonzero_std == 0]           = 1.0
    nonzero_std[torch.isnan(nonzero_std)]   = 1.0
    return nonzero_mean, nonzero_std

def zero_preserved_log_normalize(X, nonzero_mean, nonzero_std, log_output=True, zero_id=None, shift=1.0):
    """
    Log-normalise strictly positive values; assign zero_id sentinel to all
    non-positive values (zero and negative net demand both map to zero_id).
    """
    if zero_id is None:
        zero_id = cfg.earne_data.zero_id
        
    if isinstance(X, torch.Tensor):
        Y = X.clone()
    else:
        Y = torch.tensor(X, dtype=torch.float)
    is_special = (Y <= 0)          # zeros AND negatives
    Y[is_special] = 1.0            # temporary safe value for log
    Y_log = torch.log(Y)
    Y_log = (Y_log - nonzero_mean) / nonzero_std + shift
    Y_res = Y_log if log_output else torch.exp(Y_log)
    Y_res[is_special] = zero_id
    return Y_res

def zero_preserved_log_denormalize(Y, nonzero_mean, nonzero_std, log_input=True, zero_id=None, shift=1.0):
    if zero_id is None:
        zero_id = cfg.earne_data.zero_id
        
    X = Y.clone()
    # "Snap-to-zero": Neural networks rarely hit an exact sentinel.
    # We treat anything within a reasonable buffer of the zero_id as 0.0.
    # Data values start around -45, sentinel is at -50. Buffer of 2.0 is safe.
    is_special = (X < zero_id + 2.0)
    
    X_log = X if log_input else torch.log(X)
    X_log = (X_log - shift) * nonzero_std + nonzero_mean
    X_res = torch.exp(X_log)
    
    # Force the snap
    X_res[is_special] = 0.0
    return X_res


# ==========================================
# 1. SHARED SPLIT HELPER
# ==========================================
def _calc_splits(timestamps, seq_len, n_raw, train_split, val_split):
    """
    Compute chronological train / val / test sample-index splits.

    Three-year rule (Case A):
        If the data spans >= 3 distinct years, the third year onwards is held
        out as test.  The preceding block is sub-split at ratio
        train_split / val_split  (default ≈ 82 %).

    Short-series fallback (Case B):
        Fewer than 3 years or no timestamps → first 67 % is train+val,
        last 33 % is test.  Same sub-split ratio within the block.

    Args:
        timestamps : list of timestamp-like objects (aligned to raw time
                     dimension) or None.
        seq_len    : model look-back window length.
        n_raw      : total raw time steps available (after limit_t).
        train_split, val_split : floats from cfg.train.

    Returns:
        (train_idx, val_idx, test_idx) — three lists of int sample indices.
    """
    n_samples = n_raw - seq_len - 1
    tv_train_ratio = train_split / val_split

    if timestamps is not None and len(timestamps) > seq_len:
        sample_years = [
            pd.Timestamp(timestamps[i + seq_len]).year for i in range(n_samples)
        ]
        unique_years = sorted(set(sample_years))

        if len(unique_years) >= 3:
            year3 = unique_years[2]
            test_start = next(i for i, y in enumerate(sample_years) if y >= year3)
            train_end = int(test_start * tv_train_ratio)
            return (
                list(range(train_end)),
                list(range(train_end, test_start)),
                list(range(test_start, n_samples)),
            )

    # Case B: short series or no timestamps
    tv_end = int(n_samples * 0.67)
    train_end = int(tv_end * tv_train_ratio)
    return (
        list(range(train_end)),
        list(range(train_end, tv_end)),
        list(range(tv_end, n_samples)),
    )


# ==========================================
# 2. EARNe GRAPH DATASET (MASTER BUNDLE)
# ==========================================
class EARNeGraphDataset(Dataset):
    """
    EARNe Dataset — bundle format v2.

    process() stores RAW (unscaled) energy tensors.  Normalisation is applied
    at load time in __init__ based on cfg.earne_data.norm_mode.  This means
    one bundle supports both 'minmax' and 'zero_log' without re-running the
    expensive DuckDB ETL in process().

    Bundle format v1 (legacy pre-scaled 'net_scaled' key) will raise a clear
    error asking you to re-run process().
    """

    def __init__(self, root, seq_len=96, transform=None, pre_transform=None, pre_filter=None):
        self.seq_len = seq_len
        super().__init__(root, transform, pre_transform, pre_filter)

        # ---- 1. Load bundle ----
        self._bundle = torch.load(self.processed_paths[0], weights_only=False)

        # ---- 1a. Migration guard ----
        if 'net_raw' not in self._bundle:
            if 'net_scaled' in self._bundle:
                raise KeyError(
                    "Bundle is in legacy pre-scaled format (v1). "
                    "Normalisation is now applied at load time — re-run process() "
                    "to generate a v2 bundle with raw energy values."
                )
            raise KeyError(
                f"Bundle missing 'net_raw'. Keys present: {list(self._bundle.keys())}"
            )

        # ---- 1b. Structural validation ----
        import warnings
        required_keys = [
            'net_raw', 'load_raw', 'pv_raw', 'mask_raw',
            'weather_data', 'weather_features', 'temporal_data',
            'pos', 'macs', 'zips',
        ]
        missing = [k for k in required_keys if k not in self._bundle]
        if missing:
            raise KeyError(f"Master bundle is missing keys: {missing}")
        if 'timestamps' not in self._bundle:
            warnings.warn(
                "Master bundle has no 'timestamps' key. Re-run process() to include "
                "timestamps. Prediction timestamps will be unavailable.",
                UserWarning, stacklevel=2,
            )

        n_macs = len(self._bundle['macs'])
        n_zips = len(self._bundle['zips'])
        n_pos  = self._bundle['pos'].shape[0]
        if not (n_macs == n_zips == n_pos):
            raise ValueError(
                f"Bundle metadata mismatch: macs={n_macs}, zips={n_zips}, pos={n_pos}"
            )

        # ---- 2. Master metadata ----
        self.master_macs = self._bundle['macs']
        self.master_zips = self._bundle['zips']

        # ---- 3. Node filtering ----
        self.node_mask    = self._get_node_mask()
        self.node_indices = torch.where(self.node_mask)[0]
        self.num_nodes    = len(self.node_indices)
        if self.num_nodes == 0:
            raise ValueError(
                "Filtering resulted in 0 nodes. Check cfg.earne_data.filter_zips/macs."
            )

        # ---- 4. Time limiting ----
        total_t      = self._bundle['net_raw'].shape[0]
        cadence      = cfg.earne_data.get('cadence_minutes', 15)
        req_t        = cfg.earne_data.days * 24 * (60 // cadence)
        self.limit_t = min(total_t, req_t)

        # ---- 5. Timestamps (needed before norm for split computation) ----
        raw_ts = self._bundle.get('timestamps', None)
        self.timestamps = raw_ts[:self.limit_t] if raw_ts is not None else None

        # ---- 6. Raw energy slices (filtered + time-limited) ----
        net_raw  = self._bundle['net_raw'] [:self.limit_t, self.node_indices]
        load_raw = self._bundle['load_raw'][:self.limit_t, self.node_indices]
        pv_raw   = self._bundle['pv_raw']  [:self.limit_t, self.node_indices]

        # ---- 7. Normalise (anti-leakage: fits only on training portion) ----
        (
            self.net_scaled, self.load_scaled, self.pv_scaled,
            self.import_scaled, self.export_scaled,
        ) = self._apply_normalization(net_raw, load_raw, pv_raw)

        # ---- 8. Other tensors ----
        self.mask_raw      = self._bundle['mask_raw']    [:self.limit_t, self.node_indices]
        self.weather_data  = self._bundle['weather_data'][:self.limit_t, self.node_indices]
        self.temporal_data = self._bundle['temporal_data'][:self.limit_t]
        self.pos           = self._bundle['pos'][self.node_indices]
        self.active_macs   = [self.master_macs[i] for i in self.node_indices]
        self.active_zips   = [self.master_zips[i] for i in self.node_indices]

        # ---- 9. Weather feature indexing ----
        master_feats     = self._bundle['weather_features']
        req_feats        = cfg.earne_data.weather_features
        self.weather_idx = [master_feats.index(f) for f in req_feats if f in master_feats]

        # ---- 10. Graph topology ----
        self.edge_index = self._generate_topology()

        # Satisfy GraphGym requirement for an internal 'data' attribute
        self._data = self.get(0)
        self.data  = self._data

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _apply_normalization(self, net_raw, load_raw, pv_raw):
        """
        Fit a scaler on the training portion only (anti-leakage), then
        normalise all five energy tensors (net, load, pv, import, export).
        Persists scaler artefacts to root/ for use by earne_metric.

        import_raw = net_raw.clamp(min=0)  — power drawn from grid (≥0)
        export_raw = (-net_raw).clamp(min=0) — power pushed to grid (≥0)

        Both use the *same* scaler as net/load/pv to preserve relative
        magnitudes.  In zero_log mode, the zero-sentinel (-3) marks time
        steps where that direction is inactive.

        Returns:
            (net_scaled, load_scaled, pv_scaled, import_scaled, export_scaled)
        """
        norm_mode    = cfg.earne_data.get('norm_mode', 'minmax')
        n_raw        = net_raw.shape[0]
        train_idx, _, _ = _calc_splits(
            self.timestamps, self.seq_len, n_raw,
            cfg.train.train_split, cfg.train.val_split,
        )
        # All raw time steps touched by training samples
        train_end_raw = len(train_idx) + self.seq_len

        import_raw = net_raw.clamp(min=0)
        export_raw = (-net_raw).clamp(min=0)

        if norm_mode == 'minmax':
            train_flat = torch.cat([
                net_raw [:train_end_raw].reshape(-1),
                load_raw[:train_end_raw].reshape(-1),
                pv_raw  [:train_end_raw].reshape(-1),
            ]).numpy().reshape(-1, 1)
            scaler = MinMaxScaler().fit(train_flat)
            joblib.dump(scaler, Path(self.root) / 'scaler.pkl')

            def _scale(x):
                return torch.tensor(
                    scaler.transform(x.numpy().reshape(-1, 1)).reshape(x.shape),
                    dtype=torch.float,
                )
            return (
                _scale(net_raw), _scale(load_raw), _scale(pv_raw),
                _scale(import_raw), _scale(export_raw),
            )

        elif norm_mode == 'zero_log':
            train_flat = torch.cat([
                net_raw [:train_end_raw].reshape(-1),
                load_raw[:train_end_raw].reshape(-1),
                pv_raw  [:train_end_raw].reshape(-1),
            ]).numpy()
            nz_mean, nz_std = zero_preserved_log_stats(train_flat)
            params = {'mode': 'zero_log', 'nz_mean': nz_mean, 'nz_std': nz_std}
            torch.save(params, Path(self.root) / 'norm_params.pt')
            return (
                zero_preserved_log_normalize(net_raw,    nz_mean, nz_std),
                zero_preserved_log_normalize(load_raw,   nz_mean, nz_std),
                zero_preserved_log_normalize(pv_raw,     nz_mean, nz_std),
                zero_preserved_log_normalize(import_raw, nz_mean, nz_std),
                zero_preserved_log_normalize(export_raw, nz_mean, nz_std),
            )

        raise ValueError(f"Unknown norm_mode: {norm_mode!r}")

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    @property
    def processed_file_names(self):
        return ['earne_master_bundle.pt']

    def _get_node_mask(self):
        mask = torch.ones(len(self.master_macs), dtype=torch.bool)
        if cfg.earne_data.filter_zips:
            zip_mask = torch.zeros_like(mask)
            for z in cfg.earne_data.filter_zips:
                zip_mask |= (torch.tensor(self.master_zips) == z)
            mask &= zip_mask
        if cfg.earne_data.filter_macs:
            mac_mask = torch.zeros_like(mask)
            for m in cfg.earne_data.filter_macs:
                for idx, master_m in enumerate(self.master_macs):
                    if master_m == m:
                        mac_mask[idx] = True
            mask &= mac_mask
        return mask

    def _generate_topology(self):
        import warnings
        mode = cfg.earne_data.graph_mode
        if mode == 'full_graph':
            if self.num_nodes > 200:
                warnings.warn(
                    f"full_graph with {self.num_nodes} nodes → {self.num_nodes**2} edges; "
                    "consider spatial_knn to avoid GPU OOM.",
                    stacklevel=2,
                )
            edges = list(permutations(range(self.num_nodes), 2))
            return torch.tensor(edges, dtype=torch.long).t().contiguous()

        elif mode == 'spatial_knn':
            k = min(cfg.earne_data.k_neighbors, self.num_nodes - 1)
            if k <= 0:
                return torch.zeros((2, 0), dtype=torch.long)
            dist_mat = distance_matrix(self.pos, self.pos)
            src, dst = [], []
            for i in range(self.num_nodes):
                for j in np.argsort(dist_mat[i])[1:k + 1]:
                    src.append(i); dst.append(j)
            return torch.tensor([src, dst], dtype=torch.long)

        # 'learned_corr' and unknown: empty static topology
        # (dynamic edges are built per-forward in EARNeNetwork)
        return torch.zeros((2, 0), dtype=torch.long)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def len(self):
        return self.net_scaled.shape[0] - self.seq_len - 1

    def get_split_indices(self):
        """
        Chronological train/val/test splits via _calc_splits().
        Returns (train_idx, val_idx, test_idx).
        """
        return _calc_splits(
            self.timestamps, self.seq_len, self.net_scaled.shape[0],
            cfg.train.train_split, cfg.train.val_split,
        )

    def get(self, idx):
        # x: [Nodes, Seq_Len, Dim_In]
        if cfg.earne_data.dual_read:
            x_import = self.import_scaled[idx : idx + self.seq_len].t()
            x_export = self.export_scaled[idx : idx + self.seq_len].t()
            x = torch.stack([x_import, x_export], dim=-1)  # [N, T, 2]
        else:
            x = self.net_scaled[idx : idx + self.seq_len].t().unsqueeze(-1)  # [N, T, 1]
        tgt    = idx + self.seq_len
        y_load = self.load_scaled[tgt]
        y_pv   = self.pv_scaled[tgt]
        y_mask = self.mask_raw[tgt]

        # [Nodes, 4] packet for earne_loss: (Load, PV, Mask, NetDemand)
        true_packet = torch.stack([y_load, y_pv, y_mask, self.net_scaled[tgt]], dim=1)

        if self.weather_idx and not cfg.earne_data.mask_weather:
            w_step = self.weather_data[tgt][:, self.weather_idx]
        else:
            w_step = torch.zeros((self.num_nodes, len(self.weather_idx)), dtype=torch.float)

        t_broadcast = self.temporal_data[tgt].repeat(self.num_nodes, 1)
        condition_tensor = torch.cat([w_step, t_broadcast], dim=1)
        timestamp = self.timestamps[tgt] if self.timestamps is not None else None

        return Data(
            x=x, y=true_packet, weather=condition_tensor,
            edge_index=self.edge_index, pos=self.pos,
            macs=self.active_macs, zips=self.active_zips,
            y_load=y_load, y_pv=y_pv, y_net_demand=self.net_scaled[tgt],
            mask=y_mask, num_nodes=self.num_nodes, timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # ETL  (runs once; stores raw energy — norm applied at load time)
    # ------------------------------------------------------------------

    def _get_db_connection(self):
        con = duckdb.connect(database=':memory:')
        con.execute(f"CREATE VIEW raw_energy AS SELECT * FROM read_parquet('{cfg.earne_data.clean_hive}/**/*.parquet', hive_partitioning=true)")
        con.execute(f"CREATE VIEW raw_weather AS SELECT * FROM read_parquet('{cfg.earne_data.weather_hive}/**/*.parquet', hive_partitioning=true, union_by_name=true)")
        con.execute(f"CREATE VIEW mac_meta AS SELECT * FROM read_csv('{cfg.earne_data.metadata_csv}', all_varchar=True)")
        con.execute(f"CREATE VIEW zip_to_station AS SELECT * FROM read_csv('{cfg.earne_data.mapping_csv}', all_varchar=True)")

        all_weather = [
            "solar_radiation_avg", "solar_radiation_max", "sunshine_duration_min",
            "air_temperature", "soil_temp_5cm",
        ]
        weather_cols_sql = ", ".join([f"avg(COALESCE({c}, 0)) as avg_{c}" for c in all_weather])
        con.execute(
            f"CREATE VIEW fleet_avg_weather AS "
            f"SELECT timestamp, {weather_cols_sql} FROM raw_weather GROUP BY timestamp"
        )

        imputed_w_sql = ", ".join(
            [f"COALESCE(w.{c}, favg.avg_{c}, 0) as {c}" for c in all_weather]
        )
        con.execute(f"""
            CREATE VIEW energy_joined AS
            SELECT e.*, m.zip_code, {imputed_w_sql}
            FROM raw_energy e
            LEFT JOIN mac_meta m ON e.MAC = m.MAC
            LEFT JOIN zip_to_station z
                ON TRY_CAST(m.zip_code AS INT) = TRY_CAST(z.two_number_zip AS INT)
            LEFT JOIN raw_weather w
                ON e.MessageTimestamp = w.timestamp AND z.weather_station_id = w.station
            LEFT JOIN fleet_avg_weather favg ON e.MessageTimestamp = favg.timestamp
        """)
        return con, all_weather

    def process(self):
        con, all_weather = self._get_db_connection()
        cadence = cfg.earne_data.get('cadence_minutes', 15)

        w_sql = ", ".join([f"avg({c}) as {c}" for c in all_weather])
        con.execute(f"""
            CREATE VIEW energy_resampled AS
            SELECT
                time_bucket(INTERVAL '{cadence} minutes', MessageTimestamp) as timestamp,
                MAC,
                avg(net_demand) as net_demand,
                avg(load)       as load,
                avg(inverter_w) as pv,
                {w_sql}
            FROM energy_joined GROUP BY 1, 2
        """)

        pivot_net = (
            con.execute("PIVOT energy_resampled ON MAC USING first(net_demand) "
                        "GROUP BY timestamp ORDER BY timestamp")
            .df().set_index('timestamp')
        )
        master_macs = pivot_net.columns.tolist()
        pivot_load = (
            con.execute("PIVOT energy_resampled ON MAC USING first(load) "
                        "GROUP BY timestamp ORDER BY timestamp")
            .df().set_index('timestamp').reindex(columns=master_macs)
        )
        pivot_pv = (
            con.execute("PIVOT energy_resampled ON MAC USING first(pv) "
                        "GROUP BY timestamp ORDER BY timestamp")
            .df().set_index('timestamp').reindex(columns=master_macs)
        )

        # Temporal sine/cosine encoding
        ts = pd.to_datetime(pivot_net.index)
        temp_t = torch.tensor(np.stack([
            np.sin(2 * np.pi * ts.month   / 12.0), np.cos(2 * np.pi * ts.month   / 12.0),
            np.sin(2 * np.pi * ts.weekday / 7.0),  np.cos(2 * np.pi * ts.weekday / 7.0),
            np.sin(2 * np.pi * ts.hour    / 24.0), np.cos(2 * np.pi * ts.hour    / 24.0),
        ], axis=1), dtype=torch.float)

        # Metadata alignment
        meta_df = con.execute("SELECT * FROM mac_meta").df().drop_duplicates(subset=['MAC'])
        zip_coords = pd.read_csv(cfg.earne_data.zipcode_coords)
        meta_df['two_num_zip'] = (
            pd.to_numeric(meta_df['zip_code'], errors='coerce').fillna(0).astype(int)
        )
        meta_df = meta_df.merge(
            zip_coords, left_on='two_num_zip', right_on='two_number_zip', how='left'
        )
        meta_df = meta_df.set_index('MAC').reindex(master_macs).reset_index()

        # Weather: always MinMax-scaled (no experiment variation on weather norm)
        T_total = pivot_net.shape[0]
        timestamps_list = pivot_net.index.tolist()
        train_idx_proc, _, _ = _calc_splits(
            timestamps_list, cfg.model.seq_len, T_total,
            cfg.train.train_split, cfg.train.val_split,
        )
        weather_train_end = len(train_idx_proc) + cfg.model.seq_len

        w_list, weather_scalers = [], {}
        for feat in all_weather:
            w_df = (
                con.execute(f"PIVOT energy_resampled ON MAC USING first({feat}) "
                            f"GROUP BY timestamp ORDER BY timestamp")
                .df().set_index('timestamp').reindex(columns=master_macs)
            )
            feat_tensor = torch.tensor(w_df.fillna(0).values, dtype=torch.float)
            feat_scaler = MinMaxScaler().fit(
                feat_tensor[:weather_train_end].reshape(-1, 1).numpy()
            )
            weather_scalers[feat] = feat_scaler
            scaled = feat_scaler.transform(
                feat_tensor.reshape(-1, 1).numpy()
            ).reshape(feat_tensor.shape)
            w_list.append(torch.tensor(scaled, dtype=torch.float))

        joblib.dump(weather_scalers, Path(self.root) / 'weather_scaler.pkl')

        torch.save({
            # v2: raw energy (normalisation applied at load time)
            'net_raw':  torch.tensor(pivot_net .fillna(0).values, dtype=torch.float),
            'load_raw': torch.tensor(pivot_load.fillna(0).values, dtype=torch.float),
            'pv_raw':   torch.tensor(pivot_pv  .fillna(0).values, dtype=torch.float),
            # other tensors
            'mask_raw':         torch.tensor(
                pivot_net.notna().astype(float).values, dtype=torch.float
            ),
            'weather_data':     torch.stack(w_list, dim=-1),
            'weather_features': all_weather,
            'temporal_data':    temp_t,
            'pos':  torch.tensor(
                meta_df[['latitude', 'longitude']].fillna(0).values, dtype=torch.float
            ),
            'macs': master_macs,
            'zips': meta_df['two_num_zip'].values.tolist(),
            'timestamps': timestamps_list,
        }, self.processed_paths[0])


# ==========================================
# 3. LOADER
# ==========================================

@register_loader('earne_loader_new')
def load_earne_dataset(format, name, dataset_dir):
    dataset = EARNeGraphDataset(root=cfg.earne_data.processed_root, seq_len=cfg.model.seq_len)
    dataset.task = 'graph'
    train_idx, val_idx, test_idx = dataset.get_split_indices()
    dataset.data.train_graph_index = torch.tensor(train_idx, dtype=torch.long)
    dataset.data.val_graph_index   = torch.tensor(val_idx,   dtype=torch.long)
    dataset.data.test_graph_index  = torch.tensor(test_idx,  dtype=torch.long)
    return dataset
