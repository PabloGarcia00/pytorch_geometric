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
    Y = np.copy(X)
    is_zero = (Y == 0)
    Y[is_zero] = np.nan
    Y_log = np.log(Y)
    nonzero_mean = torch.tensor(np.nanmean(Y_log, axis=0, keepdims=True), dtype=torch.float)
    nonzero_std = torch.tensor(np.nanstd(Y_log, axis=0, keepdims=True), dtype=torch.float)
    nonzero_mean[torch.isnan(nonzero_mean)] = 0
    nonzero_std[nonzero_std == 0] = 1.0
    nonzero_std[torch.isnan(nonzero_std)] = 1.0
    return nonzero_mean, nonzero_std

def zero_preserved_log_normalize(X, nonzero_mean, nonzero_std, log_output=True, zero_id=-3, shift=1.0):
    """
    X: Input numpy array or torch tensor.
    log_output: If True, returns Y_log. If False, returns exp(Y_log).
    """
    if isinstance(X, torch.Tensor):
        Y = X.clone()
    else:
        Y = torch.tensor(X, dtype=torch.float)
    
    is_zero = (Y == 0)
    Y[is_zero] = 1.0 # Temporary to avoid log(0)
    Y_log = torch.log(Y)
    Y_log = (Y_log - nonzero_mean) / nonzero_std + shift
    
    if log_output:
        Y_res = Y_log
    else:
        Y_res = torch.exp(Y_log)
        
    Y_res[is_zero] = zero_id
    return Y_res

def zero_preserved_log_denormalize(Y, nonzero_mean, nonzero_std, log_input=True, zero_id=-3, shift=1.0):
    """
    Y: Input normalized tensor.
    log_input: If True, assumes Y is in log-space.
    """
    X = Y.clone()
    is_zero = (X == zero_id)
    X[is_zero] = 0.0 # Temporary
    
    if log_input:
        X_log = X
    else:
        X_log = torch.log(X)
        
    X_log = (X_log - shift) * nonzero_std + nonzero_mean
    X_res = torch.exp(X_log)
    X_res[is_zero] = 0.0
    return X_res

# ==========================================
# 1. EARNe GRAPH DATASET (MASTER BUNDLE)
# ==========================================
class EARNeGraphDataset(Dataset):
    """
    EARNe Dataset optimized for:
    1. Information Replacement (Weather Masking)
    2. Spatial Isolation (k=0-N experiments)
    3. Zero-Shot Geographic Transfer (MAC/Zip Filtering)
    """
    def __init__(self, root, seq_len=96, transform=None, pre_transform=None, pre_filter=None):
        self.seq_len = seq_len
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # 1. Load the Master Bundle
        self._bundle = torch.load(self.processed_paths[0], weights_only=False)

        # 1a. Validate bundle structure
        import warnings
        required_keys = [
            'net_scaled', 'load_scaled', 'pv_scaled', 'mask_raw',
            'weather_data', 'weather_features', 'temporal_data',
            'pos', 'macs', 'zips',
        ]
        missing = [k for k in required_keys if k not in self._bundle]
        if missing:
            raise KeyError(f"Master bundle is missing keys: {missing}")
        
        # Dual-read optional for backward compatibility
        if 'import_scaled' not in self._bundle or 'export_scaled' not in self._bundle:
            warnings.warn("Bundle missing 'import_scaled' or 'export_scaled'. Run process() to include them.")

        if 'timestamps' not in self._bundle:
            warnings.warn(
                "Master bundle has no 'timestamps' key. Re-run process() to include timestamps. "
                "Prediction timestamps will be unavailable.",
                UserWarning,
                stacklevel=2,
            )

        n_macs = len(self._bundle['macs'])
        n_zips = len(self._bundle['zips'])
        n_pos = self._bundle['pos'].shape[0]
        if not (n_macs == n_zips == n_pos):
            raise ValueError(
                f"Bundle metadata length mismatch: macs={n_macs}, zips={n_zips}, pos={n_pos}"
            )

        # 2. Extract Master Metadata
        self.master_macs = self._bundle['macs']
        self.master_zips = self._bundle['zips']
        
        # 3. Apply Node Filtering (Transfer/Isolation Experiments)
        self.node_mask = self._get_node_mask()
        self.node_indices = torch.where(self.node_mask)[0]
        self.num_nodes = len(self.node_indices)
        
        if self.num_nodes == 0:
            raise ValueError("Filtering resulted in 0 nodes. Check cfg.earne_data.filter_zips/macs.")

        # 4. Filter Core Tensors (Views into the requested slice)
        total_t = self._bundle['net_scaled'].shape[0]
        cadence = cfg.earne_data.get('cadence_minutes', 15)
        requested_t = cfg.earne_data.days * 24 * (60 // cadence)
        self.limit_t = min(total_t, requested_t)
        
        # Sliced views: [T_limit, N_filtered, Features]
        self.net_scaled = self._bundle['net_scaled'][:self.limit_t, self.node_indices]
        self.import_scaled = self._bundle.get('import_scaled', self.net_scaled)[:self.limit_t, self.node_indices]
        self.export_scaled = self._bundle.get('export_scaled', self.net_scaled)[:self.limit_t, self.node_indices]
        self.load_scaled = self._bundle['load_scaled'][:self.limit_t, self.node_indices]
        self.pv_scaled = self._bundle['pv_scaled'][:self.limit_t, self.node_indices]
        self.mask_raw = self._bundle['mask_raw'][:self.limit_t, self.node_indices]
        self.weather_data = self._bundle['weather_data'][:self.limit_t, self.node_indices]
        
        # Global Metadata (Filtered)
        self.temporal_data = self._bundle['temporal_data'][:self.limit_t]
        self.pos = self._bundle['pos'][self.node_indices]
        self.active_macs = [self.master_macs[i] for i in self.node_indices]
        self.active_zips = [self.master_zips[i] for i in self.node_indices]
        self.timestamps = self._bundle.get('timestamps', None)
        if self.timestamps is not None:
            self.timestamps = self.timestamps[:self.limit_t]
        
        # 5. Dynamic Weather Feature Indexing
        master_feats = self._bundle['weather_features']
        req_feats = cfg.earne_data.weather_features
        self.weather_idx = [master_feats.index(f) for f in req_feats if f in master_feats]
        
        # 6. Topology Generation
        self.edge_index = self._generate_topology()
        
        # Satisfy GraphGym requirement for an internal 'data' attribute
        self._data = self.get(0)
        self.data = self._data

    @property
    def processed_file_names(self):
        return ['earne_master_bundle.pt']

    def _get_node_mask(self):
        mask = torch.ones(len(self.master_macs), dtype=torch.bool)
        
        # Filter by Zip Code
        if cfg.earne_data.filter_zips:
            zip_mask = torch.zeros_like(mask)
            for z in cfg.earne_data.filter_zips:
                zip_mask |= (torch.tensor(self.master_zips) == z)
            mask &= zip_mask
            
        # Filter by MAC Address
        if cfg.earne_data.filter_macs:
            mac_mask = torch.zeros_like(mask)
            for m in cfg.earne_data.filter_macs:
                # Direct string comparison is slow, but MAC lists are usually small
                indices = [i for i, master_m in enumerate(self.master_macs) if master_m == m]
                for idx in indices:
                    mac_mask[idx] = True
            mask &= mac_mask
            
        return mask

    def _generate_topology(self):
        import warnings
        mode = cfg.earne_data.graph_mode
        if mode == 'full_graph':
            if self.num_nodes > 200:
                warnings.warn(
                    f"full_graph with {self.num_nodes} nodes produces "
                    f"{self.num_nodes ** 2} edges — consider using spatial_knn to avoid GPU OOM.",
                    stacklevel=2,
                )
            edges = list(permutations(range(self.num_nodes), 2))
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        elif mode == 'spatial_knn':
            k = cfg.earne_data.k_neighbors
            # Safety Check: k cannot exceed num neighbors
            k = min(k, self.num_nodes - 1)
            
            if k <= 0:
                return torch.zeros((2, 0), dtype=torch.long)
            
            dist_mat = distance_matrix(self.pos, self.pos)
            src, dst = [], []
            for i in range(self.num_nodes):
                # Argsort indices, index 0 is node i itself
                nearest = np.argsort(dist_mat[i])[1:k+1]
                for j in nearest:
                    src.append(i); dst.append(j)
            return torch.tensor([src, dst], dtype=torch.long)
        
        return torch.zeros((2, 0), dtype=torch.long)

    def len(self):
        return self.net_scaled.shape[0] - self.seq_len - 1

    def get_split_indices(self):
        """
        Compute chronological train/val/test index splits.

        Three-year rule (Case A):
            If data spans >= 3 distinct years, the third year onwards is held out as
            test.  The preceding block is sub-split at ratio
            cfg.train.train_split / cfg.train.val_split (default ≈ 82%).

        Short-series fallback (Case B):
            If fewer than 3 years are available (common in pilot/ablation runs with
            cfg.earne_data.days < 730), the first 67% is used for train+val and the
            remaining 33% for test.  The same sub-split ratio is applied within the
            train+val block.

        Returns:
            train_idx, val_idx, test_idx — three lists of integer sample indices.
        """
        n = self.len()
        train_split = cfg.train.train_split
        val_split = cfg.train.val_split
        # Fraction of the train+val block that becomes training
        tv_train_ratio = train_split / val_split

        if self.timestamps is not None and len(self.timestamps) > self.seq_len:
            import pandas as pd

            # Map each sample index to its *target* timestamp year
            sample_years = [
                pd.Timestamp(self.timestamps[i + self.seq_len]).year
                for i in range(n)
            ]
            unique_years = sorted(set(sample_years))

            if len(unique_years) >= 3:
                # Case A: chronological 3-year split
                year3 = unique_years[2]
                test_start = next(i for i, y in enumerate(sample_years) if y >= year3)
                train_end = int(test_start * tv_train_ratio)
                return (
                    list(range(train_end)),
                    list(range(train_end, test_start)),
                    list(range(test_start, n)),
                )

        # Case B: short / no-timestamp fallback
        tv_end = int(n * 0.67)
        train_end = int(tv_end * tv_train_ratio)
        return (
            list(range(train_end)),
            list(range(train_end, tv_end)),
            list(range(tv_end, n)),
        )

    def get(self, idx):
        # x: [Nodes, Seq_Len, Dim_In]
        if cfg.earne_data.get('dual_read', False):
            x_import = self.import_scaled[idx : idx + self.seq_len].t()
            x_export = self.export_scaled[idx : idx + self.seq_len].t()
            x = torch.stack([x_import, x_export], dim=-1)
        else:
            x = self.net_scaled[idx : idx + self.seq_len].t().unsqueeze(-1)
        
        target_idx = idx + self.seq_len
        y_load = self.load_scaled[target_idx]
        y_pv = self.pv_scaled[target_idx]
        y_mask = self.mask_raw[target_idx]
        
        # Quantile Target Packet: [Nodes, 4] -> (Load, PV, Mask, NetDemand)
        true_packet = torch.stack([y_load, y_pv, y_mask, self.net_scaled[target_idx]], dim=1)
        
        # Weather Processing with "Information Replacement" Toggle
        if self.weather_idx and not cfg.earne_data.mask_weather:
            w_step = self.weather_data[target_idx][:, self.weather_idx]
        else:
            w_step = torch.zeros((self.num_nodes, len(self.weather_idx)), dtype=torch.float)
        
        # Global Temporal broadcast
        t_step = self.temporal_data[target_idx]
        t_broadcast = t_step.repeat(self.num_nodes, 1)
        condition_tensor = torch.cat([w_step, t_broadcast], dim=1)

        timestamp = self.timestamps[target_idx] if self.timestamps is not None else None

        return Data(
            x=x, y=true_packet, weather=condition_tensor,
            edge_index=self.edge_index, pos=self.pos,
            macs=self.active_macs, zips=self.active_zips,
            y_load=y_load, y_pv=y_pv, y_net_demand=self.net_scaled[target_idx],
            mask=y_mask, num_nodes=self.num_nodes, timestamp=timestamp
        )

    def _get_db_connection(self):
        con = duckdb.connect(database=':memory:')
        con.execute(f"CREATE VIEW raw_energy AS SELECT * FROM read_parquet('{cfg.earne_data.clean_hive}/**/*.parquet', hive_partitioning=true)")
        con.execute(f"CREATE VIEW raw_weather AS SELECT * FROM read_parquet('{cfg.earne_data.weather_hive}/**/*.parquet', hive_partitioning=true, union_by_name=true)")
        con.execute(f"CREATE VIEW mac_meta AS SELECT * FROM read_csv('{cfg.earne_data.metadata_csv}', all_varchar=True)")
        con.execute(f"CREATE VIEW zip_to_station AS SELECT * FROM read_csv('{cfg.earne_data.mapping_csv}', all_varchar=True)")

        all_weather = [
            "solar_radiation_avg", "solar_radiation_max", "sunshine_duration_min", 
            "air_temperature", "soil_temp_5cm"
        ]
        weather_cols_sql = ", ".join([f"avg(COALESCE({c}, 0)) as avg_{c}" for c in all_weather])
        con.execute(f"CREATE VIEW fleet_avg_weather AS SELECT timestamp, {weather_cols_sql} FROM raw_weather GROUP BY timestamp")

        # Two-Tier Imputation FIXED logic
        imputed_w_sql = ", ".join([f"COALESCE(w.{c}, favg.avg_{c}, 0) as {c}" for c in all_weather])

        con.execute(f"""
            CREATE VIEW energy_joined AS
            SELECT e.*, m.zip_code, {imputed_w_sql}
            FROM raw_energy e
            LEFT JOIN mac_meta m ON e.MAC = m.MAC
            LEFT JOIN zip_to_station z ON TRY_CAST(m.zip_code AS INT) = TRY_CAST(z.two_number_zip AS INT)
            LEFT JOIN raw_weather w ON e.MessageTimestamp = w.timestamp AND z.weather_station_id = w.station
            LEFT JOIN fleet_avg_weather favg ON e.MessageTimestamp = favg.timestamp
        """)
        return con, all_weather

    def process(self):
        con, all_weather = self._get_db_connection()
        cadence = cfg.earne_data.get('cadence_minutes', 15)
        
        w_sql = ", ".join([f"avg({c}) as {c}" for c in all_weather])
        con.execute(f"""
            CREATE VIEW energy_resampled AS
            SELECT time_bucket(INTERVAL '{cadence} minutes', MessageTimestamp) as timestamp, MAC,
                avg(net_demand) as net_demand, avg(load) as load, avg(inverter_w) as pv, {w_sql}
            FROM energy_joined GROUP BY 1, 2
        """)

        # Strict Master PIVOT Alignment
        pivot_net = con.execute("PIVOT energy_resampled ON MAC USING first(net_demand) GROUP BY timestamp ORDER BY timestamp").df().set_index('timestamp')
        master_macs = pivot_net.columns.tolist()
        
        pivot_load = con.execute("PIVOT energy_resampled ON MAC USING first(load) GROUP BY timestamp ORDER BY timestamp").df().set_index('timestamp').reindex(columns=master_macs)
        pivot_pv = con.execute("PIVOT energy_resampled ON MAC USING first(pv) GROUP BY timestamp ORDER BY timestamp").df().set_index('timestamp').reindex(columns=master_macs)
        
        # Temporal sine/cosine
        ts = pd.to_datetime(pivot_net.index)
        temp_t = torch.tensor(np.stack([
            np.sin(2 * np.pi * ts.month / 12.0), np.cos(2 * np.pi * ts.month / 12.0),
            np.sin(2 * np.pi * ts.weekday / 7.0), np.cos(2 * np.pi * ts.weekday / 7.0),
            np.sin(2 * np.pi * ts.hour / 24.0), np.cos(2 * np.pi * ts.hour / 24.0)
        ], axis=1), dtype=torch.float)

        # Meta alignment
        meta_df = con.execute("SELECT * FROM mac_meta").df().drop_duplicates(subset=['MAC'])
        zip_coords = pd.read_csv(cfg.earne_data.zipcode_coords)
        meta_df['two_num_zip'] = pd.to_numeric(meta_df['zip_code'], errors='coerce').fillna(0).astype(int)
        meta_df = meta_df.merge(zip_coords, left_on='two_num_zip', right_on='two_number_zip', how='left')
        meta_df = meta_df.set_index('MAC').reindex(master_macs).reset_index()
        
        # Raw Weather Pivoting
        w_list = []
        for feat in all_weather:
            w_df = con.execute(f"PIVOT energy_resampled ON MAC USING first({feat}) GROUP BY timestamp ORDER BY timestamp").df().set_index('timestamp').reindex(columns=master_macs)
            w_list.append(torch.tensor(w_df.fillna(0).values, dtype=torch.float))
        w_tensor = torch.stack(w_list, dim=-1)

        # Weather Scaling
        train_end = int(cfg.train.train_split * pivot_net.shape[0])
        w_list_scaled = []
        weather_scalers = {}
        for i, feat in enumerate(all_weather):
            # Extract individual feature tensor [T, N]
            feat_tensor = w_tensor[:, :, i]
            # Fit on training data: flatten to [T_train * N, 1]
            train_feat = feat_tensor[:train_end].reshape(-1, 1).numpy()
            feat_scaler = MinMaxScaler().fit(train_feat)
            weather_scalers[feat] = feat_scaler
            
            # Transform full tensor
            scaled_feat = feat_scaler.transform(feat_tensor.reshape(-1, 1).numpy()).reshape(feat_tensor.shape)
            w_list_scaled.append(torch.tensor(scaled_feat, dtype=torch.float))
            
        w_tensor_scaled = torch.stack(w_list_scaled, dim=-1)
        joblib.dump(weather_scalers, Path(self.root) / 'weather_scaler.pkl')

        # Anti-Leakage Scaling (Energy)
        norm_mode = cfg.earne_data.get('norm_mode', 'minmax')
        if norm_mode == 'minmax':
            train_v = np.concatenate([pivot_net.iloc[:train_end].values.flatten(), pivot_load.iloc[:train_end].values.flatten(), pivot_pv.iloc[:train_end].values.flatten()]).reshape(-1, 1)
            scaler = MinMaxScaler().fit(train_v)
            joblib.dump(scaler, Path(self.root) / 'scaler.pkl')

            net_scaled = torch.tensor(scaler.transform(pivot_net.fillna(0).values.reshape(-1, 1)).reshape(pivot_net.shape), dtype=torch.float)
            load_scaled = torch.tensor(scaler.transform(pivot_load.fillna(0).values.reshape(-1, 1)).reshape(pivot_load.shape), dtype=torch.float)
            pv_scaled = torch.tensor(scaler.transform(pivot_pv.fillna(0).values.reshape(-1, 1)).reshape(pivot_pv.shape), dtype=torch.float)
            
            # Dual-read signals (import/export)
            import_scaled = torch.tensor(scaler.transform(pivot_net.fillna(0).clip(lower=0).values.reshape(-1, 1)).reshape(pivot_net.shape), dtype=torch.float)
            export_scaled = torch.tensor(scaler.transform((-pivot_net).fillna(0).clip(lower=0).values.reshape(-1, 1)).reshape(pivot_net.shape), dtype=torch.float)
            
            norm_params = {'mode': 'minmax'}
        elif norm_mode == 'zero_log':
            # Combine all energy data for global stats (to maintain relative scales)
            train_v = np.concatenate([pivot_net.iloc[:train_end].values.flatten(), pivot_load.iloc[:train_end].values.flatten(), pivot_pv.iloc[:train_end].values.flatten()])
            nz_mean, nz_std = zero_preserved_log_stats(train_v)
            
            net_scaled = zero_preserved_log_normalize(pivot_net.fillna(0).values, nz_mean, nz_std)
            load_scaled = zero_preserved_log_normalize(pivot_load.fillna(0).values, nz_mean, nz_std)
            pv_scaled = zero_preserved_log_normalize(pivot_pv.fillna(0).values, nz_mean, nz_std)
            
            # Dual-read signals (import/export)
            import_scaled = zero_preserved_log_normalize(pivot_net.fillna(0).clip(lower=0).values, nz_mean, nz_std)
            export_scaled = zero_preserved_log_normalize((-pivot_net).fillna(0).clip(lower=0).values, nz_mean, nz_std)
            
            norm_params = {
                'mode': 'zero_log',
                'nz_mean': nz_mean,
                'nz_std': nz_std
            }
            torch.save(norm_params, Path(self.root) / 'norm_params.pt')
        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")

        torch.save({
            'net_scaled': net_scaled,
            'import_scaled': import_scaled,
            'export_scaled': export_scaled,
            'load_scaled': load_scaled,
            'pv_scaled': pv_scaled,
            'mask_raw': torch.tensor(pivot_net.notna().astype(float).values, dtype=torch.float),
            'weather_data': w_tensor_scaled,
            'weather_features': all_weather,
            'temporal_data': temp_t,
            'pos': torch.tensor(meta_df[['latitude', 'longitude']].fillna(0).values, dtype=torch.float),
            'macs': master_macs,
            'zips': meta_df['two_num_zip'].values.tolist(),
            'timestamps': pivot_net.index.tolist(),
            'norm_params': norm_params
        }, self.processed_paths[0])

@register_loader('earne_loader_new')
def load_earne_dataset(format, name, dataset_dir):
    dataset = EARNeGraphDataset(root=cfg.earne_data.processed_root, seq_len=cfg.model.seq_len)
    dataset.task = 'graph'
    train_idx, val_idx, test_idx = dataset.get_split_indices()
    dataset.data.train_graph_index = torch.tensor(train_idx, dtype=torch.long)
    dataset.data.val_graph_index = torch.tensor(val_idx, dtype=torch.long)
    dataset.data.test_graph_index = torch.tensor(test_idx, dtype=torch.long)
    return dataset
