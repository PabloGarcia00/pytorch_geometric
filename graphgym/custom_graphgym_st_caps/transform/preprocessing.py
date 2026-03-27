import numpy as np
import pandas as pd
import torch
import json
import hashlib
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
import pickle

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        self.cadence_minutes = self.preprocessing_config.get('cadence_minutes', 15)
        self.imputer = self.preprocessing_config.get('imputer', 'ffill_bfill')
        self.outlier_method = self.preprocessing_config.get('outlier_method', 'hampel')
        self.missing_threshold = self.preprocessing_config.get('missing_threshold', 0.3)
        self.scaler_type = self.preprocessing_config.get('scaler_type', 'robust')
        self.compute_calendar = self.preprocessing_config.get('compute_calendar_features', True)
        self.compute_weather = self.preprocessing_config.get('compute_weather_features', False)
        self.cache_dir = Path(self.preprocessing_config.get('cache_dir', 'custom_graphgym_st_caps/preprocessed'))
        self.mi_k_neighbors = self.preprocessing_config.get('mi_k_neighbors', 5)
        self.mi_epsilon = self.preprocessing_config.get('mi_epsilon', 1e-6)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {'dropped_points': 0, 'imputed_points': 0, 'outliers_fixed': 0, 'windows_kept': 0, 'windows_dropped': 0}
        self.scalers, self.imputer_params, self.outlier_params, self.graph_params = {}, {}, {}, None

    def get_config_hash(self) -> str:
        config_str = json.dumps(self.preprocessing_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def normalize_timestamps(self, df: pd.DataFrame, timezone: str = 'UTC') -> pd.DataFrame:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']): df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None: df['timestamp'] = df['timestamp'].dt.tz_localize(timezone, ambiguous='infer')
        else: df['timestamp'] = df['timestamp'].dt.tz_convert(timezone)
        df = df.sort_values(['site_id', 'timestamp'])
        df = df.drop_duplicates(subset=['site_id', 'timestamp'], keep='first')
        return df

    def resample_to_cadence(self, df: pd.DataFrame) -> pd.DataFrame:
        resampled_dfs = []
        for site_id in df['site_id'].unique():
            site_df = df[df['site_id'] == site_id].set_index('timestamp')
            resampled = site_df[['net_demand', 'load', 'pv']].resample(f'{self.cadence_minutes}T').mean()
            resampled['site_id'] = site_id
            resampled_dfs.append(resampled.reset_index())
        result = pd.concat(resampled_dfs, ignore_index=True)
        result['net_demand'] = result['load'] - result['pv']
        return result

    def mark_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df['missing_net'] = df['net_demand'].isna()
        df['missing_load'] = df['load'].isna()
        df['missing_pv'] = df['pv'].isna()
        return df

    def impute_missing(self, df: pd.DataFrame, method: str = None, fit_data: pd.DataFrame = None) -> pd.DataFrame:
        method = method or self.imputer
        df = df.copy()
        if method == 'ffill_bfill':
            limit = self.preprocessing_config.get('ffill_limit', 4)
            for col in ['net_demand', 'load', 'pv']:
                df[col] = df.groupby('site_id')[col].transform(lambda x: x.fillna(method='ffill', limit=limit).fillna(method='bfill', limit=limit))
        return df

    def detect_outliers(self, df: pd.DataFrame, method: str = None, fit_data: pd.DataFrame = None) -> pd.DataFrame:
        return df # Simplified for refactoring

    def apply_physical_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[df['load'] < 0, 'load'] = 0
        df.loc[df['pv'] < 0, 'pv'] = 0
        return df

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.compute_calendar: return df
        df = df.copy()
        dt = pd.to_datetime(df['timestamp'])
        df['hour'], df['day_of_week'], df['month'] = dt.dt.hour, dt.dt.dayofweek, dt.dt.month
        return df

    def add_weather_features(self, df: pd.DataFrame, weather_file: str = None) -> pd.DataFrame:
        return df

    def add_rolling_features(self, df: pd.DataFrame, windows: List[int] = [4, 12]) -> pd.DataFrame:
        df = df.copy()
        for window in windows:
            for col in ['net_demand', 'load', 'pv']:
                df[f'{col}_mean_{window}'] = df.groupby('site_id')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        return df

    def fit_scalers(self, df: pd.DataFrame) -> None:
        self.scalers = {}
        for site_id in df['site_id'].unique():
            site_data = df[df['site_id'] == site_id]
            scaler = RobustScaler() if self.scaler_type == 'robust' else StandardScaler()
            scaler.fit(site_data[['net_demand', 'load', 'pv']])
            self.scalers[site_id] = scaler

    def transform_scalers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for site_id in df['site_id'].unique():
            if site_id in self.scalers:
                site_mask = df['site_id'] == site_id
                df.loc[site_mask, ['net_demand', 'load', 'pv']] = self.scalers[site_id].transform(df.loc[site_mask, ['net_demand', 'load', 'pv']])
        return df

    def compute_mutual_information(self, df: pd.DataFrame) -> np.ndarray:
        sites = sorted(df['site_id'].unique())
        n_sites = len(sites)
        adjacency = np.zeros((n_sites, n_sites))
        self.graph_params = {'sites': sites, 'adjacency': adjacency}
        return adjacency

    def save_artifacts(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, dataset: str) -> None:
        config_hash = self.get_config_hash()
        train_df.to_parquet(self.cache_dir / f"{dataset}_train_{config_hash}.parquet")
        val_df.to_parquet(self.cache_dir / f"{dataset}_val_{config_hash}.parquet")
        test_df.to_parquet(self.cache_dir / f"{dataset}_test_{config_hash}.parquet")
        with open(self.cache_dir / f"{dataset}_scalers_{config_hash}.pkl", 'wb') as f: pickle.dump(self.scalers, f)

    def load_artifacts(self, dataset: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        config_hash = self.get_config_hash()
        train_path = self.cache_dir / f"{dataset}_train_{config_hash}.parquet"
        if train_path.exists():
            return pd.read_parquet(train_path), pd.read_parquet(self.cache_dir / f"{dataset}_val_{config_hash}.parquet"), pd.read_parquet(self.cache_dir / f"{dataset}_test_{config_hash}.parquet")
        return None

    def preprocess(self, df: pd.DataFrame, dataset: str = 'pecan') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.normalize_timestamps(df)
        df = self.resample_to_cadence(df)
        df = self.mark_missing(df)
        from custom_graphgym_st_caps.loader.loader import split_data
        train_indices, val_indices, test_indices = split_data(df, dataset)
        train_df, val_df, test_df = df.iloc[train_indices].copy(), df.iloc[val_indices].copy(), df.iloc[test_indices].copy()
        train_df = self.impute_missing(train_df)
        val_df = self.impute_missing(val_df)
        test_df = self.impute_missing(test_df)
        train_df = self.apply_physical_constraints(train_df)
        self.fit_scalers(train_df)
        train_df = self.transform_scalers(train_df)
        val_df = self.transform_scalers(val_df)
        test_df = self.transform_scalers(test_df)
        self.save_artifacts(train_df, val_df, test_df, dataset)
        return train_df, val_df, test_df

def preprocess_dataset(config: Dict, dataset: str, data_path: str) -> bool:
    preprocessor = DataPreprocessor(config)
    if preprocessor.load_artifacts(dataset) is not None: return True
    df = pd.read_csv(Path(data_path) / f"{dataset}_raw.csv", parse_dates=['timestamp'])
    preprocessor.preprocess(df, dataset)
    return True
