import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')

from torch_geometric.data import Data
from torch_geometric.graphgym.register import register_loader
from torch_geometric.graphgym.config import cfg

class BTMDataset(Dataset):
    def __init__(self, data, indices, m=35, normalize_per_site=True):
        self.data = data
        self.indices = indices
        self.m = m
        self.normalize_per_site = normalize_per_site
        if self.normalize_per_site:
            self.stats = {}
            for site_id in data['site_id'].unique():
                site_data = data[data['site_id'] == site_id]
                self.stats[site_id] = {
                    'net_mean': site_data['net_demand'].mean(),
                    'net_std': site_data['net_demand'].std() + 1e-6,
                    'load_mean': site_data['load'].mean(),
                    'load_std': site_data['load'].std() + 1e-6,
                    'pv_mean': site_data['pv'].mean(),
                    'pv_std': site_data['pv'].std() + 1e-6
                }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        site_id = self.data.iloc[actual_idx]['site_id']
        window_data = self.data.iloc[actual_idx - self.m : actual_idx + 1]
        net_demand = window_data['net_demand'].values
        load = self.data.iloc[actual_idx]['load']
        pv = self.data.iloc[actual_idx]['pv']
        if self.normalize_per_site:
            stats = self.stats[site_id]
            net_demand = (net_demand - stats['net_mean']) / stats['net_std']
            load = (load - stats['load_mean']) / stats['load_std']
            pv = (pv - stats['pv_mean']) / stats['pv_std']
            
        # GraphGym and PyG expect Data objects
        return Data(
            x=torch.FloatTensor(net_demand).unsqueeze(-1), # [SeqLen, 1]
            y_load=torch.FloatTensor([load]),
            y_pv=torch.FloatTensor([pv]),
            site_id=site_id,
            mask=torch.FloatTensor([1.0]) # Assuming valid if in dataset
        )

@register_loader('st_caps_loader')
def load_st_caps_dataset(format, name, dataset_dir):
    # Mapping GraphGym logic to our custom loader
    batch_size = cfg.train.batch_size
    m = cfg.st_caps.m
    
    # Assuming dataset_dir points to where CSVs are
    # or using cfg.earne_data.raw_data_path
    data_path = cfg.earne_data.raw_data_path
    
    # Reuse existing get_dataloader logic
    train_loader, val_loader, test_loader = get_dataloader(
        dataset="earne", # default for now
        data_path=data_path,
        batch_size=batch_size,
        m=m
    )
    
    # Return a list of loaders [train, val, test] as GraphGym expects
    # Note: GraphGym usually expects a Dataset object from register_loader
    # but some versions handle a list of Loaders.
    # To be safe, we can return the Dataset or a custom object.
    
    # For now, let's return the Dataset and let GraphGym handle the loaders
    # But GraphGym's create_loader uses the returned dataset.
    
    # Let's check how earne_loader_new does it: it returns a Dataset.
    return train_loader.dataset 


def load_pecan_street(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Pecan Street data not found at {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    required_cols = ['timestamp', 'site_id', 'net_demand', 'load', 'pv']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in {path}. Expected: {required_cols}")
    return df

def load_earne(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"EARN-E data not found at {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"]) 
    required_cols = ['timestamp', 'site_id', 'net_demand', 'load', 'pv']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in {path}")
    return df

def load_ausgrid(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Ausgrid data not found at {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    required_cols = ['timestamp', 'site_id', 'net_demand', 'load', 'pv']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in {path}")
    return df

def split_data(data: pd.DataFrame, dataset: str) -> Tuple[list, list, list]:
    if dataset == 'pecan':
        train_indices, val_indices, test_indices = [], [], []
        for month in data['timestamp'].dt.to_period('M').unique():
            month_data = data[data['timestamp'].dt.to_period('M') == month]
            month_indices = month_data.index.tolist()
            n = len(month_indices)
            n_train_val = int(n * 0.8)
            n_train = int(n_train_val * 0.65 / 0.8)
            train_indices.extend(month_indices[:n_train])
            val_indices.extend(month_indices[n_train:n_train_val])
            test_indices.extend(month_indices[n_train_val:])
    else:  
        data = data.reset_index(drop=True) 
        data['year'] = data['timestamp'].dt.year
        years = sorted(data['year'].unique())
        if len(years) >= 3:
            train_val_data = data[data['year'].isin(years[:2])]
            test_data = data[data['year'] == years[2]]
        else:
            n = len(data)
            train_val_data = data.iloc[:int(n * 0.67)]
            test_data = data.iloc[int(n * 0.67):]
        train_val_indices = train_val_data.index.tolist()
        test_indices = test_data.index.tolist()
        n_tv = len(train_val_indices)
        n_train = int(n_tv * 0.65 / 0.8)
        train_indices = train_val_indices[:n_train]
        val_indices = train_val_indices[n_train:]
    return train_indices, val_indices, test_indices

def collate_batch(batch):
    net_demand = torch.stack([item['net_demand'] for item in batch])
    load = torch.cat([item['load'] for item in batch])
    pv = torch.cat([item['pv'] for item in batch])
    site_ids = [item['site_id'] for item in batch]
    return {
        'net_demand': net_demand,  
        'load': load,              
        'pv': pv,                  
        'site_ids': site_ids       
    }

def get_dataloader(dataset: str, data_path: str, batch_size: int,
                  num_workers: int = 4, m: int = 35) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if dataset == 'pecan':
        data = load_pecan_street(Path(data_path) / 'pecan_street.csv')
    elif dataset == "earne":
        data = load_earne(Path(data_path) / 'earne_raw.csv')
    else:
        data = load_ausgrid(Path(data_path) / 'ausgrid.csv')
    data = data[m:].reset_index(drop=True)
    train_indices, val_indices, test_indices = split_data(data, dataset)
    train_indices = [i for i in train_indices if i >= m]
    val_indices = [i for i in val_indices if i >= m]
    test_indices = [i for i in test_indices if i >= m]
    train_dataset = BTMDataset(data, train_indices, m=m)
    val_dataset = BTMDataset(data, val_indices, m=m)
    test_dataset = BTMDataset(data, test_indices, m=m)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    return train_loader, val_loader, test_loader

def get_dataloader_with_preprocessing(dataset: str, data_path: str, config: Dict,
                                      batch_size: int, num_workers: int = 4,
                                      m: int = 35):
    from custom_graphgym.transform.preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor(config)
    artifacts = preprocessor.load_artifacts(dataset)
    if artifacts is not None:
        train_df, val_df, test_df = artifacts
        def get_valid_indices(df, m):
            mask = (df['site_id'] == df['site_id'].shift(m))
            return df[mask].index.tolist()
        train_dataset = BTMDataset(train_df, get_valid_indices(train_df, m), m=m, normalize_per_site=False)
        val_dataset = BTMDataset(val_df, get_valid_indices(val_df, m), m=m, normalize_per_site=False)
        test_dataset = BTMDataset(test_df, get_valid_indices(test_df, m), m=m, normalize_per_site=False)
    else:
        return get_dataloader(dataset, data_path, batch_size, num_workers, m)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    return train_loader, val_loader, test_loader
