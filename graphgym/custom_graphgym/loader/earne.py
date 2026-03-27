import torch
import pandas as pd
from pathlib import Path
from itertools import permutations
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
import joblib

from torch_geometric.graphgym.register import register_loader
from torch_geometric.graphgym.config import cfg
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

class DatasetAdapter(ABC):
    COLS = ['timestamp', 'id', 'load', 'pv', 'net_demand']

    @abstractmethod
    def standardize(self) -> pd.DataFrame:
        """
        Force every child class to implement this.
        Must return columns: [timestamp, id, load, pv, net_demand]
        """
        pass

    def validate(self, df: pd.DataFrame):
        missing = [col for col in self.COLS if col not in df.columns]
        assert not missing, f"Adapter output is missing required columns: {missing}"
        return df

class EARNeDataAdapter(DatasetAdapter):
    def __init__(self, path, cadence_minutes=15, include_id_path=None) -> None:
        self.path = Path(path)
        self.cadence_minutes = cadence_minutes
        self.include_id_path = Path(include_id_path) if include_id_path else None
        self.label_decoder = None
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} not found!")

    def standardize(self) -> pd.DataFrame:
        all_files = [f for f in self.path.glob("*.parquet") if '_' not in f.stem]
        if not all_files:
            raise FileNotFoundError(f"No parquet files found in {self.path}")
            
        dfs = []
        for file in all_files:
            df = pd.read_parquet(file, columns=[
                "MessageTimestamp",
                "MAC",
                "pv",
                "load",
                "net_demand"
            ])

            df = df.rename(columns={
                "MAC": "id",
                "MessageTimestamp": "timestamp",
            })
            dfs.append(df)

        df = pd.concat(dfs)

        if self.include_id_path and self.include_id_path.exists():
            include_ids = pd.read_csv(self.include_id_path, header=None).squeeze().astype(str).to_list()
            df = df.loc[df["id"].astype(str).isin(include_ids)]
            assert not df.empty, f"Pre-filtering returned empty DataFrame! Check {self.include_id_path}"

        labels, uniques = pd.factorize(df["id"])
        df["id"] = labels

        self.label_decoder = {int(i): str(mac) for i, mac in enumerate(uniques)}

        df = (
            df.set_index("timestamp")
            .groupby("id")
            .resample(f"{self.cadence_minutes}min")
            .mean()
            .reset_index()
        )
        return self.validate(df)

class GraphDataset(InMemoryDataset):
    def __init__(self,
                 raw_path,
                 root,
                 seq_len=24,
                 train_split=0.8,
                 val_split=0.9,
                 start_date=None,
                 end_date=None,
                 include_id_path=None,
                 cadence_minutes=15,
                 ):
        self.raw_path = raw_path
        self.seq_len = seq_len
        self.train_split_ratio = train_split
        self.val_split_ratio = val_split
        self.id_to_include_path = include_id_path
        self.cadence_minutes = cadence_minutes
        self.start_date = start_date
        self.end_date = end_date

        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def raw_paths(self):
        return [str(Path(self.raw_path).absolute())]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def generate_complete_edge_index(self, num_nodes):
        edges = list(permutations(range(num_nodes), 2))
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def filter_date(self, df):
        start_date = self.start_date if self.start_date else df.index.min()
        end_date = self.end_date if self.end_date else df.index.max()

        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        filtered_df = df.loc[start_date:end_date]
        if filtered_df.empty:
            raise ValueError(f"Filtering resulted in an empty dataset. "
                             f"Check if {start_date} to {end_date} exist in data")
        return filtered_df

    def process(self):
        save_dir = Path(self.root)
        # Check if we are using dummy data based on the path
        if "dummy" in str(self.raw_path):
            # Expecting a single parquet file or similar for dummy
            raw_file = Path(self.raw_paths[0])
            if raw_file.is_dir():
                raw_file = next(raw_file.glob("*.parquet"))
            df = pd.read_parquet(raw_file)
            # Standardize dummy if needed, or assume it matches COLS
            if 'MessageTimestamp' in df.columns:
                df = df.rename(columns={"MessageTimestamp": "timestamp", "MAC": "id"})
        else:
            df = EARNeDataAdapter(path=self.raw_paths[0],
                                  cadence_minutes=self.cadence_minutes,
                                  include_id_path=self.id_to_include_path).standardize()

        cols = ['load', 'pv', 'net_demand']

        scaler = MinMaxScaler()
        scaler.fit(df[cols].to_numpy().reshape(-1, 1))
        df[cols] = scaler.transform(
            df[cols].to_numpy().reshape(-1, 1)
        ).reshape(-1, len(cols))

        save_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = save_dir / 'scaler.pkl'
        joblib.dump(scaler, scaler_path)

        load, pv, net_demand = (
            df.pivot(
                index="timestamp", columns="id", values=col
            ).sort_index()
            for col in cols
        )
        if self.start_date or self.end_date:
            load, pv, net_demand = (
                self.filter_date(d)
                for d in [load, pv, net_demand]
            )

        combined_mask = (
            load.notna() &
            pv.notna() &
            net_demand.notna()
        )

        combined_mask = torch.tensor(combined_mask.to_numpy().copy(), dtype=torch.float32)

        load, pv, net_demand = (
            torch.tensor(X.fillna(0).to_numpy().copy(), dtype=torch.float32)
            for X in [load, pv, net_demand]
        )

        total_len = net_demand.size(0) - self.seq_len
        train_end = int(total_len * self.train_split_ratio)
        val_end = int(total_len * self.val_split_ratio)
        num_nodes = load.shape[1]
        data = [
            Data(
                load=load,
                pv=pv,
                net_demand=net_demand,
                mask=combined_mask,
                train_split=train_end,
                val_split=val_end,
                edge_index=self.generate_complete_edge_index(num_nodes),
                num_nodes=num_nodes
            )
        ]
        self.save(data, self.processed_paths[0])

    def len(self):
        if self._data is None:
            return 0
        return self._data.net_demand.size(0) - self.seq_len - 1

    def get_split_indices(self):
        total_indices = list(range(self.len()))
        train_idx = total_indices[:self._data.train_split]
        val_idx = total_indices[self._data.train_split: self._data.val_split]
        test_idx = total_indices[self._data.val_split:]

        return train_idx, val_idx, test_idx

    def get(self, idx):
        current = idx + self.seq_len
        window_net_demand = self._data.net_demand[idx: current, :].t()
        target_load = self._data.load[current, :]
        target_pv = self._data.pv[current, :]
        target_net_demand = self._data.net_demand[current, :]
        mask = self._data.mask[current, :]
        edge_index = self._data.edge_index
        num_nodes = self._data.num_nodes

        return Data(
            x=window_net_demand.unsqueeze(-1),
            y_load=target_load,
            y_pv=target_pv,
            y_net_demand=target_net_demand,
            mask=mask,
            edge_index=edge_index,
            num_nodes=num_nodes
        )

@register_loader('earne_disagg')
def load_earne_disagg(format, name, dataset_dir):
    import warnings
    warnings.warn(
        "The 'earne_disagg' loader (loader/earne.py) is deprecated. "
        "Use 'earne_loader_new' (loader/graph_dataset.py) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if format != 'earne_disagg':
        return None
    dataset = GraphDataset(
        root=dataset_dir,
        raw_path=cfg.earne_data.raw_data_path,
        seq_len=cfg.model.seq_len,
        train_split=cfg.train.train_split,
        val_split=cfg.train.val_split,
        include_id_path=cfg.earne_data.include_id_path,
        cadence_minutes=cfg.earne_data.cadence_minutes
    )

    train_idx, val_idx, test_idx = dataset.get_split_indices()

    # GraphGym expects indices for graph tasks
    # We assign them to the internal data object
    dataset._data.train_graph_index = torch.tensor(train_idx, dtype=torch.long)
    dataset._data.val_graph_index = torch.tensor(val_idx, dtype=torch.long)
    dataset._data.test_graph_index = torch.tensor(test_idx, dtype=torch.long)

    return dataset
