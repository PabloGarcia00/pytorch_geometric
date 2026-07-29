import argparse
import os
import sys

sys.path.insert(0, "/home/llan/projects/messm/pytorch_geometric-single-read/graphgym")  # adjust as needed
import json
import pprint
from pathlib import Path

import custom_graphgym  # noqa — registers all custom modules
import pandas as pd
import torch
import tqdm

from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.graphgym.config import cfg, load_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule

# ── 1. Fix safe globals (required for torch.load) ────────────────────────────
torch.serialization.add_safe_globals(
    [DataEdgeAttr, DataTensorAttr, GlobalStorage]
)

# ── 2. Load grid experiment ────────────────────────────────────────────────────────────
GRID_DIR = "/home/llan/projects/messm/pytorch_geometric-single-read/graphgym/results/earne_exp3_grid_exp3"
OUT_DIR = "/home/llan/projects/messm/pytorch_geometric-single-read/graphgym/output/earne_exp3"


def traverse_grid_dir():
    data = {}
    for path in Path(GRID_DIR).iterdir():
        if path.name == "agg":
            continue
        data[path.name] = {
            "config": path / "config.yaml",
            "ckpt": list(path.rglob("*.ckpt")),
            "run_stats": list(path.rglob("*.csv")),
            "test": list(path.rglob("test/stats.json")),
        }
    return data


grid = traverse_grid_dir()

csvs = {key: val["run_stats"] for key, val in grid.items()}
dfs = []
for key, val in csvs.items():
    for file in val:
        seed = file.parts[-2].replace("version_", "")
        df = pd.read_csv(file)
        df = df.set_index("epoch")
        val = df[[c for c in df.columns if "val" in c]].dropna(
            how="any", axis=0
        )
        train = df[[c for c in df.columns if "train" in c]].dropna(
            how="any", axis=0
        )
        df = pd.concat([train, val], axis=1)
        df["seed"] = seed
        df["model"] = key
        dfs.append(df)

print(len(dfs))
dfs = pd.concat(dfs, axis=0)
print(dfs.head())
print(dfs.columns)
mae = [c for c in dfs.columns if "mae" in c]
loss = [c for c in dfs.columns if "loss" in c]

best_seeds = dfs.groupby("model")["val_loss"].idxmin()
filtered_dfs = []
for name, frame in dfs.groupby("model"):
    filtered_dfs.append(
        frame[frame["seed"] == frame.iloc[best_seeds[name]]["seed"]]
    )


filtered_dfs = pd.concat(filtered_dfs)
print(filtered_dfs)
print(mae, loss)

output_data = {}
for model in filtered_dfs["model"].unique():
    model_df = filtered_dfs[filtered_dfs["model"] == model].sort_index()
    model_df = model_df.reset_index()
    output_data[model] = model_df[["epoch"] + loss + mae].to_dict(orient="list")
print(output_data)


with open("exp3_mean_only.json", "w") as f:
    json.dump(output_data, f, indent=4)


# best ckpt per experiment — already computed above
best_seed_map = {}
for name, frame in dfs.groupby("model"):
    if name in best_seeds:
        best_seed_map[name] = str(frame.iloc[best_seeds[name]]["seed"])


def recreate_test_dataset():
    # ── Recreate dataset ────────────────────────────────────────
    for EXP in Path(GRID_DIR).iterdir():
        if not EXP.is_dir():
            continue
        if EXP.name == "agg":
            continue
        CONFIG = Path(f"/home/llan/projects/messm/pytorch_geometric-single-read/graphgym/configs/earne_exp3_grid_exp3/{EXP.name}.yaml")
        if not CONFIG.exists():
            CONFIG = Path(f"/home/llan/projects/messm/pytorch_geometric-single-read/graphgym/configs/earne_exp3_grid_exp3/{EXP.name}.yaml_done")
        cfg.set_new_allowed(True)
        load_cfg(
            cfg,
            argparse.Namespace(
                cfg_file=CONFIG,
                opts=[],
            ),
        )

        cfg.accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(cfg.accelerator)

        # ── 3. Recreate dataset + test loader ────────────────────────────────────────
        datamodule = GraphGymDataModule()
        test_loader = datamodule.test_dataloader()
        dataset = test_loader.dataset

        # ── 4. Load checkpoint ────────────────────────────────────────────────────────

        CKPT_PATHS = list(EXP.rglob("**/*.ckpt"))
        CKPT_PATHS = [(p, p.parts[-3]) for p in CKPT_PATHS]
        print(f"Found {len(CKPT_PATHS)} replicates", end="\n")

        # filter the best seed
        best_seed = best_seed_map.get(EXP.name, CKPT_PATHS[0][1] if CKPT_PATHS else "0")
        CKPT_PATHS = [(p, s) for p, s in CKPT_PATHS if s == best_seed]
        print(f"{EXP.name} — using seed {best_seed}")
        print()

        all_true = []
        all_pred = []

        for ckpt_path, seed in CKPT_PATHS:
            model = create_model()
            checkpoint = torch.load(
                ckpt_path, map_location=device, weights_only=False
            )
            state_dict = checkpoint.get(
                "state_dict", checkpoint.get("model_state_dict", checkpoint)
            )
            # Reconcile 'model.' prefix if needed
            model_keys = list(model.state_dict().keys())
            ckpt_keys = list(state_dict.keys())
            if model_keys[0].startswith("model.") and not ckpt_keys[
                0
            ].startswith("model."):
                state_dict = {f"model.{k}": v for k, v in state_dict.items()}
            elif ckpt_keys[0].startswith("model.") and not model_keys[
                0
            ].startswith("model."):
                state_dict = {
                    k.replace("model.", "", 1): v for k, v in state_dict.items()
                }

            model.load_state_dict(state_dict)
            pprint.pprint(model)
            model = model.to(device).eval()

            true_array = []
            pred_array = []

            # ── 5. Run inference on test set ──────────────────────────────────────────────
            with torch.no_grad():
                for batch in tqdm.tqdm(test_loader):
                    batch = batch.to(device)
                    pred, true = model(
                        batch
                    )  # returns (pred, true) per GraphGymModule.forward()
                    # pred: [N, n_quantiles * 2]  (load + PV quantiles)
                    # true: [N, 4]               (y_load, y_pv, mask, y_net_demand)

                    t_per_batch = pred.shape[0] // dataset.num_nodes
                    pred_array.append(
                        pred.reshape(
                            dataset.num_nodes,
                            t_per_batch,
                            cfg.model.n_quantiles * 2,
                        )
                        .permute(0, 2, 1)  # → [num_nodes, 6, t_per_batch]
                        .cpu()
                    )
                    true_array.append(
                        true.reshape(dataset.num_nodes, t_per_batch, 4)
                        .permute(0, 2, 1)  # → [num_nodes, 4, t_per_batch]
                        .cpu()
                    )

            all_true.append(torch.cat(true_array, dim=2))
            all_pred.append(torch.cat(pred_array, dim=2))
            # pred: [N, n_quantiles * 2, timesteps]  (load + PV quantiles)
            # true: [N, 4, timesteps]               (y_load, y_pv, mask, y_net_demand)
        true_name = OUT_DIR + "/" + CKPT_PATHS[0][0].parts[-4] + "_true" + ".pt"
        torch.save(torch.cat(all_true, dim=2), true_name)
        pred_name = OUT_DIR + "/" + CKPT_PATHS[0][0].parts[-4] + "_pred" + ".pt"
        torch.save(torch.cat(all_pred, dim=2), pred_name)


recreate_test_dataset()
