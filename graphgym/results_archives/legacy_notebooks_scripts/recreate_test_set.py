# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (GraphGym)
#     language: python
#     name: graphgym
# ---

import json
import os
# %%
import sys
from pathlib import Path

import pandas as pd
import torch

# Get the root directory (one level up from /notebooks)

sys.path.append("/home/llan/projects/messm/pytorch_geometric-single-read/graphgym/")
os.chdir("/home/llan/projects/messm/pytorch_geometric-single-read/graphgym")

import argparse

import custom_graphgym  # noqa
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
from custom_graphgym.loader.graph_dataset import (
    EARNeGraphDataset,
    load_earne_dataset,
)
from custom_graphgym.loader.normalization import (
    denorm_ihs,
    denorm_log1p,
    denorm_minmax,
    denorm_zscore,
)

from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train

# %load_ext autoreload
# %autoreload 2


# %%

def _load_norm_params():
    e = cfg.earne_data.energy_norm_mode
    w = cfg.earne_data.weather_norm_mode
    g = cfg.earne_data.graph_mode
    path = Path(cfg.dataset.dir) / f"norm_params_{e}_{w}_{g}.pt"
    raw = torch.load(path, weights_only=False)
    return raw


def init_graphgym_config(config_path, opts=None):
    """Initializes the global GraphGym configuration."""
    cfg.set_new_allowed(True)
    set_cfg(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    # Create a mock args object to satisfy load_cfg requirements
    args = argparse.Namespace()
    args.cfg_file = config_path
    args.accelerator = device
    args.opts = opts if opts else []
    
    load_cfg(cfg, args)
    return cfg

def find_stats(exp_path, split="val"):
    records = []
    for file in Path(exp_path).glob(f"**/{split}/stats.json"):
        with open(file) as f:
            for line in f:
                record = json.loads(line)
                record["exp"] = file.parts[-4]  # e.g. earne_exp3-weather_mode=False-mp=3
                record["split"] = split
                records.append(record)
    return pd.DataFrame(records)    

denorm_dict = {
   "ihs" : denorm_ihs,
    "log1p": denorm_log1p,
    "minmax": denorm_minmax,
    "zscore": denorm_zscore,
}


# %%

# %%
# run the simulation here
import numpy as np


def collect_results(model, test_loader, params, load_denorm, pv_denorm):
    """Run model inference and denormalise predictions."""
    net_demand, actual_load, actual_pv, pred_load, pred_pv, mask = [], [], [], [], [], []
    n_q = cfg.model.n_quantiles
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch.to("cuda")
            pred, _ = model(batch)
            
            # Store data
            net_demand.append(batch.y_net_demand.cpu().numpy())
            actual_load.append(batch.y_load.cpu().numpy())
            actual_pv.append(batch.y_pv.cpu().numpy())
            pred_load.append(pred[:, :n_q].cpu().numpy())
            pred_pv.append(pred[:, n_q : 2 * n_q].cpu().numpy())
            mask.append(batch.mask.cpu().numpy())

    # Aggregate results
    res = {
        "net_demand": np.concatenate(net_demand, axis=0),
        "actual_load": np.concatenate(actual_load, axis=0),
        "actual_pv": np.concatenate(actual_pv, axis=0),
        "pred_load": np.concatenate(pred_load, axis=0),
        "pred_pv": np.concatenate(pred_pv, axis=0),
    }
    mask = np.concatenate(mask, axis=0).astype(bool)

    res["net_demand"] = denorm_ihs(torch.tensor(res["net_demand"], dtype=torch.float), params["net_scaled"][0], params["net_scaled"][1]).numpy()
    res["actual_load"] = load_denorm(torch.tensor(res["actual_load"], dtype=torch.float), params["load_scaled"][0], params["load_scaled"][1]).numpy()
    res["pred_load"] = load_denorm(torch.tensor(res["pred_load"], dtype=torch.float), params["load_scaled"][0], params["load_scaled"][1]).numpy()
    res["actual_pv"] = pv_denorm(torch.tensor(res["actual_pv"], dtype=torch.float), params["pv_scaled"][0], params["pv_scaled"][1]).numpy()
    res["pred_pv"] = pv_denorm(torch.tensor(res["pred_pv"], dtype=torch.float), params["pv_scaled"][0], params["pv_scaled"][1]).numpy()
    return res, mask
    


# %%
def rmse(true_array, pred_array, mask_bool_array):
    t = np.asarray(true_array)[mask_bool_array]
    p = np.asarray(pred_array)[mask_bool_array]
    return np.sqrt(np.mean((p - t) ** 2))


def nrmse(true_array, pred_array, mask_bool_array):
    t = np.asarray(true_array)[mask_bool_array]
    p = np.asarray(pred_array)[mask_bool_array]
    mean_obs = np.mean(t)
    return np.sqrt(np.mean((p - t) ** 2)) / mean_obs

def mae(true_array, pred_array, mask_bool_array):
    t = np.asarray(true_array)[mask_bool_array]
    p = np.asarray(pred_array)[mask_bool_array]
    s = np.abs(p-t)
    return np.mean(s) 

def mape(true_array, pred_array, mask_bool_array):
    t = np.asarray(true_array)[mask_bool_array]
    p = np.asarray(pred_array)[mask_bool_array]
    # exclude near-zero actuals to avoid division instability
    valid = np.abs(t) > 1e-10
    return np.mean(np.abs((p[valid] - t[valid]) / t[valid])) * 100


def r_squared(true_array, pred_array, mask_bool_array):
    t = np.asarray(true_array)[mask_bool_array]
    p = np.asarray(pred_array)[mask_bool_array]
    return np.corrcoef(p, t)[0, 1] ** 2


# --- pandas apply example ---

def compute_metrics(row):
    """Row-wise wrapper for use with df.apply(compute_metrics, axis=1)."""
    mask = row["mask"]
    return pd.Series({
        "RMSE":    rmse(row["true"], row["pred"], mask),
        "nRMSE":   nrmse(row["true"], row["pred"], mask),
        "MAE":     mae(row["true"], row["pred"], mask),
        "MAPE":    mape(row["true"], row["pred"], mask),
        "R2":      r_squared(row["true"], row["pred"], mask),
    })


rng = np.random.default_rng(0)
n = 100

df = pd.DataFrame({
    "true": [rng.normal(10, 2, n) for _ in range(3)],
    "pred": [rng.normal(10, 2, n) for _ in range(3)],
    "mask": [rng.choice([True, False], n, p=[0.8, 0.2]) for _ in range(3)],
})

metrics = df.apply(compute_metrics, axis=1)
print(metrics.round(4))

# %%
# load the experiments here:
seed = 0

exp_path = Path("/home/llan/projects/messm/pytorch_geometric-single-read/graphgym/results/earne_exp3_grid_exp3")
exp_configs = list(exp_path.glob("**/config.yaml"))
# model_paths = list(exp_path.glob("**/*.ckpt"))
model_paths = [list(p.parent.joinpath(f"{str(seed)}/ckpt").glob("**/*.ckpt"))[-1] for p in exp_configs]
paired = list(zip(exp_configs, model_paths))

# load config
cfg = init_graphgym_config(exp_configs[-1])
cfg.accelerator = "cuda"

# load model
model = create_model()
ckpt = torch.load(model_paths[-1])
model.load_state_dict(ckpt["state_dict"])

# load normalization parameters
params =  _load_norm_params()
load_denorm_type, pv_denorm_type = cfg.earne_data.energy_norm_mode
load_denorm, pv_denorm = denorm_dict[load_denorm_type], denorm_dict[pv_denorm_type]

# load dataset
data = GraphGymDataModule.load_from_checkpoint(model_paths[-1])
test_loader = data.test_dataloader()
train_loader = data.train_dataloader() 

results, mask = collect_results(model, test_loader, params, load_denorm, pv_denorm)


# %%
val_stats = find_stats(exp_path, "val")
train_stats = find_stats(exp_path, "train")

fig, ax = plt.subplots(1, 2, figsize=(12, 3))
sns.lineplot(train_stats, x="epoch", y="loss", hue="exp", ax=ax[0], legend=True)
sns.lineplot(val_stats.groupby("exp").apply(lambda g: g.iloc[20:]).reset_index(), x="epoch", y="loss", hue="exp", ax=ax[1])
sns.move_legend(ax[1], loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()

# %% jupyter={"source_hidden": true}
val_stats10101010101010

# %%
stats.groupby("exp")["loss"].min().reset_index()


# %%
model_paths

# %%

# %%
model = create_model()
ckpt = torch.load(path)
model.load_state_dict(ckpt["state_dict"])

# %%
# checking weight distribution of models
fig, ax = plt.subplots(2, len(model_paths), figsize=(3 * len(model_paths), 6))
ep

for i, (conf, p) in enumerate(paired):
    # load model
    cfg = init_graphgym_config(conf)
    cfg.accelerator = "cuda"
    model = create_model()
    ckpt = torch.load(p)
    model.load_state_dict(ckpt["state_dict"])

    try:
        weather_weight = np.concat([val.flatten().cpu() for key, val in model.state_dict().items() if "weather" in key and "weight" in key])
        sns.histplot(weather_weight[weather_weight > eps], ax=ax[1, i])
        print(len(weather_weight < eps))
    except:
        print("no weather encoder used")
    net_weight = np.concat([val.flatten().cpu() for key, val in model.state_dict().items() if "net_encoder" in key and "weight" in key])
    sns.histplot(net_weight[net_weight > eps], ax=ax[0, i], bins=200)
    print(len(np.where(net_weight < eps)))
    
    

plt.tight_layout()

# %%
# checking weight distribution of models
eps = 1e-8

for i, (conf, p) in enumerate(paired):
    # load model
    cfg = init_graphgym_config(conf)
    cfg.accelerator = "cuda"
    model = create_model()
    ckpt = torch.load(p)
    model.load_state_dict(ckpt["state_dict"])
    print("####", p)
    agg_dead = 0
    n_param = 0
    
    for key, val in model.state_dict().items():
        absolute = val.flatten().cpu().abs()
        n_dead_weight = (absolute < eps).sum().item()
        agg_dead += n_dead_weight
        total_weight = absolute.numel()
        n_param += total_weight
        print(f"--- {key}: {(n_dead_weight/total_weight) * 100} [{n_dead_weight}]")
        print()
    print(f"TOTAL: {agg_dead/n_param * 100} [{agg_dead}]")
    print()
    
        
        


# %%
np.where(weather_weight < 10e-7)

# %%
np.abs(weather_weight).min()
np.abs(net_weight).min()

# %%

# %%
weather_weight = np.concat([val.flatten().cpu() for key, val in model.state_dict().items() if "weather" in key and "weight" in key])
net_weight = np.concat([val.flatten().cpu() for key, val in model.state_dict().items() if "net_encoder" in key and "weight" in key])

# %%
sns.kdeplot(weather_weight, clip=(-0.2, 0.2))

# %%
sns.kdeplot(net_weight, clip=(-0.2, 0.2))

# %%
iradiance, cloud cover, temp

# %%
results

# %%
# get metrics for grid experiment
metrics = []
for i in range(len(exp_configs)):
    exp_name = exp_configs[i].parts[-2]
    print(exp_name)
    # load config
    cfg = init_graphgym_config(exp_configs[i])
    cfg.accelerator = "cuda"
    
    # load model
    model = create_model()
    ckpt = torch.load(model_paths[i])
    model.load_state_dict(ckpt["state_dict"])Dag Edwin,

    
    # load normalization parameters
    params =  _load_norm_params()
    load_denorm_type, pv_denorm_type = cfg.earne_data.energy_norm_mode
    load_denorm, pv_denorm = denorm_dict[load_denorm_type], denorm_dict[pv_denorm_type]

    # load dataset
    data = GraphGymDataModule.load_from_checkpoint(model_paths[i])
    test_loader = data.test_dataloader()
    train_loader = data.train_dataloader() 

    # collect results 
    results, mask = collect_results(model, test_loader, params, load_denorm, pv_denorm)
    
    df = pd.DataFrame({
        "true": [results["actual_pv"], results["actual_load"]],
        "pred": [results["pred_pv"][:, 1], results["pred_load"][:, 1]],
        "mask": [mask, mask],
    })
    m = df.apply(compute_metrics, axis=1)
    m.index = ["pv", "load"]
    m["model"] = exp_name
    metrics.append(m)
    print(m)


# %%

# %%
exp_configs[0].parts[-2]

# %%
grid_conc = pd.concat(metrics).reset_index(names="load_type")
grid_conc = grid_conc.set_index(["model", "load_type"])

# %%
grid_conc.sort_values(["R2"]).to_clipboard()

# %%
earne_exp2-weather_mode=True-mp=2-graph_mode=full_graph-norm_mode=log1p


# %%
test_loader.dataset[1]

# %%
# run the simulation here
import numpy as np


def collect_results(model, test_loader, denorm_params):
    """Run model inference and denormalise predictions."""
    net_demand, actual_load, actual_pv, pred_load, pred_pv, mask = [], [], [], [], [], []
    n_q = cfg.model.n_quantiles
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch.to("cuda")
            pred, _ = model(batch)
            
            # Store data
            net_demand.append(batch.y_net_demand.cpu().numpy())
            actual_load.append(batch.y_load.cpu().numpy())
            actual_pv.append(batch.y_pv.cpu().numpy())
            pred_load.append(pred[:, :n_q].cpu().numpy())
            pred_pv.append(pred[:, n_q : 2 * n_q].cpu().numpy())
            mask.append(batch.mask.cpu().numpy())

    # Aggregate results
    res = {
        "net_demand": np.concatenate(net_demand, axis=0),
        "actual_load": np.concatenate(actual_load, axis=0),
        "actual_pv": np.concatenate(actual_pv, axis=0),
        "pred_load": np.concatenate(pred_load, axis=0),
        "pred_pv": np.concatenate(pred_pv, axis=0),
    }
    mask = np.concatenate(mask, axis=0).astype(bool)

    res["net_demand"] = denorm_fn(torch.tensor(res["net_demand"], dtype=torch.float), params["net_scaled"][0], params["net_scaled"][1]).numpy()
    res["actual_load"] = denorm_fn(torch.tensor(res["actual_load"], dtype=torch.float), params["load_scaled"][0], params["load_scaled"][1]).numpy()
    res["pred_load"] = denorm_fn(torch.tensor(res["pred_load"], dtype=torch.float), params["load_scaled"][0], params["load_scaled"][1]).numpy()
    res["actual_pv"] = denorm_fn(torch.tensor(res["actual_pv"], dtype=torch.float), params["pv_scaled"][0], params["pv_scaled"][1]).numpy()
    res["pred_pv"] = denorm_fn(torch.tensor(res["pred_pv"], dtype=torch.float), params["pv_scaled"][0], params["pv_scaled"][1]).numpy()
    return res, mask
    
results, mask = collect_results(model, test_loader, (p1, p2, denorm_fn))


# %%
np.unique(mask, return_counts=True)

import matplotlib.pyplot as plt
# %%
import numpy as np
import seaborn as sns


def plot_regression_fit(true, pred):
    """Scatterplot with a 2nd order polynomial fit."""
    plt.scatter(true, pred, alpha=0.5, label='Data')
    
    # 2nd order polynomial fit
    z = np.polyfit(true, pred, 2)
    p = np.poly1d(z)
    x_range = np.linspace(true.min(), true.max(), 100)
    
    plt.plot(x_range, p(x_range), "r--", label='2nd Order Fit')
    plt.xlabel('Actual (W)'); plt.ylabel('Predicted (W)')
    plt.legend()
    plt.show()


plot_regression_fit(results["actual_load"][mask][:100_000], results["pred_load"][mask, 1][:100_000])


# %%
def plot_residuals(true, pred):
    """Residual plot with 2nd order fit and density distribution."""
    residuals = pred - true
    
    # Jointplot handles the scatter and the marginal density automatically
    g = sns.jointplot(x=true, y=residuals, kind='scatter', alpha=0.5)
    
    # Add 2nd order fit to the joint ax
    z = np.polyfit(true, residuals, 2)
    p = np.poly1d(z)
    x_range = np.linspace(true.min(), true.max(), 100)
    g.ax_joint.plot(x_range, p(x_range), "r--")
    
    # Zero line
    g.ax_joint.axhline(0, color='black', lw=2)
    
    g.set_axis_labels('Actual (W)', 'Residuals (W)')
    plt.show()


# %%
plot_residuals(results["actual_pv"][mask][:100_000], results["pred_pv"][mask][:100_000, 1])


# %%
def rmse(true_array, pred_array, mask_bool_array):
    t = np.asarray(true_array)[mask_bool_array]
    p = np.asarray(pred_array)[mask_bool_array]
    return np.sqrt(np.mean((p - t) ** 2))


def nrmse(true_array, pred_array, mask_bool_array):
    t = np.asarray(true_array)[mask_bool_array]
    p = np.asarray(pred_array)[mask_bool_array]
    mean_obs = np.mean(t)
    return np.sqrt(np.mean((p - t) ** 2)) / mean_obs

def mae(true_array, pred_array, mask_bool_array):
    t = np.asarray(true_array)[mask_bool_array]
    p = np.asarray(pred_array)[mask_bool_array]
    s = np.abs(p-t)
    return np.mean(s) 

def mape(true_array, pred_array, mask_bool_array):
    t = np.asarray(true_array)[mask_bool_array]
    p = np.asarray(pred_array)[mask_bool_array]
    # exclude near-zero actuals to avoid division instability
    valid = np.abs(t) > 1e-10
    return np.mean(np.abs((p[valid] - t[valid]) / t[valid])) * 100


def r_squared(true_array, pred_array, mask_bool_array):
    t = np.asarray(true_array)[mask_bool_array]
    p = np.asarray(pred_array)[mask_bool_array]
    return np.corrcoef(p, t)[0, 1] ** 2


# --- pandas apply example ---

def compute_metrics(row):
    """Row-wise wrapper for use with df.apply(compute_metrics, axis=1)."""
    mask = row["mask"]
    return pd.Series({
        "RMSE":    rmse(row["true"], row["pred"], mask),
        "nRMSE":   nrmse(row["true"], row["pred"], mask),
        "MAE":     mae(row["true"], row["pred"], mask),
        "MAPE":    mape(row["true"], row["pred"], mask),
        "R2":      r_squared(row["true"], row["pred"], mask),
    })


rng = np.random.default_rng(0)
n = 100

df = pd.DataFrame({
    "true": [rng.normal(10, 2, n) for _ in range(3)],
    "pred": [rng.normal(10, 2, n) for _ in range(3)],
    "mask": [rng.choice([True, False], n, p=[0.8, 0.2]) for _ in range(3)],
})

metrics = df.apply(compute_metrics, axis=1)
print(metrics.round(4))

# %%
results["pred_pv"][:, 1]

# %%
results["actual_pv"]

# %%
df = pd.DataFrame({
    "true": [results["actual_pv"], results["actual_load"]],
    "pred": [results["pred_pv"][:, 1], results["pred_load"][:, 1]],
    "mask": [mask, mask],
})

# %%

# %%
df

# %%
metrics = df.apply(compute_metrics, axis=1)

# %%
metrics

# %%
pd.DataFrame(results["net_demand"]).describe()

# %%
train_loader.dataset.data is test_loader.dataset.data


# %%
# Assuming your dataset returns (x, y) or a batch object
def retrieve_data(loader):
    all_x = torch.stack([loader.dataset[i].x for i in range(len(loader.dataset))])
    return all_x


# %%
train_data = retrieve_data(train_loader)

# %%
train_data = denorm_fn(train_data[..., -1, :].flatten(), p1, p2)

# %%
pd.DataFrame(train_data.numpy()).describe()

# %%
test_data = retrieve_data(test_loader)

# %%
test_data = denorm_fn(test_data[..., -1, :].flatten(), p1, p2)

# %%
pd.DataFrame(test_data.numpy()).describe()

# %%
