import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg, set_cfg
from custom_graphgym.config.earne import set_cfg_earne
from custom_graphgym.loader.graph_dataset import EARNeGraphDataset

"""
EARNe DATA LOADER TRAINING SCRIPT
This script explains the "Mental Model" of how GraphGym handles our spatial-temporal data.
Run these blocks in your REPL to see the transformation of data.
"""

# ==========================================
# BLOCK 1: THE CONFIGURATION ENGINE
# ==========================================
# Why? GraphGym uses a global 'cfg' object. Our custom loader reads from this 
# every time it is initialized. Changing 'cfg' changes the "Lens" of the data.
set_cfg(cfg)
set_cfg_earne(cfg)

# Think: If you change this list, the 'get()' method below will automatically 
# slice the big Master Bundle to only include these columns.
cfg.earne_data.weather_features = ["solar_radiation_avg", "air_temperature"]
print(f"Current Weather Features: {cfg.earne_data.weather_features}")


# ==========================================
# BLOCK 2: UNDERSTANDING THE 'GET' INDEXING
# ==========================================
# In a standard image dataset, idx=0 returns the 1st image.
# In EARNe, idx=0 returns a "Window" of time ending at t=seq_len.

def explain_windowing(idx, seq_len=96):
    start = idx
    end = idx + seq_len
    target = idx + seq_len # The "Future" step we want to predict
    print(f"For idx={idx}:")
    print(f"  - History Window: t=[{start} to {end-1}]")
    print(f"  - Target (y)    : t={target}")

explain_windowing(idx=0)
# Think: Why do we subtract 1 in len()? 
# If T=100 and seq_len=96, idx=0 is our only valid sample. 
# idx=1 would require target t=97, which doesn't exist.


# ==========================================
# BLOCK 3: TENSOR BROADCASTING (The "Condition" Secret)
# ==========================================
# Our model receives two main inputs per node:
# 1. Historical Net Demand: [Nodes, Seq_Len, 1] -> (The 'x' attribute)
# 2. Conditions: [Nodes, Feats] -> (The 'weather' attribute)

# How do we combine Weather (per node) and Time (global)?
num_nodes = 3
num_weather_feats = 2
num_time_feats = 6 # (month_sin, month_cos, etc.)

# Mock data
w_step = torch.randn(num_nodes, num_weather_feats)
t_step = torch.randn(num_time_feats)

# The Trick: Broadcast the global time to every node
t_broadcast = t_step.repeat(num_nodes, 1) # Shape: [3, 6]
conditions = torch.cat([w_step, t_broadcast], dim=1) # Shape: [3, 8]

print(f"\nCondition Tensor Shape: {conditions.shape}")
print("Think: Node 0 and Node 1 now have different weather but IDENTICAL time features.")


# ==========================================
# BLOCK 4: TOPOLOGY DYNAMICS (The Spatial Isolated Test)
# ==========================================
# The 'edge_index' defines who talks to whom.
from itertools import permutations

def mock_topology(mode='spatial_knn', n=3, k=2):
    if mode == 'full_graph':
        # Every node connected to every other node
        edges = list(permutations(range(n), 2))
        return torch.tensor(edges).t()
    else:
        # Only k-nearest (here we just mock 1->2, 2->3)
        return torch.tensor([[0, 1], [1, 2]])

print("\nFull Graph Edges (3 nodes):")
print(mock_topology('full_graph'))


# ==========================================
# BLOCK 5: THE FINAL 'DATA' OBJECT
# ==========================================
# This is what actually enters the GNN. 
# It's basically a dictionary that PyG understands as a Graph.
sample_graph = Data(
    x = torch.randn(num_nodes, 96, 1),   # History
    edge_index = mock_topology(n=3),     # Connectivity
    weather = conditions,                # Static/Dynamic conditions
    y = torch.randn(num_nodes, 3)        # Target (Load, PV, Mask)
)

print("\n--- The PyG Data Object ---")
print(sample_graph)
# Think: If you set k=0 (Spatial Isolation), edge_index becomes empty.
# The GNN then behaves exactly like an MLP because no messages are passed.
