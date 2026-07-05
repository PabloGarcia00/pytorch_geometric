import torch
import time
import numpy as np
from torch_geometric.graphgym.config import cfg
from custom_graphgym.loader.graph_builder import GraphBuilder
from custom_graphgym.loss.earne_loss import earne_loss_complex
from custom_graphgym.loader.graph_dataset import (
    zero_preserved_log_stats, 
    zero_preserved_log_normalize, 
    zero_preserved_log_denormalize
)

def benchmark_normalization():
    print("\n--- Benchmarking Normalization ---")
    # Mock cfg for zero_log
    cfg.earne_data.zero_id = 0.0
    
    # Generate 1M points (typical for large fleet processing)
    data = torch.exp(torch.randn(1000000) * 0.8 + 7.0) # Load-like data
    data[torch.rand(1000000) > 0.8] = 0.0 # 20% zeros
    
    # 1. Stats calculation
    t0 = time.time()
    mean, std = zero_preserved_log_stats(data.numpy())
    print(f"Stats calculation (1M points): {time.time()-t0:.4f}s")
    
    # 2. Normalization
    t0 = time.time()
    normed = zero_preserved_log_normalize(data, mean, std, shift=10.0)
    print(f"Normalization (1M points): {time.time()-t0:.4f}s")
    
    # 3. Denormalization
    t0 = time.time()
    denormed = zero_preserved_log_denormalize(normed, mean, std, shift=10.0)
    print(f"Denormalization (1M points): {time.time()-t0:.4f}s")
    
    # Correctness
    recon_err = torch.abs(data - denormed).max().item()
    print(f"Max Reconstruction Error: {recon_err:.6f} Watts")

def benchmark_graph_builder():
    print("\n--- Benchmarking GraphBuilder (Learned Correlation) ---")
    N_total = 1600 # 32 batch * 50 nodes
    T = 96
    batch_size = 32
    nodes_per_graph = 50
    
    net_demand = torch.randn(N_total, T).cuda()
    batch_vec = torch.arange(batch_size).repeat_interleave(nodes_per_graph).cuda()
    
    builder = GraphBuilder(lambda_threshold=0.25)
    
    # Warmup
    _, _ = builder.build_graph(net_demand, batch_vec=batch_vec)
    
    # Benchmark
    iters = 100
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        edge_index, _ = builder.build_graph(net_demand, batch_vec=batch_vec)
    torch.cuda.synchronize()
    
    avg_time = (time.time() - t0) / iters
    print(f"Average build_graph time ({N_total} nodes, batch={batch_size}): {avg_time*1000:.2f}ms")
    print(f"Edges generated: {edge_index.shape[1]}")

def benchmark_loss_function():
    print("\n--- Benchmarking Loss Function (earne_loss) ---")
    N = 1600
    n_quantiles = 3
    
    # Mock inputs
    pred = torch.randn(N, n_quantiles * 2).cuda()
    true = torch.randn(N, 4).cuda() # load, pv, mask, net_demand
    
    # Mock cfg
    cfg.model.loss_fun = 'earne_loss'
    cfg.model.quantiles = [0.1, 0.5, 0.9]
    cfg.train.physics_weight = 0.1
    cfg.train.pv_sentinel_weight = 0.1
    cfg.earne_data.zero_id = 0.0
    
    # Attach required tensors to batch (mocking earne_head behavior)
    import torch_geometric.graphgym.register as register
    class MockBatch:
        def __init__(self):
            self.q_load = torch.randn(N, 3).cuda()
            self.q_pv = torch.randn(N, 3).cuda()
            self.device = torch.device('cuda')
    
    register.batch = MockBatch()
    
    # Warmup
    _, _ = earne_loss_complex(pred, true)
    
    # Benchmark
    iters = 500
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        loss, _ = earne_loss_complex(pred, true)
    torch.cuda.synchronize()
    
    print(f"Average earne_loss time: {(time.time()-t0)/iters*1000:.4f}ms")

if __name__ == "__main__":
    benchmark_normalization()
    if torch.cuda.is_available():
        benchmark_graph_builder()
        benchmark_loss_function()
    else:
        print("\nCUDA not available, skipping GPU benchmarks.")
