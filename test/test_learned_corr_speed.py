import torch
import time

def build_graph_mock(N_total, batch_size, threshold=0.25):
    # N_total = 1600 (32 * 50)
    # abs_corr [1600, 1600]
    abs_corr = torch.rand(N_total, N_total).cuda()
    
    # batch_vec [1600]
    nodes_per_graph = N_total // batch_size
    batch_vec = torch.arange(batch_size).repeat_interleave(nodes_per_graph).cuda()
    
    # Mask cross-graph (simulating current implementation)
    mask_batch = (batch_vec.unsqueeze(0) == batch_vec.unsqueeze(1))
    abs_corr = abs_corr * mask_batch
    
    # Differentiable thresholding
    k = 100.0
    mask = torch.sigmoid(k * (abs_corr - threshold))
    
    adjacency = torch.exp(abs_corr) * mask
    
    return adjacency

def test():
    N_total = 1600
    batch_size = 32
    
    print(f"Testing with N_total={N_total}, batch_size={batch_size}")
    
    # Start timing
    t0 = time.time()
    adj = build_graph_mock(N_total, batch_size)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Build graph took: {t1-t0:.4f}s")
    
    # Check nonzero edges
    t2 = time.time()
    edge_index = adj.nonzero()
    torch.cuda.synchronize()
    t3 = time.time()
    print(f"Nonzero() took: {t3-t2:.4f}s")
    print(f"Number of edges: {edge_index.shape[0]}")
    
    # Total possible edges in batch blocks: 32 * 50 * 50 = 80,000
    # Total entries in 1600x1600 = 2,560,000
    
    # If sigmoid makes everything non-zero:
    # Cross-graph entries: abs_corr was 0.
    # mask = sigmoid(100 * (0 - 0.25)) = sigmoid(-25) > 0.
    # So nonzero() will return ALL 2.56M entries!

if __name__ == "__main__":
    if torch.cuda.is_available():
        test()
    else:
        print("CUDA not available, skipping GPU test.")
