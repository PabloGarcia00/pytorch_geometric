import numpy as np

# Mocking the configuration for the test
class MockCfg:
    def __init__(self):
        self.earne_data = type('obj', (object,), {'zero_id': 0.0})

cfg = MockCfg()

def zero_preserved_log_stats(X):
    Y = np.copy(X)
    Y[Y <= 0] = np.nan
    Y_log = np.log(Y)
    nonzero_mean = np.nanmean(Y_log)
    nonzero_std  = np.nanstd(Y_log)
    if np.isnan(nonzero_mean): nonzero_mean = 0.0
    if nonzero_std == 0 or np.isnan(nonzero_std): nonzero_std = 1.0
    return nonzero_mean, nonzero_std

def zero_preserved_log_normalize(X, nonzero_mean, nonzero_std, shift=10.0):
    zero_id = cfg.earne_data.zero_id
    X = np.array(X, dtype=np.float32)
    Y = np.copy(X)
    is_special = (Y <= 0)
    Y[is_special] = 1.0
    Y_log = np.log(Y)
    Y_res = (Y_log - nonzero_mean) / nonzero_std + shift
    Y_res[is_special] = zero_id
    return Y_res

def zero_preserved_log_denormalize(Y, nonzero_mean, nonzero_std, shift=10.0):
    zero_id = cfg.earne_data.zero_id
    X = np.copy(Y)
    is_special = (X < zero_id + 0.1)
    X_log = (X - shift) * nonzero_std + nonzero_mean
    X_res = np.exp(X_log)
    X_res[is_special] = 0.0
    return X_res

def run_scaling_test():
    print("=== EARNe Scaling Test (Numpy-only) ===")
    
    # 1. Generate Synthetic Data (Load in Watts)
    np.random.seed(42)
    positives = np.random.lognormal(mean=7.0, sigma=0.8, size=800) # Mean ~1500W
    zeros = np.zeros(200)
    data = np.concatenate([positives, zeros])
    
    nz_mean, nz_std = zero_preserved_log_stats(data)
    print(f"Stats: nz_mean={nz_mean:.4f}, nz_std={nz_std:.4f}")
    
    # 2. Forward-Backward Consistency
    normed = zero_preserved_log_normalize(data, nz_mean, nz_std)
    denormed = zero_preserved_log_denormalize(normed, nz_mean, nz_std)
    
    recon_error = np.abs(data - denormed).max()
    print(f"Max Reconstruction Error: {recon_error:.6f} Watts")
    
    # 3. Sensitivity Analysis
    val = 2000.0
    v_norm = zero_preserved_log_normalize([val], nz_mean, nz_std)[0]
    print(f"\n2000W maps to: {v_norm:.4f}")
    
    for eps in [0.01, 0.1, 0.5, 1.0]:
        v_noisy = v_norm + eps
        v_denorm = zero_preserved_log_denormalize([v_noisy], nz_mean, nz_std)[0]
        print(f"  Error +{eps:.2f} in norm-space -> {v_denorm:.1f} Watts (Diff: {v_denorm - 2000:.1f}W)")

    # 4. The \"Gap\" (Death Zone) Analysis
    print(f"\nGap Analysis (Sentinel={cfg.earne_data.zero_id}, Data Range starts > 1.0)")
    gap_values = np.array([0.05, 0.11, 0.5, 1.0, 1.5, 2.0, 2.5])
    for g in gap_values:
        val_w = zero_preserved_log_denormalize([g], nz_mean, nz_std)[0]
        status = "SNAPPED" if g < cfg.earne_data.zero_id + 0.1 else "ACTIVE"
        print(f"  Norm-prediction {g:5.2f} ({status:7}) -> {val_w:12.4f} Watts")

    # 5. Gradient Analysis (Steepness)
    zero_target_norm = cfg.earne_data.zero_id
    small_pos_norm = zero_preserved_log_normalize([1.0], nz_mean, nz_std)[0]
    print(f"\nGradient Cliff:")
    print(f"  Target 0W (Norm: {zero_target_norm})")
    print(f"  Target 1W (Norm: {small_pos_norm:.4f})")
    print(f"  Distance in Norm-Space: {abs(small_pos_norm - zero_target_norm):.4f}")

if __name__ == "__main__":
    run_scaling_test()
