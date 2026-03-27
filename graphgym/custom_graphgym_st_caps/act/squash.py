import torch

def squash(s: torch.Tensor) -> torch.Tensor:
    """
    Squashing function for capsules (Eq. 18 in Sabour et al.)
    
    Args:
        s: Input vector
        
    Returns:
        Squashed vector with norm < 1
    """
    norm = torch.norm(s, dim=-1, keepdim=True)
    norm_sq = norm ** 2
    return (norm_sq / (1 + norm_sq)) * (s / (norm + 1e-8))
