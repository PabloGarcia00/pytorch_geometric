import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.graphgym.config import cfg

@register_node_encoder('earne_temporal')
class EARNeTemporalEncoder(nn.Module):
    """
    Temporal encoder for EARNe.
    Crushes [Batch * Nodes, seq_len, dim_in] -> [Batch * Nodes, hidden_channels]
    Using a 1D CNN to process the temporal history of net demand.
    """
    def __init__(self, emb_dim):
        super().__init__()
        
        # emb_dim in GraphGym is usually the 'hidden_channels' for the rest of the GNN
        self.seq_len = cfg.model.seq_len
        self.dim_in = cfg.model.dim_in
        self.kernel_size = cfg.conv1d_kernel_size

        assert self.seq_len >= self.kernel_size, (
            f"seq_len ({self.seq_len}) must be >= conv1d_kernel_size ({self.kernel_size})"
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(self.dim_in, 16, kernel_size=self.kernel_size, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * self.seq_len, emb_dim),
            nn.ReLU()
        )

    def forward(self, batch):
        # batch.x shape: [TotalNodes, Seq_Len, Dim_In]
        x = batch.x
        
        # Prepare for Conv1d: [TotalNodes, Dim_In, Seq_Len]
        x = x.permute(0, 2, 1) 
        
        # Encode
        batch.x = self.encoder(x) # -> [TotalNodes, emb_dim]
        
        return batch
