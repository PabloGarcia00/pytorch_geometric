from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_scheduler


@register_scheduler("warmup_cos")
def warmup_cos_scheduler(optimizer, max_epoch):
    warmup_epochs = cfg.train.warmup.epochs
    warmup = LambdaLR(
        optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs
    )
    cosine = CosineAnnealingLR(optimizer, T_max=max_epoch - warmup_epochs)
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
