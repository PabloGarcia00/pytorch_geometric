import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from custom_graphgym.train.earne_train import build_trainer_and_wandb

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.register import register_train


@register_train("knn_train")
def train(
    model: GraphGymModule,
    datamodule: GraphGymModule,
    logger: bool = True,
    trainer_config: Optional[Dict[str, Any]] = None,
):
    """Train-loop entry point for baseline_knn (custom_graphgym/network/
    baseline_knn.py), a non-gradient per-node lookup model.

    "Fitting" is a one-shot, no-grad pass populating the per-node neighbor
    bank (fit_cache) -- there is no SGD loop. trainer.fit() is still called,
    but with max_epoch forced to 0: Lightning's configure_optimizers() (and
    GraphGym's stock LoggerCallback, which unconditionally reads
    trainer.lr_scheduler_configs[0] on every val/test batch) both assume an
    optimizer/scheduler were set up, which only happens as part of fit().
    Calling trainer.validate()/test() directly without ever calling fit()
    first leaves trainer.lr_scheduler_configs empty and crashes that
    logging path -- max_epoch=0 satisfies the plumbing while running zero
    real training steps (baseline_knn's one parameter exists solely to keep
    torch.optim.Adam([]) from raising on an empty parameter list; it's
    disconnected from the prediction path).

    Reuses the same Lightning Trainer/wandb/CSV logging construction as
    earne_train.train() (via build_trainer_and_wandb) so this run's
    stats.json/wandb structure is identical in shape to a gradient-trained
    run's.

    Selected via cfg.train.mode = "knn_train" in baseline_knn's config,
    dispatched through the register.train_dict lookup in main.py.
    """
    model.model.fit_cache(datamodule.train_dataloader())

    cfg.optim.max_epoch = 0
    trainer, wrun = build_trainer_and_wandb(model, logger, trainer_config)
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wrun.finish()

    # GraphGym's Logger unconditionally provisions a train/ stats directory
    # for every run (cfg.share.num_splits is always 3 -- see
    # torch_geometric/graphgym/loader.py), but max_epoch=0 means no training
    # epoch ever writes a stats.json into it. Left in place, that empty
    # directory makes upstream agg_runs() (called unconditionally at the end
    # of main.py) crash trying to read a train/stats.json that was never
    # written. Removing it here is a KNN-only, project-code fix that leaves
    # agg_runs() itself untouched.
    train_dir = Path(cfg.run_dir) / "train"
    if train_dir.is_dir() and not (train_dir / "stats.json").exists():
        shutil.rmtree(train_dir)
