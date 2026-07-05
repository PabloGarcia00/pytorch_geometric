import warnings
from typing import Any, Dict, Optional

import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.graphgym import create_loader, register
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import pl
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.register import register_train


class EARNeLoggerCallback(LoggerCallback):
    """Extends LoggerCallback to bridge all stats into Lightning's metric system."""

    def on_validation_epoch_end(self, trainer, pl_module):
        # Capture full stats before write_epoch resets the logger
        self._log_to_lightning(trainer, pl_module, self.val_logger, "val")
        super().on_validation_epoch_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_to_lightning(trainer, pl_module, self.train_logger, "train")
        super().on_train_epoch_end(trainer, pl_module)

    def _log_to_lightning(self, trainer, pl_module, logger, split):
        if logger._size_current == 0:
            return

        # Reconstruct full stats dict exactly as write_epoch does
        basic_stats = logger.basic()
        custom_stats = logger.custom()

        # Task metrics — needs _true and _pred before reset
        task_stats = {}
        for custom_metric in cfg.custom_metrics:
            func = register.metric_dict.get(custom_metric)
            if func:
                task_stats[custom_metric] = func(
                    logger._true, logger._pred, logger.task_type
                )

        if not task_stats:
            if logger.task_type == "regression":
                task_stats = logger.regression()

        stats = {**basic_stats, **task_stats, **custom_stats}

        for key, value in stats.items():
            if isinstance(value, (int, float)):
                pl_module.log(
                    f"{split}_{key}",
                    float(value),
                    on_epoch=True,
                    prog_bar=(key == "loss"),
                    sync_dist=False,
                )


@register_train("earne_train")
def train(
    model: GraphGymModule,
    datamodule: GraphGymModule,
    logger: bool = True,
    trainer_config: Optional[Dict[str, Any]] = None,
):
    warnings.filterwarnings("ignore", ".*use `CSVLogger` as the default.*")

    callbacks = []
    if logger:
        callbacks.append(EARNeLoggerCallback())
    if cfg.train.enable_ckpt:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=get_ckpt_dir(),
                monitor=cfg.train.early_stopping.monitor,
                mode=cfg.train.early_stopping.mode,
                save_top_k=1,  # keep only the best checkpoint
                verbose=True,
            )
        )
    if cfg.train.early_stopping.enable:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=cfg.train.early_stopping.monitor,
                patience=cfg.train.early_stopping.patience,
                mode=cfg.train.early_stopping.mode,
                verbose=True,
            )
        )

    # csv logger and wandb logger
    wrun = wandb.init(
        project=cfg.train.wandb.project,
        name=cfg.train.wandb.run_name or None,
        reinit="create_new",
    )

    wrun.config.update(dict(cfg), allow_val_change=True)
    wrun.summary["model_architecture"] = str(model)

    pl_logger = [
        CSVLogger(save_dir=cfg.out_dir),
        WandbLogger(
            save_dir=cfg.out_dir,
            log_model=False,
        ),
    ]

    trainer_config = trainer_config or {}
    limit = cfg.train.get("limit_batches", 0)
    if limit > 0:
        trainer_config.setdefault("limit_train_batches", limit)
        trainer_config.setdefault("limit_val_batches", limit)
        trainer_config.setdefault("limit_test_batches", limit)
    trainer = pl.Trainer(
        **trainer_config,
        enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        logger=pl_logger,
        default_root_dir=cfg.out_dir,
        max_epochs=cfg.optim.max_epoch,
        accelerator=cfg.accelerator,
        devices="auto" if not torch.cuda.is_available() else cfg.devices,
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    wrun.finish()
