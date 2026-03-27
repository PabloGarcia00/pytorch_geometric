import torch
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, List
from tqdm import tqdm
from custom_graphgym_st_caps.network.model import STSGCCaps
from custom_graphgym_st_caps.optimizer.optimizer import get_optimizer
from custom_graphgym_st_caps.metric.metrics import calculate_metrics

class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = config['device']
        self.model = STSGCCaps(config).to(self.device)
        self.optimizer_encoder_decoder, self.optimizer_capsule = get_optimizer(self.model, config)
        self.history = {
            'train_loss':    [],
            'val_loss':      [],
            'train_metrics': [],
            'val_metrics':   []
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_step(self, batch: Dict) -> Dict:
        net_demand = batch['net_demand'].to(self.device)
        load_true = batch['load'].to(self.device)
        pv_true = batch['pv'].to(self.device)

        self.model.train()
        self.optimizer_encoder_decoder.zero_grad()
        outputs = self.model(net_demand, load_true, pv_true)
        if 'edge_pred' in outputs:
            recon_loss, _ = self.model.loss_fn.recon_loss(
                outputs['edge_pred'], outputs['edge_true'],
                outputs['node_pred'], outputs['node_true']
            )
            recon_loss.backward(retain_graph=True)
            self.optimizer_encoder_decoder.step()

        self.optimizer_capsule.zero_grad()
        outputs = self.model(net_demand, load_true, pv_true)
        est_loss, est_metrics = self.model.loss_fn.disagg_loss(
            outputs['load_pred'], outputs['load_true'],
            outputs['pv_pred'], outputs['pv_true']
        )
        est_loss.backward()
        self.optimizer_capsule.step()

        total_loss, metrics = self.model.loss_fn(outputs)
        return metrics

    def validate(self, val_loader) -> Dict:
        self.model.eval()
        all_load_pred, all_load_true, all_pv_pred, all_pv_true = [], [], [], []
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                net_demand = batch['net_demand'].to(self.device)
                load_true = batch['load'].to(self.device)
                pv_true = batch['pv'].to(self.device)
                outputs = self.model(net_demand, load_true, pv_true)
                loss, _ = self.model.loss_fn(outputs)
                total_loss += loss.item()
                all_load_pred.append(outputs['load_pred'])
                all_load_true.append(outputs['load_true'])
                all_pv_pred.append(outputs['pv_pred'])
                all_pv_true.append(outputs['pv_true'])

        all_load_pred = torch.cat(all_load_pred)
        all_load_true = torch.cat(all_load_true)
        all_pv_pred = torch.cat(all_pv_pred)
        all_pv_true = torch.cat(all_pv_true)

        load_metrics = calculate_metrics(all_load_pred, all_load_true)
        pv_metrics = calculate_metrics(all_pv_pred, all_pv_true)

        return {
            'val_loss':  total_loss / len(val_loader),
            'load_rmse': load_metrics['rmse'],
            'load_mae':  load_metrics['mae'],
            'load_mape': load_metrics['mape'],
            'pv_rmse':   pv_metrics['rmse'],
            'pv_mae':    pv_metrics['mae'],
            'pv_mape':   pv_metrics['mape']
        }

    def train(self, train_loader, val_loader, test_loader):
        print("\nStarting training...")
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_metrics = []
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config["epochs"]}')
            for batch in pbar:
                metrics = self.train_step(batch)
                train_metrics.append(metrics)
                pbar.set_postfix({'loss': f"{metrics['total_loss']:.4f}"})

            val_metrics = self.validate(val_loader)
            avg_train_loss = np.mean([m['total_loss'] for m in train_metrics])
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_metrics'].append(val_metrics)

            if val_metrics['val_loss'] < self.best_val_loss - self.config['early_stopping_epsilon']:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                self.save_checkpoint(f'best_model.pth')
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.load_checkpoint('best_model.pth')
        test_metrics = self.validate(test_loader)
        self.history['test_metrics'] = test_metrics

    def save_checkpoint(self, filename: str):
        checkpoint = {
            'model_state_dict':          self.model.state_dict(),
            'optimizer_encoder_decoder': self.optimizer_encoder_decoder.state_dict(),
            'optimizer_capsule':         self.optimizer_capsule.state_dict(),
            'history':                   self.history,
            'config':                    self.config
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str):
        if Path(filename).exists():
            checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_encoder_decoder.load_state_dict(checkpoint['optimizer_encoder_decoder'])
            self.optimizer_capsule.load_state_dict(checkpoint['optimizer_capsule'])
            self.history = checkpoint['history']

    def save_results(self, filename: str):
        results = {'config': self.config, 'history': self.history, 'best_val_loss': float(self.best_val_loss)}
        with open(filename, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
