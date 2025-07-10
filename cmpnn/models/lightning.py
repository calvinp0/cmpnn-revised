import time
import os
from typing import Optional, Union, List
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from cmpnn.models.aggregators import SumAggregator, MeanAggregator, NormMeanAggregator
from cmpnn.models.cmpnn import CMPNNEncoder
from cmpnn.models.ffns import MLP, ScaledOutputLayer, HybridRegressionHead, PeriodicHead
from cmpnn.models.utils import initialize_weights, plot_train_val_scatter
from cmpnn.optimizer.noam import NoamLikeOptimizer
from cmpnn.loss.utils import get_metric_by_name
# from cmpnn.loss.masked_loss import MaskedMetricWrapper

# import dataset
from torch.utils.data import DataLoader, Dataset


class CMPNNLightningModule(pl.LightningModule):
    """
    A PyTorch Lightning module for the CMPNN model.
    """

    def __init__(self,
                 atom_fdim: int = 133,
                 bond_fdim: int = 14,
                 global_fdim: int = 0,
                 atom_messages: bool = True,
                 depth: int = 3,
                 output_size: int = 1,
                 hidden_dim: int = 128,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 bias: bool = False,
                 booster: str = 'sum',
                 comm_mode: str = 'add',
                 prediction_type: str = 'regression',
                 aggregator: str = 'mean',
                 use_batch_norm: bool = False,
                 ffn_config: dict = None,
                 optimizer_class: type = torch.optim.Adam,
                 optimizer_params: dict = None,
                 learning_rate: float = 1e-3,
                 metrics: dict = None,
                 plot_lr: bool = False,
                 dynamic_depth: str = None,
                 use_atom_residual: bool = False,
                 use_bond_residual: bool = False,
                 projection_config: dict = None,
                 ignore_value: float = -10.0,
                 target_normalizer = None,
                 cont_indices: list = None,
                 per_indices: list = None,):

        super().__init__()
        self.save_hyperparameters()
        self.output_size = output_size
        self.prediction_type = prediction_type
        self.plot_lr = plot_lr
        self.dynamic_depth = dynamic_depth
        self.use_atom_residual = use_atom_residual
        self.use_bond_residual = use_bond_residual
        if self.prediction_type == 'classification':
            self.sigmoid = nn.Sigmoid()

        if cont_indices is not None and per_indices is not None:
            self.cont_indices = cont_indices
            self.per_indices = per_indices
        elif target_normalizer is not None and hasattr(target_normalizer, 'kinds'):
            self.kinds = target_normalizer.kinds
            self.cont_indices = [i for i, k in enumerate(self.kinds) if k == 'continuous']

            # periodic indices are pairs [sin, cos] - assume target vector is already expanded
            base_index = len(self.cont_indices)
            per_pairs = sum(1 for k in self.kinds if k == 'periodic')
            self.per_indices = list(range(base_index, base_index + 2 * per_pairs))
        else:
            raise ValueError("Cannot determine target kinds. Please provide either `cont_indices` and `per_indices` or `target_normalizer.kinds`.")

        self.training_logs = []
        # Create the model
        self.model = CMPNNEncoder(
            atom_fdim=atom_fdim,
            bond_fdim=bond_fdim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            activation=activation,
            bias=bias,
            atom_messages=atom_messages,
            booster=booster,
            comm_mode=comm_mode,
            dynamic_depth=dynamic_depth
        )
        self.projection_config = projection_config if projection_config else {
            'hidden_dim': hidden_dim,
            'n_layers': 2,
            'dropout': dropout,
            'activation': activation
        }

        # Projection Layers
        if self.use_atom_residual:
            self.atom_projection = MLP(
                input_dim=atom_fdim,
                output_dim=hidden_dim,
                hidden_dim=self.projection_config['hidden_dim'],
                n_layers=self.projection_config['n_layers'],
                dropout=self.projection_config['dropout'],
                activation=self.projection_config['activation']
            )
        if self.use_bond_residual:
            self.bond_projection = MLP(
                input_dim=bond_fdim,
                output_dim=hidden_dim,
                hidden_dim=self.projection_config['hidden_dim'],
                n_layers=self.projection_config['n_layers'],
                dropout=self.projection_config['dropout'],
                activation=self.projection_config['activation']
            )

        # Set the aggregator
        if aggregator == 'mean':
            self.aggregator = MeanAggregator()
        elif aggregator == 'sum':
            self.aggregator = SumAggregator()
        elif aggregator == 'norm_mean':
            self.aggregator = NormMeanAggregator()
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        self.bn = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()



        # Determine the input dimensions for the FFN
        ffn_input_dim = hidden_dim  # CMPNN mol vector
        if use_atom_residual:
            ffn_input_dim += hidden_dim  # projected atom residual
        if use_bond_residual:
            ffn_input_dim += hidden_dim  # projected bond residual
        ffn_input_dim += global_fdim

        ffn_config = ffn_config if ffn_config else {}
        self.ffn_base = MLP(input_dim=ffn_input_dim,
                       output_dim=output_size,
                       hidden_dim=ffn_config.get('hidden_dim', hidden_dim),
                       n_layers=ffn_config.get('n_layers', 1),
                       dropout=ffn_config.get('dropout', dropout),
                       activation=ffn_config.get('activation', activation)
                       )
        self.ffn = ScaledOutputLayer(self.ffn_base,
                                     output_size=output_size,
                                     scale_init=1.0)

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params if optimizer_params else {}
        self.learning_rate = learning_rate

        # Initialize weights
        initialize_weights(self)

        if metrics is None:
            metrics = {
                "RMSE": torchmetrics.MeanSquaredError(squared=False),
                "MAE": torchmetrics.MeanAbsoluteError(),
                "R2": torchmetrics.R2Score(),
            }
        self.metrics = nn.ModuleDict({
            name: MaskedMetricWrapper(metric, ignore_value=self.ignore_value)
            for name, metric in metrics.items()
        })
        
        self.loss_metric_name, self.loss_fn_ = next(iter(self.metrics.items()))

    def forward(self, f_atoms: torch.Tensor, f_bonds: torch.Tensor,
                a2b: torch.Tensor, b2a: torch.Tensor, b2revb: torch.Tensor,
                a_scope: torch.Tensor, b_scope: torch.Tensor,
                global_features: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            f_atoms: Atom features.
            f_bonds: Bond features.
            global_features: Global features.
        
        Returns:
            Output tensor.
        """
        # Pass through the CMPNN encoder
        atom_hidden = self.model(
            f_atoms=f_atoms,
            f_bonds=f_bonds,
            a2b=a2b,
            b2a=b2a,
            b2revb=b2revb,
            a_scope=a_scope
        )

        # Aggregate atom features to molecule features
        mol_vectors = self.aggregator(atom_hidden, a_scope)

        if self.use_atom_residual:
            atom_residual = self.aggregator(f_atoms, a_scope)
            atom_residual = self.atom_projection(atom_residual)
            mol_vectors = torch.cat([mol_vectors, atom_residual], dim=-1)

        if self.use_bond_residual:
            bond_residual = self.aggregator(f_bonds, b_scope)
            bond_residual = self.bond_projection(bond_residual)
            mol_vectors = torch.cat([mol_vectors, bond_residual], dim=-1)
            

        mol_vectors = self.bn(mol_vectors)

        if global_features is not None:
            mol_vectors = torch.cat([mol_vectors, global_features], dim=-1)
        # Pass through the FFN
        output = self.ffn(mol_vectors)

        return output

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
        
        Returns:
            Loss value.
        """
        # Move tensors to the appropriate device
        atom_features = batch.f_atoms
        bond_features = batch.f_bonds
        a2b = batch.a2b
        b2a = batch.b2a
        b2revb = batch.b2revb
        a_scope = batch.a_scope
        b_scope = batch.b_scope
        targets = batch.y
        global_features = batch.global_features if hasattr(batch, 'global_features') else None

        # Forward pass
        output = self(atom_features, bond_features, a2b, b2a, b2revb, a_scope, b_scope, global_features)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # Compute loss
        loss = self.loss_fn_(output, targets)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, batch_size=targets.size(0))

        if hasattr(self.model, "sampled_depths") and self.model.sampled_depths and self.dynamic_depth is not None:
            depth = self.model.sampled_depths[-1]
            self.training_logs.append({'depth': depth, 'loss': loss.item()})
            self.log('depth_sampled', depth, prog_bar=False)

        print("Learning scale:", self.ffn.scale.data)

        return loss

    def on_train_end(self):
        log_dir = self.trainer.logger.log_dir if self.trainer.logger else "."
        print(f"training_logs length = {len(self.training_logs)}")
        # Plot LR schedule if using Noam
        if self.plot_lr:
            optimizer = self.trainer.optimizers[0]
            if isinstance(optimizer, NoamLikeOptimizer):
                path = os.path.join(log_dir, "lr_schedule.png")
                optimizer.plot_lr_schedule(save_path=path)
                print(f"[DyMPN] Saved LR schedule to {path}")

        # Save training log and depth-loss plot
        if self.training_logs:


            # Save JSON
            log_path = os.path.join(log_dir, "depth_loss_log.json")
            with open(log_path, 'w') as f:
                json.dump(self.training_logs, f)
            print(f"[DyMPN] Saved training logs to {log_path}")

            # Create depth-loss plot
            df = pd.DataFrame(self.training_logs)
            plt.figure(figsize=(10, 6))
            sns.violinplot(x='depth', y='loss', data=df, inner=None, color='lightgray')
            sns.stripplot(x='depth', y='loss', data=df, color='black', alpha=0.5, jitter=True)
            plt.title("Loss distribution per dynamic depth")
            plt.xlabel("Sampled Depth")
            plt.ylabel("Loss")

            plot_path = os.path.join(log_dir, "depth_loss_plot.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"[DyMPN] Saved depth-loss plot to {plot_path}")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
        
        Returns:
            Loss value.
        """
        # Extract features from the batch
        atom_features = batch.f_atoms
        bond_features = batch.f_bonds
        a2b = batch.a2b
        b2a = batch.b2a
        b2revb = batch.b2revb
        a_scope = batch.a_scope
        b_scope = batch.b_scope
        targets = batch.y
        global_features = batch.global_features if hasattr(batch, 'global_features') else None
        # Forward pass
        output = self(atom_features, bond_features, a2b, b2a, b2revb, a_scope, b_scope, global_features)

        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # Compute loss
        loss = self.loss_fn_(output, targets)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, batch_size=targets.size(0))

        return loss

    def on_test_epoch_start(self):
        """
        Called when the test epoch starts.
        """
        # Initialize an empty list to accumulate test outputs.
        self.test_outputs = []
        for key, metric in self.metrics.items():
            self.metrics[key] = metric

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
        
        Returns:
            Loss value.
        """
        # Extract features from the batch
        atom_features = batch.f_atoms
        bond_features = batch.f_bonds
        a2b = batch.a2b
        b2a = batch.b2a
        b2revb = batch.b2revb
        a_scope = batch.a_scope
        b_scope = batch.b_scope
        targets = batch.y
        global_features = batch.global_features if hasattr(batch, 'global_features') else None

        # Forward pass
        output = self(atom_features, bond_features, a2b, b2a, b2revb, a_scope, b_scope, global_features)

        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        self.test_outputs.append({"preds": output.detach(), "targets": targets.detach()})
        return {}

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(),
            lr=self.learning_rate,
            **self.optimizer_params
        )
        self.optimizer_instance = optimizer  # Save it for use in on_train_end
        return optimizer

    def on_test_epoch_end(self):
        # Concatenate predictions and targets from all batches.
        preds = torch.cat([out["preds"] for out in self.test_outputs], dim=0)
        targets = torch.cat([out["targets"] for out in self.test_outputs], dim=0)
        # Compute and log all specified metrics.
        metric_results = {}
        for name, metric in self.metrics.items():
            # Need the metric to be on the same device as the predictions.
            metric_results[name] = metric(preds, targets)
        flat = {f"test_{k.lower()}": v for k, v in metric_results.items()}
        self.log_dict(flat, prog_bar=True)
        # Clear the stored outputs for next test run.
        self.test_outputs.clear()

    def loss_fn(self, predictions, targets):
        """
        Compute the loss between predictions and targets.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
        
        Returns:
            Loss value.
        """
        if self.prediction_type == 'classification':
            loss = nn.BCEWithLogitsLoss()(predictions, targets)
        else:
            loss = nn.MSELoss()(predictions, targets)
        return loss

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Transfer the batch to the specified device.
        """
        return batch.to(device)


class CMPNNLightningModuleTimed(CMPNNLightningModule):
    def forward(self, *args, **kwargs):
        start = time.time()
        out = super().forward(*args, **kwargs)
        print(f"[Timing] forward: {time.time() - start:.4f}s")
        return out

    def training_step(self, batch, batch_idx):
        start = time.time()
        loss = super().training_step(batch, batch_idx)
        print(f"[Timing] training_step (incl. forward): {time.time() - start:.4f}s")
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        start = time.time()
        optimizer.step(closure=optimizer_closure)
        print(f"[Timing] optimizer_step: {time.time() - start:.4f}s")


class MultiCMPNNLightningModule(pl.LightningModule):
    """
    A PyTorch Lightning module for multiple CMPNN Encoder model.
    """

    def __init__(self,
                 atom_fdim: int = 133,
                 bond_fdim: int = 14,
                 global_fdim: int = 0,
                 shared_encoder: bool = False,
                 n_components: int = 2,
                 atom_messages: bool = True,
                 depth: int = 3,
                 output_size: int = 1,
                 hidden_dim: int = 128,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 bias: bool = False,
                 booster: str = 'sum',
                 comm_mode: str = 'add',
                 prediction_type: str = 'regression',
                 aggregator: str = 'mean',
                 use_batch_norm: bool = False,
                 ffn_config: dict = None,
                 optimizer_class: type = torch.optim.Adam,
                 optimizer_params: dict = None,
                 learning_rate: float = 1e-3,
                 metrics: dict = None,
                 plot_lr: bool = False,
                 dynamic_depth: str = None,
                 use_atom_residual: bool = False,
                 use_bond_residual: bool = False,
                 projection_config: dict = None,
                 ignore_value: float = -10.0,
                 target_normalizer = None,
                    cont_indices: list = None,
                    per_indices: list = None,
                    masked_loss: bool = False,
                    mixed_loss: bool = False,
                    scaled_output: bool = False,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.output_size = output_size
        self.prediction_type = prediction_type
        self.target_normalizer = target_normalizer
        self.plot_lr = plot_lr
        if self.prediction_type == 'classification':
            self.sigmoid = nn.Sigmoid()
        self.shared_encoder = shared_encoder
        self.training_logs = []
        self.dynamic_depth = dynamic_depth
        self.use_atom_residual = use_atom_residual
        self.use_bond_residual = use_bond_residual
        self.mixed_loss = mixed_loss
        self.scaled_output = scaled_output

        if cont_indices is not None and per_indices is not None:
            self.cont_indices = cont_indices
            self.per_indices = per_indices
        elif target_normalizer is not None and hasattr(target_normalizer, 'kinds'):
            self.kinds = target_normalizer.kinds
            self.cont_indices = [i for i, k in enumerate(self.kinds) if k == 'continuous']

            # periodic indices are pairs [sin, cos] - assume target vector is already expanded
            base_index = len(self.cont_indices)
            per_pairs = sum(1 for k in self.kinds if k == 'periodic')
            self.per_indices = list(range(base_index, base_index + 2 * per_pairs))
            self.per_angle_indices = list(range(len(self.cont_indices), len(self.cont_indices) + (len(self.per_indices) // 2)))
        else:
            raise ValueError("Cannot determine target kinds. Please provide either `cont_indices` and `per_indices` or `target_normalizer.kinds`.")



        self.projection_config = projection_config if projection_config else {
            'hidden_dim': hidden_dim,
            'n_layers': 2,
            'dropout': dropout,
            'activation': activation
        }


        if shared_encoder:
            encoder = CMPNNEncoder(
                atom_fdim, bond_fdim, atom_messages=atom_messages,
                depth=depth, hidden_dim=hidden_dim, dropout=dropout,
                activation=activation, bias=bias, booster=booster,
                comm_mode=comm_mode, dynamic_depth=dynamic_depth
            )
            self.encoders = nn.ModuleList([encoder] * n_components)
        else:
            self.encoders = nn.ModuleList([
                CMPNNEncoder(
                    atom_fdim, bond_fdim, atom_messages=atom_messages,
                    depth=depth, hidden_dim=hidden_dim, dropout=dropout,
                    activation=activation, bias=bias, booster=booster,
                    comm_mode=comm_mode, dynamic_depth=dynamic_depth
                ) for _ in range(n_components)
            ])

        if use_atom_residual:
            self.atom_projections = nn.ModuleList([
                MLP(input_dim=atom_fdim, output_dim=hidden_dim, **self.projection_config)
                for _ in range(n_components)
            ])
        if use_bond_residual:
            self.bond_projections = nn.ModuleList([
                MLP(input_dim=bond_fdim, output_dim=hidden_dim, **self.projection_config)
                for _ in range(n_components)
            ])

        # Set the aggregator
        if aggregator == 'mean':
            self.aggregator = MeanAggregator()
        elif aggregator == 'sum':
            self.aggregator = SumAggregator()
        elif aggregator == 'norm_mean':
            self.aggregator = NormMeanAggregator()
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")
        self.bn = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()

        # Determine the input dimensions for the FFN
        # Determine the input dimensions for the FFN
        per_component_dim = hidden_dim
        if use_atom_residual:
            per_component_dim += hidden_dim
        if use_bond_residual:
            per_component_dim += hidden_dim
        per_component_dim += global_fdim

        ffn_input_dim = per_component_dim * n_components

        ffn_config = ffn_config if ffn_config else {}

        self.ffn_base = MLP(input_dim=ffn_input_dim,
                       output_dim=output_size,
                       hidden_dim=ffn_config.get('hidden_dim', hidden_dim),
                       n_layers=ffn_config.get('n_layers', 3),
                       dropout=ffn_config.get('dropout', dropout),
                       activation=ffn_config.get('activation', activation)
                       )
        if scaled_output:
            self.ffn = ScaledOutputLayer(self.ffn_base,
                                         output_size=output_size,
                                         scale_init=1.0)
        else:
            self.ffn = self.ffn_base


        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params if optimizer_params else {}
        self.learning_rate = learning_rate
        self.ignore_value = ignore_value
        # Initialize weights
        initialize_weights(self)

        # If metrics is a list of strings, turn them into instantiated metric objects
        if isinstance(metrics, list):
            self.metrics = nn.ModuleDict({
                name: get_metric_by_name(name, cont_indices=self.cont_indices, per_indices=self.per_indices, per_angle_indices = self.per_angle_indices, ignore_value=self.ignore_value)
                for name in metrics
            })
        elif isinstance(metrics, dict):
            self.metrics = nn.ModuleDict(metrics)
        else:
            raise ValueError("metrics should be either a list of metric names or a dict of instantiated metric objects.")

        self.loss_metric_name, self.loss_fn_ = next(iter(self.metrics.items()))


    def forward(self, batch) -> torch.Tensor:
        
        component_outputs = []

        for i, encoder in enumerate(self.encoders):
            comp = batch.components[i]

            # Run the encoder
            atom_hidden = encoder(
                f_atoms=comp.f_atoms,
                f_bonds=comp.f_bonds,
                a2b=comp.a2b,
                b2a=comp.b2a,
                b2revb=comp.b2revb,
                a_scope=comp.a_scope
            )

            if self.training and hasattr(encoder, 'sampled_depths') and encoder.sampled_depths:
                self.training_logs.append({
                    'component': i,
                    'depth': encoder.sampled_depths[-1],
                    'loss': None
                })


            # Aggregate atom features
            mol_vector = self.aggregator(atom_hidden, comp.a_scope)

            # Optional atom residual projection
            if getattr(self, 'use_atom_residual', False):
                atom_res = self.aggregator(comp.f_atoms, comp.a_scope)
                if self.shared_encoder:
                    atom_res = self.atom_projections[0](atom_res)
                else:
                    atom_res = self.atom_projections[i](atom_res)
                mol_vector = torch.cat([mol_vector, atom_res], dim=-1)

            # Optional bond residual projection
            if getattr(self, 'use_bond_residual', False):
                bond_res = self.aggregator(comp.f_bonds, comp.b_scope)
                if self.shared_encoder:
                    bond_res = self.bond_projections[0](bond_res)
                else:
                    bond_res = self.bond_projections[i](bond_res)
                mol_vector = torch.cat([mol_vector, bond_res], dim=-1)

            mol_vector = self.bn(mol_vector)

            # Add global features if available
            global_features = getattr(comp, 'global_features', None)
            if global_features is not None:
                mol_vector = torch.cat([mol_vector, global_features], dim=-1)

            component_outputs.append(mol_vector)

        merged = torch.cat(component_outputs, dim=-1)
        output = self.ffn(merged)

        return output

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
        
        Returns:
            Loss value.
        """
        output = self.forward(batch)
        targets = batch.y
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        # Compute loss
        if self.scaled_output:

            if self.target_normalizer is not None:
                output = torch.stack([
                    self.target_normalizer.inverse_transform(p.cpu()).to(output.device)
                    for p in output
                ])
                targets = torch.stack([
                    self.target_normalizer.inverse_transform(t.cpu()).to(output.device)
                    for t in targets
                ])
            loss = self.loss_fn_.forward_test(output, targets)
        else:
            loss = self.loss_fn_(output, targets)
        preds = output
        if batch_idx % 50 == 0:
            pred_means = preds.mean(dim=0)
            pred_stds = preds.std(dim=0)
            target_means = targets.mean(dim=0)
            target_stds = targets.std(dim=0)
            print(f"[Batch {batch_idx}] Pred stats per target:")
            for i, (pred_mean, pred_std, target_mean, target_std) in enumerate(zip(pred_means, pred_stds, target_means, target_stds)):
                print(f"  - Target {i}: pred μ={pred_mean.item():.4f}, σ={pred_std.item():.4f} | target μ={target_mean.item():.4f}, σ={target_std.item():.4f}")

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, batch_size=targets.size(0))

        if self.training_logs:
            loss_value = loss.item()
            for j in range(len(self.encoders)):
                self.training_logs[-j - 1]['loss'] = loss_value

        if batch_idx % 50 == 0:
            print("Learned scale:", self.ffn.log_scale.exp().detach().cpu())

        import numpy as np
        pred_angles = torch.atan2(preds[:, 1], preds[:, 0]) * 180.0 / np.pi
        true_angles = torch.atan2(targets[:, 1], targets[:, 0]) * 180.0 / np.pi

        self.all_train_theta_pred.append(pred_angles.detach())
        self.all_train_theta_true.append(true_angles.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
        
        Returns:
            Loss value.
        """
        output = self.forward(batch)
        targets = batch.y
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        # Compute loss
        loss = self.loss_fn_(output, targets)
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, batch_size=targets.size(0))
        return loss

    def on_test_epoch_start(self):
        """
        Called when the test epoch starts.
        """
        # Initialize an empty list to accumulate test outputs.
        self.test_outputs = []
        for key, metric in self.metrics.items():
            self.metrics[key] = metric

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
        
        Returns:
            Loss value.
        """
        output = self.forward(batch)
        targets = batch.y
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        self.test_outputs.append({"preds": output.detach(), "targets": targets.detach()})
        return {}

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(),
            lr=self.learning_rate,
            **self.optimizer_params
        )
        self.optimizer_instance = optimizer
        return optimizer

    def on_train_end(self):
        if not self.training_logs:
            return

        log_dir = self.trainer.logger.log_dir if self.trainer.logger else "."

        # Save raw logs
        log_path = os.path.join(log_dir, "component_depth_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.training_logs, f)
        print(f"[MultiCMPNN] Saved training depth logs to {log_path}")

        # Create a dataframe from logs
        df = pd.DataFrame(self.training_logs)

        # Group stats for annotation
        group_stats = df.groupby(['depth', 'component'])['loss'].agg(['count', 'mean']).reset_index()

        # Start plotting
        plt.figure(figsize=(12, 7))
        ax = sns.violinplot(data=df, x='depth', y='loss', hue='component',
                            dodge=True, scale='width', inner=None, palette='Set2')
        sns.stripplot(data=df, x='depth', y='loss', hue='component',
                    dodge=True, jitter=True, alpha=0.3, color='black', ax=ax)

        # Log scale for loss
        plt.yscale('log')
        plt.title("Loss Distribution per Sampled Depth and Component")
        plt.xlabel("Sampled Depth")
        plt.ylabel("Loss (log scale)")

        # Remove duplicated legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Component")

        # Add annotations
        for _, row in group_stats.iterrows():
            depth = row['depth']
            component = row['component']
            mean_loss = row['mean']
            count = int(row['count'])

            offset = -0.2 if component == 0 else 0.2
            x_pos = depth + offset - 1

            ax.text(x_pos, mean_loss, f"{mean_loss:.1f}\n(n={count})",
                    ha='center', va='bottom', fontsize=8, color='blue')

        # Save plot and summary CSV
        plot_path = os.path.join(log_dir, "component_depth_loss_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[MultiCMPNN] Saved depth-loss plot to {plot_path}")

        group_stats.to_csv(os.path.join(log_dir, "depth_component_loss_summary.csv"), index=False)


    def on_test_epoch_end(self):
        # Concatenate predictions and targets
        preds = torch.cat([out["preds"] for out in self.test_outputs], dim=0)
        targets = torch.cat([out["targets"] for out in self.test_outputs], dim=0)
        self.test_outputs.clear()

        device = preds.device
        if self.target_normalizer is not None:
            preds = self.target_normalizer.inverse_transform(preds.cpu()).to(device)
            targets = self.target_normalizer.inverse_transform(targets.cpu()).to(device)

        # Compute and log masked metrics per target dimension
        metric_results = {}
        output_dim = preds.shape[1] if preds.ndim > 1 else 1
        for i in range(output_dim):
            for name, metric in self.metrics.items():
                if hasattr(metric, "forward_test_target"):
                    score = metric.forward_test_target(preds, targets, target_idx=i)
                else:
                    mask = targets[:, i] != self.ignore_value
                    if mask.any():
                        score = metric(preds[:, i][mask], targets[:, i][mask])
                    else:
                        score = torch.tensor(float('nan'), device=preds.device)
                metric_results[f"{name}_t{i}"] = score

        # Aggregate
        for name, metric in self.metrics.items():
            if hasattr(metric, "forward_test"):
                score = metric.forward_test(preds, targets)
                metric_results[name] = score
            else:
                mask = targets != self.ignore_value
                if mask.any():
                    score = metric(preds[mask], targets[mask])
                else:
                    score = torch.tensor(float('nan'), device=preds.device)
                metric_results[name] = score

        # Plot histograms of residuals
        log_dir = self.trainer.logger.log_dir if self.trainer.logger else "."
        fig, axes = plt.subplots(1, output_dim, figsize=(4 * output_dim, 3))
        for i in range(output_dim):
            mask_i = targets[:, i] != self.ignore_value
            res_i = (preds[:, i] - targets[:, i])[mask_i]
            ax = axes[i] if output_dim > 1 else axes
            ax.hist(res_i.cpu().numpy(), bins=30, alpha=0.7)
            ax.axvline(0, color='gray', linestyle='--')
            ax.set_title(f"Target {i}")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Count")
        fig.suptitle("Residual Distribution per Target")
        fig.tight_layout()
        plt.savefig(os.path.join(log_dir, "residual_histograms.png"))
        plt.close()
        self.log_dict(metric_results, prog_bar=True)

        grouped_results = {}
        for key, val in metric_results.items():
            if "_t" in key:
                base, tidx = key.rsplit("_t", 1)
                grouped_results.setdefault(f"t{tidx}", {})[base] = val
            else:
                grouped_results.setdefault("all", {})[key] = val

        print("\nTest Metrics (Grouped):")
        for tidx in sorted(grouped_results):
            print(f"Target {tidx}:")
            for k, v in grouped_results[tidx].items():
                print(f"  - {k}: {v:.4f}")

        return metric_results
    
    def loss_fn(self, predictions, targets):
        """
        Compute the loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
        Returns:
            Loss value.
        """
        if self.prediction_type == 'classification':
            loss = nn.BCEWithLogitsLoss()(predictions, targets)
        else:
            loss = nn.MSELoss()(predictions, targets)
        return loss

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Transfer the batch to the specified device.
        """
        return batch.to(device)


class MultiCMPNNLightningModuleTimed(MultiCMPNNLightningModule):
    def forward(self, *args, **kwargs):
        start = time.time()
        out = super().forward(*args, **kwargs)
        print(f"[Timing] forward: {time.time() - start:.4f}s")
        return out

    def training_step(self, batch, batch_idx):
        start = time.time()
        loss = super().training_step(batch, batch_idx)
        print(f"[Timing] training_step (incl. forward): {time.time() - start:.4f}s")
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        start = time.time()
        optimizer.step(closure=optimizer_closure)
        print(f"[Timing] optimizer_step: {time.time() - start:.4f}s")


class MultiCMPNNLightningModuleClean(pl.LightningModule):
    def __init__(
        self,
        atom_fdim: int = 133,
        bond_fdim: int = 14,
        global_fdim: int = 0,
        shared_encoder: bool = False,
        n_components: int = 2,
        atom_messages: bool = True,
        depth: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        activation: str = 'relu',
        bias: bool = False,
        booster: str = 'sum',
        comm_mode: str = 'add',
        aggregator: str = 'mean',
        use_batch_norm: bool = False,
        dynamic_depth: str = None,
        use_atom_residual: bool = False,
        use_bond_residual: bool = False,
        projection_config: dict = None,
        prediction_type: str = 'regression',
        ffns: List[callable] = None,
        ffn_groups: List[List[int]] = None,
        output_size: int = 1,
        scaled_output: bool = False,
        plot_lr: bool = False,
        optimizer_class: type = torch.optim.Adam,
        optimizer_params: dict = None,
        learning_rate: float = 1e-3,
        target_normalizer=None,
        cont_indices: list = None,
        per_angle_180_indices: list = None,
        per_indices: list = None,
        metrics: dict = None,
        ignore_value: float = -10.0,
        focus_target_idx: Optional[int] = None,
        ffn_config: dict = None,
        base_train_ds: Optional[Dataset] = None,
        theta_true_aug: Optional[Dataset] = None,
        adaptive_jiggle: bool = False,
        apply_bottleneck: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.all_theta_pred = []
        self.all_theta_true = []
        # Core settings
        from copy import deepcopy
        self.extra_angle_penalty_weight = 0.5
        self.adaptive_jiggle = adaptive_jiggle
        self.all_train_theta_pred = []
        self.all_train_theta_true = []
        self.all_val_theta_pred = []
        self.all_val_theta_true = []
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.optimizer_params = optimizer_params if optimizer_params else {}
        self.original_base_train_ds = deepcopy(base_train_ds)
        self.base_train_ds = base_train_ds  # original non-jiggled dataset
        self.all_train_embeddings = []
        self.all_val_embeddings   = []
        self.output_size = output_size
        self.prediction_type = prediction_type
        self.target_normalizer = target_normalizer
        self.plot_lr = plot_lr
        self.shared_encoder = shared_encoder
        self.dynamic_depth = dynamic_depth
        self.use_atom_residual = use_atom_residual
        self.use_bond_residual = use_bond_residual
        self.scaled_output = scaled_output
        self.ignore_value = ignore_value
        self.training_logs = []
        self.focus_target_idx = focus_target_idx
        self.ffn_groups = ffn_groups if ffn_groups else []
        self.ffn_config = ffn_config if ffn_config else {}
        self.apply_bottleneck = apply_bottleneck
        if self.apply_bottleneck:
            self.atom_feature_bottleneck = nn.Sequential(
    nn.Linear(atom_fdim, 64),
    nn.LayerNorm(64),
    nn.ReLU()
)
            self.atom_fdim = 64
        else:
            self.atom_fdim = atom_fdim
            self.atom_feature_bottleneck = nn.Identity()
        
        N = len(base_train_ds)
        self.register_buffer("molecule_errors", torch.zeros(N, dtype=torch.float32))
        # if self.focus_target_idx is not None:
        #     if self.scaled_output:
        #         print(f"[INFO] Disabling scaled output for focus target index {self.focus_target_idx}")
        #         self.scaled_output = False

        if prediction_type == 'classification':
            self.sigmoid = nn.Sigmoid()
        self.register_buffer("theta_true_aug", theta_true_aug)
        self._init_indices(cont_indices, per_indices, per_angle_180_indices)
        self._init_encoders(self.atom_fdim, bond_fdim, atom_messages, depth, hidden_dim, dropout, activation, bias, booster, comm_mode, n_components)
        self._init_projections(self.atom_fdim, bond_fdim, hidden_dim, projection_config, n_components)
        self._init_aggregator(aggregator, hidden_dim, use_batch_norm, global_fdim, n_components)
        self._init_ffn(ffns, self.ffn_groups, hidden_dim, global_fdim, output_size, dropout, activation, ffn_config=self.ffn_config)
        # self._init_optimizer(optimizer_class, optimizer_params, learning_rate)
        self._init_metrics(metrics)

        initialize_weights(self)
#         self.jiggle_schedule = [
#     (0, 0.0),        # start with perfect
#     (100, 0.05),      # ~3°
#     (300, 0.1),      # ~6°
#     (600, 0.15),     # ~9°
#     (1000, 0.2),      # ~11°
# ]
        self.jiggle_schedule = [
        (0, 0.03),       # 2°
        (200, 0.05),     # 3°
        (400, 0.08),     # 5°
        (700, 0.13),     # 7.5°
        (1000, 0.2),     # max 11°
    ]

        self.flat_mol_refs = []
        self.mol_to_reaction_idx = []

        for reaction_idx, datapoint in enumerate(self.base_train_ds):
            for mol in datapoint:
                self.flat_mol_refs.append(mol)
                self.mol_to_reaction_idx.append(reaction_idx)

        # Use torch.Tensor for indexing
        self.mol_to_reaction_idx = torch.tensor(self.mol_to_reaction_idx, dtype=torch.long)

    def get_current_jiggle(self,epoch, jiggle_schedule):
        """
        Linearly interpolate jiggle amount based on epoch.
        jiggle_schedule: list of (epoch, jiggle_radians) points
        """
        epochs, jiggles = zip(*jiggle_schedule)
        if epoch <= epochs[0]:
            return jiggles[0]
        elif epoch >= epochs[-1]:
            return jiggles[-1]
        else:
            for i in range(len(epochs) - 1):
                if epochs[i] <= epoch < epochs[i + 1]:
                    # Linear interpolation between two points
                    t = (epoch - epochs[i]) / (epochs[i + 1] - epochs[i])
                    return jiggles[i] * (1 - t) + jiggles[i + 1] * t
    def _init_indices(self, cont_indices, per_indices, per_angle_180_indices):
        if cont_indices is not None and per_indices is not None and per_angle_180_indices is not None:
            self.cont_indices = cont_indices
            self.per_indices = per_indices
            self.per_angle_indices = list(range(len(self.cont_indices), len(self.cont_indices) + len(self.per_indices) // 2))
            self.per_angle_180_indices = per_angle_180_indices
            self.num_cont = len(self.cont_indices)
            self.num_per = len(self.per_angle_indices)
        elif self.target_normalizer is not None and hasattr(self.target_normalizer, 'kinds'):
            self.kinds = self.target_normalizer.kinds
            self.cont_indices = [i for i, k in enumerate(self.kinds) if k == 'continuous']
            base_index = len(self.cont_indices)
            per_pairs = sum(1 for k in self.kinds if k == 'periodic')
            self.per_indices = list(range(base_index, base_index + 2 * per_pairs))
            self.per_angle_indices = list(range(len(self.cont_indices), len(self.cont_indices) + per_pairs))
            self.num_cont = len(self.cont_indices)
            self.num_per = len(self.per_angle_indices)
        else:
            raise ValueError("Cannot determine target kinds.")

        if self.focus_target_idx is not None:
            if self.focus_target_idx in self.cont_indices:
                self.cont_indices = [0]
                self.per_indices = []
                self.per_angle_indices = []
            elif self.focus_target_idx in self.per_angle_indices:
                self.cont_indices = []
                self.per_indices = [0, 1]  # sin, cos for the single periodic
                self.per_angle_indices = [0]
            else:
                raise ValueError(f"focus_target_idx={self.focus_target_idx} not found in target indices.")


    def _init_encoders(self, atom_fdim, bond_fdim, atom_messages, depth, hidden_dim, dropout, activation, bias, booster, comm_mode, n_components):
        encoder_factory = lambda: CMPNNEncoder(
            atom_fdim, bond_fdim,
            atom_messages=atom_messages,
            depth=depth,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
            booster=booster,
            comm_mode=comm_mode,
            dynamic_depth=self.dynamic_depth
        )
        if self.shared_encoder:
            shared = encoder_factory()
            self.encoders = nn.ModuleList([shared] * n_components)
        else:
            self.encoders = nn.ModuleList([encoder_factory() for _ in range(n_components)])

    def _init_projections(self, atom_fdim, bond_fdim, hidden_dim, projection_config, n_components):
        config = projection_config or {
            'hidden_dim': hidden_dim,
            'n_layers': 2,
            'dropout': self.hparams.dropout,
            'activation': self.hparams.activation
        }
        if self.use_atom_residual:
            self.atom_projections = nn.ModuleList([
                MLP(atom_fdim, hidden_dim, **config) for _ in range(n_components)
            ])
        if self.use_bond_residual:
            self.bond_projections = nn.ModuleList([
                MLP(bond_fdim, hidden_dim, **config) for _ in range(n_components)
            ])

    def _init_aggregator(self, aggregator, hidden_dim, use_batch_norm, global_fdim, n_components):
        agg_cls = {
            'mean': MeanAggregator,
            'sum': SumAggregator,
            'norm_mean': NormMeanAggregator
        }.get(aggregator)
        if agg_cls is None:
            raise ValueError(f"Unknown aggregator: {aggregator}")
        self.aggregator = agg_cls()

        dim = hidden_dim
        if self.use_atom_residual:
            dim += hidden_dim
        if self.use_bond_residual:
            dim += hidden_dim
        dim += global_fdim

        self.bn = nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity()

    def _init_ffn(self, 
                  ffns,
                  ffn_groups: List[List],
                  hidden_dim: int,
                  global_fdim: int,
                  output_size: int,
                  dropout: float,
                  activation,
                  ffn_config):

        # 1) Build the common input dimension
        per_comp_dim = hidden_dim
        if self.use_atom_residual:  per_comp_dim += hidden_dim
        if self.use_bond_residual:  per_comp_dim += hidden_dim
        per_comp_dim += global_fdim
        input_dim = per_comp_dim * self.hparams.n_components

        # 2) Initialize the FFN - if ffns is a list, it must match the number of groups
        if isinstance(ffns, list):
            if len(ffns) != len(ffn_groups):
                raise ValueError(f"Got {len(ffns)} heads but {len(ffn_groups)} groups")
            
            self.ffn_heads = nn.ModuleList(f(input_dim=input_dim) for f in ffns)
            self.ffn = None

            # 3) Build a map: target_idx -> (head_idx, slice_in_head_output)
            self.target_head_map = {}
            for head_idx, group in enumerate(ffn_groups):
                head = self.ffn_heads[head_idx]

                # assume each head has 'output_dim' attribute
                out_dim = getattr(head, 'output_dim', None)
                if out_dim is None:
                    raise ValueError(f"Head {head_idx} does not have 'output_dim' attribute.")
                per_target = out_dim//len(group)
                if per_target * len(group) != out_dim:
                    raise ValueError(f"Output dimension {out_dim} is not divisible by group size {len(group)}.")
                for i, idx in enumerate(group):
                    start = i * per_target
                    end = start + per_target
                    self.target_head_map[idx] = (head_idx, slice(start, end))
        elif callable(ffns):
            # Single head for all targets
            self.ffn = ffns(input_dim=input_dim)
            self.ffn_heads = None
            if self.scaled_output:
                self.ffn = ScaledOutputLayer(self.ffn, output_size=output_size, scale_init=1.0)
        else:
            raise ValueError("`ffns` must be a callable or a list of callables.")

    # def _init_optimizer(self, optimizer_class, optimizer_params, learning_rate):
    #     self.optimizer_class = optimizer_class
    #     self.optimizer_params = optimizer_params if optimizer_params else {}
    #     self.learning_rate = learning_rate

    def _init_metrics(self, metrics):
        self.metrics = nn.ModuleDict()
        self.loss_fn_ = None  # <- initialize it here

        if metrics is None:
            return

        for i, metric in enumerate(metrics):
            if isinstance(metric, str):
                metric_name = metric.lower()
                metric_module = get_metric_by_name(
                    name=metric_name,
                    cont_indices=self.cont_indices,
                    per_indices=self.per_indices,
                    per_angle_indices=self.per_angle_indices,
                    ignore_value=self.ignore_value
                )
            elif isinstance(metric, nn.Module):
                metric_name = metric.__class__.__name__.lower()
                metric_module = metric
            else:
                raise TypeError(f"Unsupported metric type: {type(metric)}. Must be string or nn.Module.")

            self.metrics[metric_name] = metric_module

            # Assign the first metric as the loss function
            if i == 0:
                self.loss_fn_ = metric_module

        if self.loss_fn_ is None:
            raise ValueError("No valid loss function could be initialized from metrics.")
        
        self._check_loggauss_loss()
    
    def _check_loggauss_loss(self):
        from cmpnn.loss.loggauss_loss import LogGaussLoss
        from cmpnn.models.ffns import LogGaussianHead

        # 3a) Detect any Gaussian heads
        has_gauss = False
        if getattr(self, 'ffn_heads', None):
            has_gauss = any(isinstance(h, LogGaussianHead) for h in self.ffn_heads)
        elif isinstance(getattr(self, 'ffn', None), LogGaussianHead):
            has_gauss = True

        # 3b) Safely grab the loss (or None)
        loss = getattr(self, 'loss_fn_', None)

        # 3c) If you have Gaussian heads but the loss isn’t LogGaussLoss → error
        if has_gauss and not isinstance(loss, LogGaussLoss):
            raise ValueError(
                "Detected LogGaussianHead but loss function is not LogGaussLoss. "
                "Please configure LogGaussLoss when using LogGaussianHead."
            )

        # 3d) If you’re using LogGaussLoss, but never declared a Gaussian head → error
        if isinstance(loss, LogGaussLoss) and not has_gauss:
            raise ValueError(
                "LogGaussLoss requires a LogGaussianHead in your FFN configuration. "
                "Please add LogGaussianHead when using LogGaussLoss."
            )

    def forward(self, batch) -> torch.Tensor:
        for i in range(len(self.encoders)):
            comp = batch.components[i]

            if torch.isnan(comp.f_atoms).any():
                print(f"NaN in f_atoms of component {i}")
            if hasattr(comp, "y") and torch.isnan(comp.y).any():
                print(f"NaN in y of component {i}")
        

        component_outputs = [
            self._process_component(batch.components[i], self.encoders[i], i)
            for i in range(len(self.encoders))
        ]
        merged = torch.cat(component_outputs, dim=-1)
        # 1) encode & merge your components
        if self.training:
            # keep it on CPU and detached
            self.all_train_embeddings.append(merged.detach().cpu())
        else:
            # for val we only care at epoch‐50, but it's harmless to stash every val step
            self.all_val_embeddings.append(merged.detach().cpu())

        # 2) if multi-head, run each head and keep raw outputs
        if getattr(self, 'ffn_heads', None):
            head_outs = [head(merged) for head in self.ffn_heads]
            chunks = []
            # target_head_map: tgt_idx -> (head_idx, slice)
            for tgt in sorted(self.target_head_map):
                head_idx, slot = self.target_head_map[tgt]
                out = head_outs[head_idx]
                if isinstance(out, tuple):
                    # LogGaussianHead: include logvar only if the loss needs it
                    mu, logvar = out
                    chunks.append(mu[:, slot])
                    # check if current loss_fn_ expects logvar
                    if getattr(self, 'loss_fn_', None) is not None and \
                       getattr(self.loss_fn_, 'needs_logvar', False):
                        chunks.append(logvar[:, slot])
                else:
                    # Regular head (e.g. PeriodicHead)
                    chunks.append(out[:, slot])
            # 3) concatenate per-target outputs
            preds = torch.cat(chunks, dim=-1)
        else:
            # single-head fallback: direct network output
            preds = self.ffn(merged)

        if self.training and torch.isnan(preds).any():
            print("[DEBUG] NaN detected in output predictions")
            print(preds[torch.isnan(preds)])

        return preds


    def _process_component(self, comp, encoder, idx):

        f_atoms_bottleneck = self.atom_feature_bottleneck(comp.f_atoms)
        if self.training:
            f_atoms_bottleneck = f_atoms_bottleneck + torch.randn_like(f_atoms_bottleneck) * 0.01


        atom_hidden = encoder(
            f_atoms=f_atoms_bottleneck,
            f_bonds=comp.f_bonds,
            a2b=comp.a2b,
            b2a=comp.b2a,
            b2revb=comp.b2revb,
            a_scope=comp.a_scope
        )
        if self.training and hasattr(encoder, 'sampled_depths') and encoder.sampled_depths:
            self.training_logs.append({'component': idx, 'depth': encoder.sampled_depths[-1], 'loss': None})

        mol_vector = self.aggregator(atom_hidden, comp.a_scope)

        if self.use_atom_residual:
            atom_res = self.aggregator(f_atoms_bottleneck, comp.a_scope)
            proj = self.atom_projections[0 if self.shared_encoder else idx](atom_res)
            mol_vector = torch.cat([mol_vector, proj], dim=-1)

        if self.use_bond_residual:
            bond_res = self.aggregator(comp.f_bonds, comp.b_scope)
            proj = self.bond_projections[0 if self.shared_encoder else idx](bond_res)
            mol_vector = torch.cat([mol_vector, proj], dim=-1)

        # 👇 Shift adding global features BEFORE batchnorm
        if hasattr(comp, 'global_features') and comp.global_features is not None:
            mol_vector = torch.cat([mol_vector, comp.global_features], dim=-1)

        mol_vector = self.bn(mol_vector) 

        if hasattr(comp, 'global_features') and comp.global_features is not None:
            mol_vector = torch.cat([mol_vector, comp.global_features], dim=-1)

        return mol_vector

    def _compute_loss(self, preds, targets, batch):
        if self.focus_target_idx is not None:
            preds = preds[:, self.focus_target_idx].unsqueeze(1)
            targets = targets[:, self.focus_target_idx].unsqueeze(1)

        if self.ffn_heads is not None:
            # Handle log-Gaussian head or others if needed
            if "LogGaussianHead" in [type(m).__name__ for m in self.ffn_heads]:
                log_gauss_idxs = [
                    i for i, head in enumerate(self.ffn_heads)
                    if type(head).__name__ == "LogGaussianHead"
                ]
                outpdims = sum([getattr(self.ffn_heads[i], 'output_dim', None) for i in log_gauss_idxs])
                # (Do any special handling here as needed)


        # -- compute raw per-sample loss
        loss = self.loss_fn_.forward(preds, targets)


        return loss, preds, targets

    def training_step(self, batch, batch_idx):

        # # Epoch aware loss
        # if hasattr(self.loss_fn_, 'set_epoch'):
        #     self.loss_fn_.set_epoch(self.current_epoch)

        # # Compute current jiggle
        # #jiggle_amnt = self.get_current_jiggle(self.current_epoch, self.jiggle_schedule)
        # jiggle_amnt = 0

        # # Fetch each sample's global index into theta_true_aug
        # idx = batch.components[0].idx
        # clean = self.theta_true_aug[idx]

        # # Add adaptive noise and wrap to [-pi, pi]
        # noise = jiggle_amnt * torch.randn_like(clean)
        # ang = (clean + noise + np.pi) % (2 * np.pi) - np.pi

        # # Build sin/cos + tiny noise + normalize
        # cos_sin = torch.stack([torch.sin(ang), torch.cos(ang)], dim=1)
        # eps = 0.001 * torch.randn_like(cos_sin)
        # cos_sin = cos_sin + eps
        # cos_sin = cos_sin / (cos_sin.norm(dim=1, keepdim=True) + 1e-8)
        # batch.y = cos_sin


        preds = self.forward(batch)

        #targets = batch.y.unsqueeze(1) if batch.y.dim() == 1 else batch.y
        targets = batch.y
        # =============================
        # 🔥 NEW DEBUG: Print target sample
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            print(f"[DEBUG] Train targets sample (epoch {self.current_epoch}): {targets[:5]}")
        # =============================

        loss, preds_out, targets_out = self._compute_loss(preds, targets, batch)

        if torch.isnan(loss):
            print(f"[DEBUG] NaN loss detected. preds: {preds_out[:5]}, targets: {targets_out[:5]}")
        if torch.isnan(preds).any():
            print("NaN in model output")
        if torch.isnan(targets).any():
            print("NaN in targets")

        self.log("train_loss", loss, prog_bar=True, batch_size=targets.size(0), on_epoch=True, on_step=False)

        if self.trainer is not None and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]['lr']
            self.log("lr", current_lr, prog_bar=False, on_step=False, on_epoch=True, batch_size=targets.size(0))

            if "weight_decay" in optimizer.param_groups[0]:
                current_wd = optimizer.param_groups[0]['weight_decay']
                self.log("weight_decay", current_wd, prog_bar=False, on_step=False, on_epoch=True, batch_size=targets.size(0))

        with torch.no_grad():
            sin_pred, cos_pred = preds[:, 0], preds[:, 1]
            sin_true, cos_true = targets[:, 0], targets[:, 1]

            theta_pred = torch.atan2(sin_pred, cos_pred) * 180.0 / math.pi
            theta_true = torch.atan2(sin_true, cos_true) * 180.0 / math.pi

            self.all_theta_pred.append(theta_pred)
            self.all_theta_true.append(theta_true)

            if self.current_epoch % 10 == 0:
                self.all_train_theta_pred.append(theta_pred)
                self.all_train_theta_true.append(theta_true)

            if hasattr(batch.components[0], "idx"):
                residual = (theta_pred - theta_true)
                residual = (residual + 180) % 360 - 180
                residual_error = residual.abs()

                if not hasattr(self, "molecule_errors"):
                    n_molecules = sum(len(dp) for dp in self.base_train_ds)
                    self.molecule_errors = torch.zeros(n_molecules, device=residual_error.device)

                idxs = batch.components[0].idx
                self.molecule_errors[idxs] = residual_error

        if self.training_logs:
            for j in range(len(self.encoders)):
                self.training_logs[-j - 1]['loss'] = loss.item()
        self.get_batch_size(targets)

        # true_angles = torch.atan2(targets[:, 0], targets[:, 1]) * 180.0 / math.pi
        # bin_ids = np.digitize(true_angles.cpu(), self.angle_bins) - 1
        # bins = np.linspace(-180, +180, 37)
        # counts = np.bincount(bin_ids, minlength=len(bins)-1)
        # # avoid div0
        # counts = np.where(counts==0, 1, counts)
        # sample_weights = torch.tensor(1.0 / counts[bin_ids], device=loss.device)

        # loss = (loss * sample_weights).mean()

        return loss

    def on_after_backward(self):
        # called every backward() when grads are ready
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1e9)
        # 'logger=True' puts it in metrics.csv / TensorBoard
        self.log("grad_norm", norm, prog_bar=False, logger=True, on_epoch=False, on_step=True)
    
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f"grad_norm/{name}", param.grad.norm(), on_step=False, on_epoch=True,batch_size=self.batch_size)


    def get_batch_size(self, target):
        """
        Get the batch size from the batch object.
        """
        self.batch_size = target.size(0) if hasattr(target, 'size') else 1
    def validation_step(self, batch, batch_idx):
        preds   = self.forward(batch)
        targets = batch.y if batch.y.dim() == 2 else batch.y.unsqueeze(1)

        with torch.no_grad():
            sin_pred, cos_pred = preds[:, 0], preds[:, 1]
            sin_true, cos_true = targets[:, 0], targets[:, 1]

            theta_pred = torch.atan2(sin_pred, cos_pred) * 180.0 / math.pi
            theta_true = torch.atan2(sin_true, cos_true) * 180.0 / math.pi

            if self.current_epoch % 10 == 0:
                self.all_val_theta_pred.append(theta_pred)
                self.all_val_theta_true.append(theta_true)

        # === Main loss without curriculum weighting ===
        loss = self.loss_fn_(preds, targets)

        # === Logging
        self.log("val_loss", loss, prog_bar=True, batch_size=targets.size(0), on_epoch=True, on_step=False)


        return loss

    def test_step(self, batch, batch_idx):
        preds = self.forward(batch)
        targets = batch.y.unsqueeze(1) if batch.y.dim() == 1 else batch.y
        self.test_outputs.append({"preds": preds.detach(), "targets": targets.detach()})
        return {}

    def on_train_end(self):
        if not self.training_logs:
            return

        log_dir = self.trainer.logger.log_dir if self.trainer.logger else "."

        # Save raw logs
        log_path = os.path.join(log_dir, "component_depth_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.training_logs, f)
        print(f"[MultiCMPNN] Saved training depth logs to {log_path}")

        # Create a dataframe from logs
        df = pd.DataFrame(self.training_logs)

        # Group stats for annotation
        group_stats = df.groupby(['depth', 'component'])['loss'].agg(['count', 'mean']).reset_index()

        # Start plotting
        plt.figure(figsize=(12, 7))
        ax = sns.violinplot(data=df, x='depth', y='loss', hue='component',
                            dodge=True, scale='width', inner=None, palette='Set2')
        sns.stripplot(data=df, x='depth', y='loss', hue='component',
                    dodge=True, jitter=True, alpha=0.3, color='black', ax=ax)

        # Log scale for loss
        plt.yscale('log')
        plt.title("Loss Distribution per Sampled Depth and Component")
        plt.xlabel("Sampled Depth")
        plt.ylabel("Loss (log scale)")

        # Remove duplicated legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Component")

        # Add annotations
        for _, row in group_stats.iterrows():
            depth = row['depth']
            component = row['component']
            mean_loss = row['mean']
            count = int(row['count'])

            offset = -0.2 if component == 0 else 0.2
            x_pos = depth + offset - 1

            ax.text(x_pos, mean_loss, f"{mean_loss:.1f}\n(n={count})",
                    ha='center', va='bottom', fontsize=8, color='blue')

        # Save plot and summary CSV
        plot_path = os.path.join(log_dir, "component_depth_loss_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[MultiCMPNN] Saved depth-loss plot to {plot_path}")

        group_stats.to_csv(os.path.join(log_dir, "depth_component_loss_summary.csv"), index=False)
            

    def on_test_epoch_end(self):
        import matplotlib.pyplot as plt
        import os
        import time
        import pandas as pd

        # Step 1: Gather and inverse-transform
        preds = torch.cat([out["preds"] for out in self.test_outputs], dim=0)
        targets = torch.cat([out["targets"] for out in self.test_outputs], dim=0)
        self.test_outputs.clear()

        device = preds.device
        if self.target_normalizer is not None:
            preds = torch.stack([self.target_normalizer.inverse_transform(p.cpu()).to(device) for p in preds])
            targets = torch.stack([self.target_normalizer.inverse_transform(t.cpu()).to(device) for t in targets])

        output_dim = preds.shape[1] if preds.ndim > 1 else 1
        target_indices = [self.focus_target_idx] if self.focus_target_idx is not None else list(range(output_dim))

        # Step 2: Compute metrics
        metric_results = {}
        for name, metric in self.metrics.items():
            if hasattr(metric, "evaluation"):
                score = metric.evaluation(preds, targets)
            else:
                mask = targets != self.ignore_value
                score = metric(preds[mask], targets[mask]) if mask.any() else torch.tensor(float("nan"), device=device)
            metric_results[name] = score

        # Step 3: Plot histograms
        log_dir = getattr(self.trainer.logger, "log_dir", ".")
        fig, axes = plt.subplots(1, output_dim, figsize=(4 * output_dim, 3))
        axes = axes if output_dim > 1 else [axes]

        for i, ax in zip(range(output_dim), axes):
            mask_i = (targets[:, i] != self.ignore_value) & (preds[:, i] != self.ignore_value)
            if not mask_i.any():
                continue
            residuals = preds[:, i][mask_i] - targets[:, i][mask_i]
            ax.hist(residuals.cpu().numpy(), bins=30, alpha=0.7)
            ax.axvline(0, color="gray", linestyle="--")
            ax.set_title(f"Target {i}")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Count")

        fig.suptitle("Residual Distribution per Target")
        fig.tight_layout()
        plt.savefig(os.path.join(log_dir, "residual_histograms.png"))
        plt.close()

        # Step 4: Log metrics
        for name, val in metric_results.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    self.log(f"test_{subkey.lower()}", subval, prog_bar=True)
            else:
                self.log(f"test_{name.lower()}", val, prog_bar=True)

        # Step 5: Grouped printout
        grouped_results = {}
        for key, val in metric_results.items():
            if "_t" in key:
                base, tidx = key.rsplit("_t", 1)
                grouped_results.setdefault(f"t{tidx}", {})[base] = val
            else:
                grouped_results.setdefault("all", {})[key] = val

        print("\nTest Metrics (Grouped):")
        for tidx in sorted(grouped_results):
            print(f"Target {tidx}:")
            for k, v in grouped_results[tidx].items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        print(f"  - {subk}: {subv:.4f}")
                else:
                    print(f"  - {k}: {v:.4f}")

        # Step 6: Append to CSV
        hparams_dict = dict(self.hparams)
        flat_metrics = {
            k: (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in metric_results.items()
        }

        logger = getattr(self, "logger", None) or self.trainer.logger
        row = {
            **hparams_dict,
            **flat_metrics,
            "logger_name": getattr(logger, "name", None),
            "version":     getattr(logger, "version", None),
            "timestamp":   time.strftime("%Y%m%d_%H%M%S"),
        }

        csv_path = os.path.join(logger.save_dir, "all_runs_results.csv")
        pd.DataFrame([row]).to_csv(
            csv_path,
            mode="a",
            header=not os.path.exists(csv_path),
            index=False,
        )

        return metric_results

    # def configure_optimizers(self):
        # optimizer = self.optimizer_class(
        #     self.parameters(),
        #     lr=self.learning_rate,
        #     weight_decay=self.optimizer_params.get("weight_decay", 0.0),
        #     **{k: v for k, v in self.optimizer_params.items() if k != "weight_decay"}
        # )

        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer,
        #         mode='min',            # because lower loss is better
        #         factor=0.5,             # halve the LR each time
        #         patience=20,            # if no improvement for 50 epochs
        #         min_lr=1e-7,            # don't go below 1e-7
        #         threshold=0.001 ,        # only if val_loss changes by more than 0.001 (0.1%)
        #         threshold_mode='rel',   # relative to the best
        #         verbose=True,
        #     ),
        #     'monitor': 'val_loss',    # what to monitor
        #     'interval': 'epoch',
        #     'frequency': 1
        # }

        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # warmup_steps = 5   # in epochs
        # total_epochs = 100

        # def lr_lambda(current_epoch):
        #     if current_epoch < warmup_steps:
        #         return float(current_epoch + 1) / float(warmup_steps)  # +1 to avoid zero LR at epoch 0
        #     progress = (current_epoch - warmup_steps) / float(max(1, total_epochs - warmup_steps))
        #     return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "epoch",  # <-- correct for epoch-based
        #         "frequency": 1,
        #     }
        # }

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=15,    # Number of epochs before first restart
        #     T_mult=2,  # No growth in cycle length (fixed 50, 50, 50,...)
        #     eta_min=1e-6
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler":{
        #         "scheduler": scheduler,
        #         "interval": "epoch",  # LR scheduler step happens every epoch
        #         "frequency": 1,
        #     },
            
        # }

    # def configure_optimizers(self):
    #     optimizer = self.optimizer_class(
    #         self.parameters(),
    #         lr=self.learning_rate,
    #         weight_decay=self.optimizer_params.get("weight_decay", 0.0),
    #         **{k: v for k, v in self.optimizer_params.items() if k != "weight_decay"}
    #     )

    #     cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer,
    #         T_max=self.trainer.max_epochs,   # tie T_max to total epochs
    #         eta_min=1e-6,
    #     )

    #     plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer,
    #         mode='min',
    #         factor=0.5,
    #         patience=25,
    #         threshold=1e-4,
    #         min_lr=1e-7,
    #         verbose=True,
    #     )

    #     schedulers = [
    #         {
    #             "scheduler": cosine_scheduler,
    #             "interval": "epoch",
    #             "frequency": 1,
    #             "name": "cosine_decay",
    #         },
    #         {
    #             "scheduler": plateau_scheduler,
    #             "interval": "epoch",
    #             "frequency": 1,
    #             "monitor": "val_loss",
    #             "name": "plateau_decay",
    #             "reduce_on_plateau": True,
    #         },
    #     ]

    #     return [optimizer], schedulers


    def configure_optimizers(self):
        import inspect
        OptCls = self.optimizer_class

        # Base kwargs for param groups (we’ll override weight_decay below)
        base_kwargs = {
            "lr": self.learning_rate,
            **{k: v for k, v in self.optimizer_params.items() if k != "weight_decay"}
        }
        wd = self.optimizer_params.get("weight_decay", 0.0)

        # Detect fused/foreach support
        extra_kwargs = {}
        ctor = inspect.signature(OptCls.__init__).parameters
        if OptCls is torch.optim.AdamW and "fused" in ctor:
            extra_kwargs["fused"] = True
        elif OptCls is torch.optim.AdamW and "foreach" in ctor:
            extra_kwargs["foreach"] = True

        # 1) Split into decay vs. no‐decay
        decay_params, no_decay_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # match all biases, all BatchNorm params, plus your two dead‐layer biases
            if (
                name.endswith(".bias")
                or ".bn" in name
                or name == "atom_projections.0.mlp.6.bias"
                or name == "bond_projections.0.mlp.6.bias"
            ):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        # 2) Build optimizer with two groups
        optimizer = OptCls(
            [
                {"params": decay_params,    "weight_decay": wd, "lr": self.learning_rate},
                {"params": no_decay_params, "weight_decay": 0.0, "lr": self.learning_rate},
            ],
            **extra_kwargs
        )

        # 3) LR schedulers: 5‐step linear warmup → cosine restarts
        total_warmup_steps = 5
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=total_warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        seq = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[total_warmup_steps]
        )
        return [optimizer], [{"scheduler": seq, "interval": "step", "name": "warmup_cosine"}]


    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def on_train_epoch_end(self):
        import os

        import matplotlib.pyplot as plt
        if not self.all_theta_pred:
            return  # Skip if no train predictions yet

        theta_pred = torch.cat(self.all_theta_pred).cpu()
        theta_true = torch.cat(self.all_theta_true).cpu()

        delta_deg = theta_pred - theta_true
        delta_deg = (delta_deg + 180.0) % 360.0 - 180.0

        mae = delta_deg.abs().mean()
        self.log("angle_mae_deg", mae.item(), prog_bar=True)
        num_above_45 = (delta_deg.abs() > 45).sum()
        num_above_90 = (delta_deg.abs() > 90).sum()

        print(f"[DEBUG] MAE: {mae:.2f}° | >45° Errors: {num_above_45.item()} | >90° Errors: {num_above_90.item()}")

        if not hasattr(self, "loss_curve"):
            self.loss_curve = []
        self.loss_curve.append(self.trainer.logged_metrics.get("train_loss", torch.tensor(float('nan'))).item())

        if self.current_epoch % 10 == 0:
            log_dir = self.trainer.logger.log_dir if hasattr(self.trainer, 'logger') else "."
            image_dir = os.path.join(log_dir, "images")
            os.makedirs(image_dir, exist_ok=True)

            # Save training scatter
            plt.scatter(theta_true, theta_pred, alpha=0.5)
            plt.xlabel('True Angle (deg)')
            plt.ylabel('Predicted Angle (deg)')
            plt.title('Predicted vs True Angles (Train)')
            plt.plot([-180, 180], [-180, 180], color='red')
            plt.grid(True)
            plt.savefig(os.path.join(image_dir, f"scatter_train_epoch_{self.current_epoch:04d}.png"), dpi=300)
            plt.close()

            # Save training histogram
            plt.hist(delta_deg, bins=60, range=(-180, 180), alpha=0.7)
            plt.xlabel('Prediction Error Δθ (degrees)')
            plt.ylabel('Frequency')
            plt.title('Histogram of Torsion Prediction Errors (Train)')
            plt.grid(True)
            plt.savefig(os.path.join(image_dir, f"histogram_train_epoch_{self.current_epoch:04d}.png"), dpi=300)
            plt.close()

        self.all_theta_pred.clear()
        self.all_theta_true.clear()

        if self.current_epoch % 100 == 0:
            import matplotlib.pyplot as plt
            import os

            # 1) figure out current jiggle (radians)
            jiggle_amt = self.get_current_jiggle(self.current_epoch, self.jiggle_schedule)

            # 2) sample noise for the whole dataset
            n = len(self.theta_true_aug)  # total samples
            noise = torch.randn(n, device=self.device) * jiggle_amt

            # 3) convert to degrees
            degs = noise.cpu().numpy() * 180.0 / np.pi

            # 4) plot & save
            log_dir = getattr(self.trainer.logger, "log_dir", ".")
            image_dir = os.path.join(log_dir, "jiggle_monitor")
            os.makedirs(image_dir, exist_ok=True)

            plt.hist(degs, bins=30, alpha=0.7)
            plt.title(f"Jiggle Distribution at Epoch {self.current_epoch}")
            plt.xlabel("Jiggle Amount (degrees)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(os.path.join(image_dir, f"jiggle_epoch_{self.current_epoch:04d}.png"))
            plt.close()



    def on_validation_epoch_end(self):
        import os

        if not self.all_train_theta_pred or not self.all_val_theta_pred:
            return  # Skip if no data collected yet

        if self.current_epoch % 10 == 0:
            log_dir = self.trainer.logger.log_dir if hasattr(self.trainer, 'logger') else "."
            image_dir = os.path.join(log_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            from collections import defaultdict

            plot_train_val_scatter(
                train_preds=torch.cat(self.all_train_theta_pred).cpu(),
                train_true=torch.cat(self.all_train_theta_true).cpu(),
                val_preds=torch.cat(self.all_val_theta_pred).cpu(),
                val_true=torch.cat(self.all_val_theta_true).cpu(),
                save_dir=image_dir,
                epoch=self.current_epoch
            )
            val_preds = torch.cat(self.all_val_theta_pred).cpu().numpy()
            val_true  = torch.cat(self.all_val_theta_true).cpu().numpy()

            # define your bins exactly as you did before
            bins = np.linspace(-180, 180, 37)

            # compute per‐bin absolute errors
            errors = defaultdict(list)
            for p, t in zip(val_preds, val_true):
                b = np.digitize(t, bins) - 1
                errors[b].append(abs(p - t))

            # print out each bin’s MAE
            for b, errs in sorted(errors.items()):
                lo, hi = bins[b], bins[b+1]
                mae = np.mean(errs)
                count = len(errs)
                print(f"Bin {lo:6.1f}→{hi:6.1f}: MAE={mae:5.2f}°  (n={count})")
        
        if self.current_epoch % 50 == 0:
            from sklearn.decomposition import PCA
            train_embs = np.vstack([e.cpu().numpy() for e in self.all_train_embeddings])
            val_embs   = np.vstack([e.cpu().numpy() for e in self.all_val_embeddings])

            all_embs = np.vstack([train_embs, val_embs])
            labels   = np.array([0]*len(train_embs) + [1]*len(val_embs))

            coords = PCA(n_components=2).fit_transform(all_embs)

            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(coords[labels==0,0], coords[labels==0,1],
                    s=10, alpha=0.3, label="train")
            ax.scatter(coords[labels==1,0], coords[labels==1,1],
                    s=10, alpha=0.3, label="val")
            ax.legend()
            ax.set_title(f"PCA of MPNN embeddings at epoch {self.current_epoch}")

            log_dir   = self.trainer.logger.log_dir
            image_dir = os.path.join(log_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            fig.savefig(os.path.join(image_dir, f"pca_epoch_{self.current_epoch:03d}.png"), dpi=200)
            plt.close(fig)
            self.all_train_embeddings.clear()
            self.all_val_embeddings.clear()
        # Clear the lists to free up memory

        self.all_train_theta_pred.clear()
        self.all_train_theta_true.clear()
        self.all_val_theta_pred.clear()
        self.all_val_theta_true.clear()

    # def compute_angle_weights(self, theta, epoch, warmup=200):
    #     degrees = theta * 180 / torch.pi
    #     is_hard = degrees < 160
    #     ramp = min(1.0, epoch / warmup)
    #     base_weights = torch.ones_like(degrees)
    #     base_weights[is_hard] += ramp * 5.0  # you can tune this multiplier
    #     return base_weights


