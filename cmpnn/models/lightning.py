import time
from pyparsing import Callable
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmpnn.models.cmpnn import CMPNNEncoder
from cmpnn.models.ffns import MLP
from cmpnn.optimizer.noam import NoamLikeOptimizer
from cmpnn.models.utils import initialize_weights
from cmpnn.models.aggregators import SumAggregator, MeanAggregator, NormMeanAggregator
import torchmetrics


class CMPNNLightningModule(pl.LightningModule):
    """
    A PyTorch Lightning module for the CMPNN model.
    """

    def __init__(
        self,
        atom_fdim: int = 133,
        bond_fdim: int = 14,
        global_fdim: int = 0,
        atom_messages: bool = True,
        depth: int = 3,
        output_size: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = False,
        booster: str = "sum",
        comm_mode: str = "add",
        prediction_type: str = "regression",
        aggregator: str = "mean",
        use_batch_norm: bool = False,
        ffn_config: dict = None,
        optimizer_class: type = torch.optim.Adam,
        optimizer_params: dict = None,
        learning_rate: float = 1e-3,
        metrics: dict = None,
        plot_lr: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.output_size = output_size
        self.prediction_type = prediction_type
        self.plot_lr = plot_lr
        if self.prediction_type == "classification":
            self.sigmoid = nn.Sigmoid()

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
            use_batch_norm=use_batch_norm,
        )

        # Set the aggregator
        if aggregator == "mean":
            self.aggregator = MeanAggregator()
        elif aggregator == "sum":
            self.aggregator = SumAggregator()
        elif aggregator == "norm_mean":
            self.aggregator = NormMeanAggregator()
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        self.bn = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()

        # Determine the input dimensions for the FFN
        ffn_input_dim = hidden_dim + global_fdim
        ffn_config = ffn_config if ffn_config else {}
        self.ffn = MLP(
            input_dim=ffn_input_dim,
            output_dim=output_size,
            hidden_dim=ffn_config.get("hidden_dim", hidden_dim),
            n_layers=ffn_config.get("n_layers", 1),
            dropout=ffn_config.get("dropout", dropout),
            activation=ffn_config.get("activation", activation),
        )

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params if optimizer_params else {}
        self.learning_rate = learning_rate

        # Initialize weights
        initialize_weights(self)

        if metrics is None:
            self.metrics = nn.ModuleDict(
                {
                    "RMSE": torchmetrics.MeanSquaredError(squared=False),
                    "MAE": torchmetrics.MeanAbsoluteError(),
                    "R2": torchmetrics.R2Score(),
                }
            )
        else:
            self.metrics = nn.ModuleDict(metrics)

    def forward(
        self,
        f_atoms: torch.Tensor,
        f_bonds: torch.Tensor,
        a2b: torch.Tensor,
        b2a: torch.Tensor,
        b2revb: torch.Tensor,
        a_scope: torch.Tensor,
        global_features: torch.Tensor = None,
    ) -> torch.Tensor:
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
            a_scope=a_scope,
        )

        # Aggregate atom features to molecule features
        mol_vectors = self.aggregator(atom_hidden, a_scope)

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
        targets = batch.y
        global_features = (
            batch.global_features if hasattr(batch, "global_features") else None
        )

        # Forward pass
        output = self(
            atom_features, bond_features, a2b, b2a, b2revb, a_scope, global_features
        )
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # Compute loss
        loss = F.mse_loss(output, targets)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, batch_size=targets.size(0))

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
        # Extract features from the batch
        atom_features = batch.f_atoms
        bond_features = batch.f_bonds
        a2b = batch.a2b
        b2a = batch.b2a
        b2revb = batch.b2revb
        a_scope = batch.a_scope
        targets = batch.y
        global_features = (
            batch.global_features if hasattr(batch, "global_features") else None
        )
        # Forward pass
        output = self(
            atom_features, bond_features, a2b, b2a, b2revb, a_scope, global_features
        )

        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # Compute loss
        loss = self.loss_fn(output, targets)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, batch_size=targets.size(0))

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
        targets = batch.y
        global_features = (
            batch.global_features if hasattr(batch, "global_features") else None
        )

        # Forward pass
        output = self(
            atom_features, bond_features, a2b, b2a, b2revb, a_scope, global_features
        )

        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        self.test_outputs.append(
            {"preds": output.detach(), "targets": targets.detach()}
        )
        return {}

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), lr=self.learning_rate, **self.optimizer_params
        )
        self.optimizer_instance = optimizer  # Save it for use in on_train_end
        return optimizer

    def on_train_end(self):
        if not self.plot_lr:
            return
        optimizer = self.trainer.optimizers[0]
        if isinstance(optimizer, NoamLikeOptimizer):
            import os

            log_dir = self.trainer.logger.log_dir if self.trainer.logger else "."
            path = os.path.join(log_dir, "lr_schedule.png")
            optimizer.plot_lr_schedule(save_path=path)

    def on_test_epoch_end(self):
        # Concatenate predictions and targets from all batches.
        preds = torch.cat([out["preds"] for out in self.test_outputs], dim=0)
        targets = torch.cat([out["targets"] for out in self.test_outputs], dim=0)
        # Compute and log all specified metrics.
        metric_results = {}
        for name, metric in self.metrics.items():
            # Need the metric to be on the same device as the predictions.
            metric_results[name] = metric(preds, targets)
        self.log_dict(metric_results, prog_bar=True)
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
        if self.prediction_type == "classification":
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

    def __init__(
        self,
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
        activation: str = "relu",
        bias: bool = False,
        booster: str = "sum",
        comm_mode: str = "add",
        prediction_type: str = "regression",
        aggregator: str = "mean",
        use_batch_norm: bool = False,
        ffn_config: dict = None,
        optimizer_class: type = torch.optim.Adam,
        optimizer_params: dict = None,
        learning_rate: float = 1e-3,
        metrics: dict = None,
        plot_lr: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.output_size = output_size
        self.prediction_type = prediction_type
        self.plot_lr = plot_lr
        if self.prediction_type == "classification":
            self.sigmoid = nn.Sigmoid()
        self.shared_encoder = shared_encoder

        if shared_encoder:
            self.encoders = nn.ModuleList(
                [
                    CMPNNEncoder(
                        atom_fdim,
                        bond_fdim,
                        atom_messages=atom_messages,
                        depth=depth,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        activation=activation,
                        bias=bias,
                        booster=booster,
                        comm_mode=comm_mode,
                    )
                ]
                * n_components
            )
        else:
            self.encoders = nn.ModuleList(
                [
                    CMPNNEncoder(
                        atom_fdim,
                        bond_fdim,
                        atom_messages=atom_messages,
                        depth=depth,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        activation=activation,
                        bias=bias,
                        booster=booster,
                        comm_mode=comm_mode,
                    )
                    for _ in range(n_components)
                ]
            )

        # Set the aggregator
        if aggregator == "mean":
            self.aggregator = MeanAggregator()
        elif aggregator == "sum":
            self.aggregator = SumAggregator()
        elif aggregator == "norm_mean":
            self.aggregator = NormMeanAggregator()
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")
        self.bn = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()

        # Determine the input dimensions for the FFN
        ffn_input_dim = (hidden_dim + global_fdim) * n_components
        ffn_config = ffn_config if ffn_config else {}

        self.ffn = MLP(
            input_dim=ffn_input_dim,
            output_dim=output_size,
            hidden_dim=ffn_config.get("hidden_dim", hidden_dim),
            n_layers=ffn_config.get("n_layers", 1),
            dropout=ffn_config.get("dropout", dropout),
            activation=ffn_config.get("activation", activation),
        )

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params if optimizer_params else {}
        self.learning_rate = learning_rate

        # Initialize weights
        initialize_weights(self)

        if metrics is None:
            self.metrics = nn.ModuleDict(
                {
                    "RMSE": torchmetrics.MeanSquaredError(squared=False),
                    "MAE": torchmetrics.MeanAbsoluteError(),
                    "R2": torchmetrics.R2Score(),
                }
            )
        else:
            self.metrics = nn.ModuleDict(metrics)

    def forward(self, batch) -> torch.Tensor:
        component_outputs = []

        for i, encoder in enumerate(self.encoders):
            comp = batch.components[i]
            atom_hidden = encoder(
                f_atoms=comp.f_atoms,
                f_bonds=comp.f_bonds,
                a2b=comp.a2b,
                b2a=comp.b2a,
                b2revb=comp.b2revb,
                a_scope=comp.a_scope,
            )

            mol_vector = self.aggregator(atom_hidden, comp.a_scope)
            mol_vector = self.bn(mol_vector)

            if comp.global_features is not None:
                mol_vector = torch.cat([mol_vector, comp.global_features], dim=-1)

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
        loss = F.mse_loss(output, targets)
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, batch_size=targets.size(0))
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
        loss = F.mse_loss(output, targets)
        # Log metrics
        self.log("val_loss", loss, prog_bar=True, batch_size=targets.size(0))
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
        self.test_outputs.append(
            {"preds": output.detach(), "targets": targets.detach()}
        )
        return {}

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), lr=self.learning_rate, **self.optimizer_params
        )
        self.optimizer_instance = optimizer
        return optimizer

    def on_train_end(self):
        if not self.plot_lr:
            return
        optimizer = self.trainer.optimizers[0]
        if isinstance(optimizer, NoamLikeOptimizer):
            import os

            log_dir = self.trainer.logger.log_dir if self.trainer.logger else "."
            path = os.path.join(log_dir, "lr_schedule.png")
            optimizer.plot_lr_schedule(save_path=path)

    def on_test_epoch_end(self):
        # Concatenate predictions and targets from all batches.
        preds = torch.cat([out["preds"] for out in self.test_outputs], dim=0)
        targets = torch.cat([out["targets"] for out in self.test_outputs], dim=0)
        # Compute and log all specified metrics.
        metric_results = {}
        for name, metric in self.metrics.items():
            # Need the metric to be on the same device as the predictions.
            metric_results[name] = metric(preds, targets)
        self.log_dict(metric_results, prog_bar=True)
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
        if self.prediction_type == "classification":
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


class ArrheniusMultiCMPNNLightningModule(MultiCMPNNLightningModule):
    r"""Multi-component CMPNN with Arrhenius post-processing.

    This module predicts Arrhenius parameters ``[A, n, Ea]`` for each sample and
    supports optional unscaling of both predictions and targets before the loss
    is computed.  The final loss is a sum of the parameter mean-squared error and
    the mean-squared error of the derived :math:`\ln k(T)` values computed via an
    :class:`~cmpnn.models.arrhenius.ArrheniusLayer`.
    """

    def __init__(
        self,
        *args,
        temps: list[float],
        target_unscalers: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
        lnk_mean: torch.Tensor | None = None,
        lnk_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.arrhenius = MLP(
            input_dim=1,
            output_dim=1,
            hidden_dim=64,
            n_layers=2,
            dropout=0.1,
            activation="relu",
        )

    def _encode_one(self, comp) -> torch.Tensor:
        if self.shared_encoder:
            atom_hidden = self.encoder(
                f_atoms=comp.f_atoms,
                f_bonds=comp.f_bonds,
                a2b=comp.a2b,
                b2a=comp.b2a,
                b2revb=comp.b2revb,
                a_scope=comp.a_scope,
            )
        else:
            raise RunTimeError("Use _encode_pair for non-shared case")

        mol_vector = self.aggregator(atom_hidden, comp.a_scope)
        if comp.global_features is not None:
            mol_vector = torch.cat([mol_vector, comp.global_features], dim=-1)
        return mol_vector

    def _encode_pair(self, batch) -> torch.Tensor:
        comps = batch.components
        if self.shared_encoder:
            h_d = self._encode_one(comps[0])
            h_a = self._encode_one(comps[1])
        else:
            h_d = self.encoders[0](
                f_atoms=comps[0].f_atoms,
                f_bonds=comps[0].f_bonds,
                a2b=comps[0].a2b,
                b2a=comps[0].b2a,
                b2revb=comps[0].b2revb,
                a_scope=comps[0].a_scope,
            )
            h_d = self.aggregator(h_d, comps[0].a_scope)
            if comps[0].global_features is not None:
                h_d = torch.cat([h_d, comps[0].global_features], dim=-1)

            h_a = self.encoders[1](
                f_atoms=comps[1].f_atoms,
                f_bonds=comps[1].f_bonds,
                a2b=comps[1].a2b,
                b2a=comps[1].b2a,
                b2revb=comps[1].b2revb,
                a_scope=comps[1].a_scope,
            )
            h_a = self.aggregator(h_a, comps[1].a_scope)
            if comps[1].global_features is not None:
                h_a = torch.cat([h_a, comps[1].global_features], dim=-1)

        # Build both orders
        z_fwd = torch.cat([h_d, h_a], dim=-1)  # donor | acceptor
        z_rev = torch.cat([h_a, h_d], dim=-1)  # acceptor | donor

        z_fwd = self.bn(z_fwd)
        z_rev = self.bn(z_rev)

        return z_fwd, z_rev

    def _unscale(self, tensor: torch.Tensors) -> torch.Tensor:
        """Unscale the tensor using the target unscalers."""
        if not self.target_unscalers:
            return tensor

        cols = []
        for i in range(tensor.size(1)):
            fn = self.target_unscalers[i] if i < len(self.target_unscalers) else None
            if fn is None:
                cols.append(tensor[:, i : i + 1])
            else:
                cols.append(fn(tensor[:, i : i + 1]))
        return torch.cat(cols, dim=1)

    def forward(self, batch):
        z_fwd, z_rev = self._encode_pair(batch)
        y_fwd = self.head_fwd(z_fwd)
        y_rev = self.head_rev(z_rev)
        return {"y_fwd": y_fwd, "y_rev": y_rev}
