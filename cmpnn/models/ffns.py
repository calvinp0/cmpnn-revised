from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn

from cmpnn.models.utils import get_activation_fn  # adjust import as needed


class FFN(nn.Module, ABC):
    """
    An abstract base class for a feed-forward network.
    Defines the interface for mapping input features to output predictions.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        pass


class MLP(FFN):
    """
    A multilayer perceptron (MLP) implementing a feed-forward network.
    This network consists of an input layer, configurable hidden layers, and an output layer.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 300,
                 n_layers: int = 1,
                 dropout: float = 0.0,
                 activation: str = 'relu'):
        """
        Initializes the MLP.
        
        Args:
            input_dim: Dimension of the input features.
            output_dim: Dimension of the output predictions.
            hidden_dim: Size of the hidden layers.
            n_layers: Number of hidden layers (not counting the output layer).
            dropout: Dropout probability.
            activation: Activation function name.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        act = get_activation_fn(activation)
        layers = []

        # First layer: input to hidden.
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Additional hidden layers.
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer.
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        return self.mlp(x)

    def __repr__(self):
        return (f"MLP(input_dim={self.input_dim}, output_dim={self.output_dim}, "
                f"hidden_dim={self.hidden_dim}, n_layers={self.n_layers}, dropout={self.dropout})")

    @classmethod
    def build(cls,
              input_dim: int,
              output_dim: int,
              hidden_dim: int = 300,
              n_layers: int = 1,
              dropout: float = 0.0,
              activation: str = 'relu') -> 'MLP':
        """
        Build an MLP model with the specified parameters.
        """
        return cls(input_dim=input_dim,
                   output_dim=output_dim,
                   hidden_dim=hidden_dim,
                   n_layers=n_layers,
                   dropout=dropout,
                   activation=activation)

class ScaledOutputLayer(nn.Module):
    def __init__(self, base_layer, output_size, scale_init=0.1, clamp_min=1e-4, clamp_max=10.0):
        """
        Initializes the ScaledOutput layer.
        
        Args:
            base_layer: The base layer (e.g., MLP) to which the scaling is applied.
            output_size: The size of the output.
            scale_init: Initial value for the scaling parameter.
        """
        super().__init__()
        self.base = base_layer
        self.log_scale = nn.Parameter(torch.ones(output_size) * torch.log(torch.tensor(scale_init)))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Forward pass through the ScaledOutput layer.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after applying the base layer and scaling.
        """
        raw_out = self.base(x)
        scale = torch.exp(self.log_scale)
        scale = torch.clamp(scale, self.clamp_min, self.clamp_max)
        return raw_out * scale
    
    def __repr__(self):
        return (f"ScaledOutputLayer(base={self.base}, "
                f"log_scale={self.log_scale}, clamp_min={self.clamp_min}, clamp_max={self.clamp_max})")
    def __str__(self):
        return (f"ScaledOutputLayer(base={self.base}, "
                f"log_scale={self.log_scale}, clamp_min={self.clamp_min}, clamp_max={self.clamp_max})")
    
    @classmethod
    def build(cls,
              base_layer: nn.Module,
              output_size: int,
              scale_init: float = 0.1,
              clamp_min: float = 1e-4,
              clamp_max: float = 10.0) -> 'ScaledOutputLayer':
        """
        Build a ScaledOutput layer with the specified parameters.
        """
        return cls(base_layer=base_layer,
                   output_size=output_size,
                   scale_init=scale_init,
                   clamp_min=clamp_min,
                   clamp_max=clamp_max)
    def get_parameters(self):
        """
        Returns the parameters of the ScaledOutput layer.
        """
        return {
            'log_scale': self.log_scale,
            'clamp_min': self.clamp_min,
            'clamp_max': self.clamp_max
        }

class HybridRegressionHead(FFN):
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 3,
                 hidden_dim: int = 300,
                 n_layers: int = 1,
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 freeze_logvar: bool = False,
                 init_value: float = -2.0):
        super().__init__()
        self.num_outputs = output_dim
        self.mu = MLP(input_dim, self.num_outputs, hidden_dim, n_layers=n_layers, dropout=dropout, activation=activation)

        self.freeze_logvar = freeze_logvar

        if freeze_logvar:
            self.freeze_logvar = True
            self.register_buffer("frozen_logvar", torch.full((output_dim,), init_value))  # maybe -2.0
        else:
            self.freeze_logvar = False
            self.logvar = MLP(input_dim, self.num_outputs, hidden_dim, n_layers=n_layers, dropout=dropout, activation=activation)


    def forward(self, x):
        mu = self.mu(x)
        
        if self.freeze_logvar:
            logvar = self.frozen_logvar.expand(x.size(0), -1)
        else:
            logvar = self.logvar(x)
        
        return mu, logvar


    def __repr__(self):
        if self.freeze_logvar:
            return f"HybridRegressionHead(mu={self.mu}, logvar=<frozen>)"
        else:
            return f"HybridRegressionHead(mu={self.mu}, logvar={self.logvar})"


class LogGaussianHead(FFN):
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 3,
                 hidden_dim: int = 300,
                 n_layers: int = 1,
                 dropout: float = 0.0,
                 activation: str = 'relu'):
        super().__init__()
        self.output_dim = output_dim
        self.mu = MLP(input_dim, self.output_dim, hidden_dim, n_layers=n_layers, dropout=dropout, activation=activation)
        self.logvar = MLP(input_dim, self.output_dim, hidden_dim, n_layers=n_layers, dropout=dropout, activation=activation)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def __repr__(self):
        return f"LogGaussianHead(mu={self.mu}, logvar={self.logvar})"
    
    def __str__(self):
        return f"LogGaussianHead(mu={self.mu}, logvar={self.logvar})"
    
    def get_parameters(self):
        """
        Returns the parameters of the LogGaussianHead.
        """
        return {
            'mu': self.mu,
            'logvar': self.logvar
        }


class PeriodicHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = 'relu',
        use_residual: bool = False,
    ):
        """
        PeriodicHead with optional global residual connection.

        Args:
            input_dim:    dimensionality of input features
            output_dim:   number of periodic outputs (2 for sin/cos)
            hidden_dim:   width of hidden layers
            n_layers:     number of hidden layers in the MLP
            dropout:      dropout probability
            activation:   activation name (e.g. 'relu')
            use_residual: if True, adds a linear skip from input -> output_dim
        """
        super().__init__()
        # core MLP
        self.net = MLP(input_dim, output_dim, hidden_dim, n_layers, dropout, activation)
        self.use_residual = use_residual
        if use_residual:
            # project input to match sin/cos output shape
            self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # main branch
        out = self.net(x)
        # add optional skip connection
        if self.use_residual:
            res = self.skip(x)
            out = out + res
        # normalize to unit circle
        return out / (out.norm(dim=-1, keepdim=True) + 1e-8)

    def __repr__(self):
        return f"PeriodicHead(net={self.net}, use_residual={self.use_residual})"

    def get_parameters(self):
        params = {'net': self.net}
        if self.use_residual:
            params['skip'] = self.skip
        return params


class CompositeHeads(nn.Module):

    def __init__(self, heads: list):
        """
        Initializes the CompositeHeads with a list of heads.
        
        Args:
            heads: List of FFN heads.
        """
        super().__init__()
        self.heads = nn.ModuleList(heads)

class AngularHuberHead(FFN):
    def __init__(self, input_dim: int, output_dim = 1,
                                  hidden_dim: int = 300,
                 n_layers: int = 1,
                 dropout: float = 0.0,
                 activation: str = 'relu', eps_deg: float = 5):
        super().__init__()
        self.net = MLP(input_dim, 1, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout, activation=activation)
        self.output_dim = output_dim
        self.eps  = math.radians(eps_deg)

    def forward(self, x):
        return self.net(x)[:, 0]
    

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, activation='relu'):
        super().__init__()
        act = get_activation_fn(activation)
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = act
        self.drop1 = nn.Dropout(dropout)

        self.lin2 = nn.Linear(hidden_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.lin1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.lin2(out)
        out = self.norm2(out)
        out = self.drop2(out)

        return out + residual


class DeepPeriodicHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 300,
        n_blocks: int = 3,
        dropout: float = 0.0,
        activation: str = 'relu'
    ):
        """
        Deep residual-style periodic head:
          - project input → 2-dim (sin/cos)
          - pass through several residual MLP blocks
          - normalize to unit circle
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_blocks = n_blocks
        # initial projection to 2D
        self.first = nn.Linear(input_dim, 2)
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(2, hidden_dim, dropout, activation)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # project to 2 dims
        out = self.first(x)
        # apply a sequence of small residual blocks
        for blk in self.blocks:
            out = blk(out)
        # final normalization onto unit circle
        return out / (out.norm(dim=-1, keepdim=True) + 1e-8)

    def __repr__(self):
        return (
            f"DeepPeriodicHead(input_dim={self.input_dim}, "
            f"n_blocks={self.n_blocks})"
        )
