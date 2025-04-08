import torch
import torch.nn as nn
from abc import ABC, abstractmethod
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
