from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class MessagePassing(ABC, nn.Module):
    """
    Abstract base class for message passing layers.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def initialize(self, atom_features: torch.Tensor, edge_attr: torch.Tensor,
                   edge_index: torch.Tensor) -> torch.Tensor:
        """Compute initial hidden states for edges from atom and bond features."""
        pass

    @abstractmethod
    def message(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Aggregate messages from neighboring edges for each edge."""
        pass

    @abstractmethod
    def update(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Update edge hidden states using aggregated messages and a residual connection."""
        pass
