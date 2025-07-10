import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from scipy.stats import truncnorm
from cmpnn.models.gru import BatchGRU
from cmpnn.models.messagepassing import MessagePassing
from cmpnn.models.utils import get_activation_fn, index_select_ND


class CMPNNEncoder(MessagePassing):
    """
    CMPNNEncoder is a message passing neural network encoder for molecular property prediction.
    It uses a message passing mechanism to learn representations of molecular graphs.
    """

    def __init__(self, atom_fdim: int, bond_fdim: int, atom_messages: bool, depth: int = 3,
                 dropout: float = 0.1, hidden_dim: int = 128, use_batch_norm: bool = True,
                 activation: str = 'relu', bias: bool = True, booster: str = 'sum', comm_mode: str = 'add', dynamic_depth: str = None):
        """
        """
        assert comm_mode in ['add', 'mlp', 'gru',
                             'ip'], f"Invalid comm_mode: {comm_mode}. Must be one of ['add', 'mlp', 'gru', 'ip']"
        assert booster in ['sum', 'sum_max', 'attention',
                           'mean'], f"Invalid booster: {booster}. Must be one of ['sum', 'sum_max', 'attention', 'mean']"
        assert depth > 0, f"Depth must be greater than 0, got {depth}"

        assert dynamic_depth in [None, 'uniform', 'truncnorm'], f"Invalid dynamic_depth: {dynamic_depth}. Must be one of [None, 'uniform', 'truncnorm']"
        self.booster = booster
        self.comm_mode = comm_mode
        self.dynamic_depth = dynamic_depth
        super().__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = atom_messages
        self.depth = depth
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.sampled_depths = []
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act_func = get_activation_fn(activation)

        # Input layers
        self.W_i_atom = nn.Linear(atom_fdim, hidden_dim, bias=bias).float()
        w = self.W_i_atom.weight
        if torch.isnan(w).any() or torch.isinf(w).any():
            print("💀 W_i_atom.weight is already NaN/Inf")
        self.W_i_bond = nn.Linear(bond_fdim, hidden_dim, bias=bias).float()

        # Additional Layers for aggregation and readout
        self.W_o = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.lr = nn.Linear(hidden_dim * 3, hidden_dim, bias=bias)
        self.gru = BatchGRU(hidden_dim)
        self.bn_atom = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.bn_bond = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()

        # Communication Mode
        if comm_mode == 'mlp':
            self.atom_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                get_activation_fn(activation)
            )
        elif comm_mode == 'gru':
            self.atom_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Create stack of message passing layers
        max_depth = depth + 3 if dynamic_depth else depth
        self.W_h_bn = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
            for _ in range(max_depth - 1)
        ])


    def initialize(self, f_atoms: torch.Tensor, f_bonds: torch.Tensor):
        """
        Initialize the atom and bond representations.
        """
        if torch.isnan(f_atoms).any():
            print("f_atoms has NaN BEFORE W_i_atom")

        if torch.isinf(f_atoms).any():
            print("f_atoms has Inf BEFORE W_i_atom")

        max_val = f_atoms.abs().max()
        if max_val > 1e3:
            print("⚠️ f_atoms max abs value:", max_val)
        with torch.amp.autocast_mode(dtype=torch.float32):

            input_atom = self.act_func(self.W_i_atom(f_atoms))
            input_atom = self.bn_atom(input_atom)
            assert input_atom.shape[-1] == self.hidden_dim, f"Input atom hidden_dim mismatch: {input_atom.shape[-1]} vs {self.hidden_dim}"
            assert input_atom.dtype == torch.float32

            message_atom = input_atom.clone()

            input_bond = self.act_func(self.W_i_bond(f_bonds))
            input_bond = self.bn_bond(input_bond)
            assert input_bond.shape[-1] == self.hidden_dim, f"Input bond hidden_dim mismatch: {input_bond.shape[-1]} vs {self.hidden_dim}"
            assert input_bond.dtype == torch.float32

            message_bond = input_bond.clone()

        if torch.isnan(input_atom).any():
            print("input_atom has NaN AFTER W_i_atom")
        return input_atom, message_atom, input_bond, message_bond

    def message(self, message_bond: torch.Tensor, a2b: torch.Tensor) -> torch.Tensor:
        agg = index_select_ND(message_bond, a2b)  # [num_atoms, max_bonds, hidden]

        if self.booster == 'sum':
            return agg.sum(dim=1)
        elif self.booster == 'mean':
            return agg.mean(dim=1)
        elif self.booster == 'sum_max':
            return agg.sum(dim=1) * agg.max(dim=1)[0]
        elif self.booster == 'attention':
            # scale attention scores by sqrt(hidden_dim) for stability
            scores = torch.softmax(torch.sum(agg, dim=2) / (self.hidden_dim ** 0.5), dim=1).unsqueeze(-1)  # (n_atoms, max_bonds, 1)
            return torch.sum(agg * scores, dim=1)  # Weighted sum
        else:
            raise ValueError(f"Unsupported booster: {self.booster}")

    def update(self, message_atom: torch.Tensor, message_bond: torch.Tensor,
               input_bond: torch.Tensor, b2a: torch.Tensor, b2revb: torch.Tensor,
               depth_index: int) -> torch.Tensor:
        """
        Update bond messages based on the current atom messages.
        """
        rev_message = message_bond[b2revb]
        updated = message_atom[b2a] - rev_message
        updated = self.W_h_bn[depth_index](updated)

        updated = self.dropout_layer(self.act_func(input_bond + updated))
        return updated

    def communicate(self, message_atom: torch.Tensor, agg: torch.Tensor):
        if self.comm_mode == 'mlp':
            fused = torch.cat([message_atom, agg], dim=1)
            return self.atom_mlp(fused)
        elif self.comm_mode == 'gru':
            return self.atom_gru(agg, message_atom)
        elif self.comm_mode == 'ip':
            return message_atom * agg
        else:
            return message_atom + agg

    def final_communicate(self, agg_message, message_atom, input_atom, a_scope):
        """
        Final communication step before readout.
        """
        combined = torch.cat([agg_message, message_atom, input_atom], dim=1)
        updated = self.lr(combined)
        return self.gru(updated, a_scope)


    def dynamic_passing(self):
        """
        Dynamically sample the depth for message passing based on the specified method.
        """
        if self.dynamic_depth == 'uniform':
            min_depth = max(1, self.depth - 2)
            max_depth = self.depth + 2
            ndepth = torch.randint(min_depth, max_depth + 1, (1,)).item()
            self.sampled_depths.append(ndepth)
            return ndepth

        elif self.dynamic_depth == 'truncnorm':
            lower, upper = max(1, self.depth - 3), self.depth + 3
            mu, sigma = self.depth, 1
            X = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            ndepth = int(X.rvs(1))
            # clamp sampled depth to valid [lower, upper]
            ndepth = max(lower, min(upper, ndepth))
            self.sampled_depths.append(ndepth)
            return ndepth

        else:
            return self.depth


    def forward(self, f_atoms: torch.Tensor, f_bonds: torch.Tensor,
                a2b: torch.Tensor, b2a: torch.Tensor, b2revb: torch.Tensor,
                a_scope: torch.Tensor, mask = None) -> torch.Tensor:
        """
        Forward pass through the encoder.
        """
        # Initialize atom and bond messages
        input_atom, message_atom, input_bond, message_bond = self.initialize(f_atoms, f_bonds)

        if mask is not None:
            input_atom = input_atom.clone()
            input_atom[mask] = 0.0
            print("input_atom[mask] mean:", input_atom[mask].mean().item())
            print("input_atom[mask] std:", input_atom[mask].std().item())

        # Dynamic depth adjustment
        if self.training and self.dynamic_depth is not None:
            ndepth = self.dynamic_passing()
        else:
            ndepth = self.depth

        # Message passing
        for depth in range(ndepth - 1):
            agg = self.message(message_bond, a2b)
            message_atom = self.communicate(message_atom, agg)
            message_bond = self.update(message_atom, message_bond, input_bond, b2a, b2revb, depth)

        # Final Communication
        agg_message = self.message(message_bond, a2b)
        agg_message = self.final_communicate(agg_message, message_atom, input_atom, a_scope)

        atom_hiddens = self.act_func(self.W_o(agg_message))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        

        return atom_hiddens

    def __repr__(self):
        base_repr = super().__repr__().splitlines()
        header = f"{self.__class__.__name__}(\n  comm_mode={self.comm_mode}, booster={self.booster}"
        indented_submodules = ["  " + line for line in base_repr[1:]]
        return "\n".join([header, *indented_submodules])

    def save_sampled_depths(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.sampled_depths, f)
        print(f"Sampled depths saved to {path}")
