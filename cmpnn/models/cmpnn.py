from cmpnn.models.messagepassing import MessagePassing
import torch
import torch.nn as nn
from cmpnn.models.utils import get_activation_fn, index_select_ND
from cmpnn.models.gru import BatchGRU


class CMPNNEncoder(MessagePassing):
    """
    CMPNNEncoder is a message passing neural network encoder for molecular property prediction.
    It uses a message passing mechanism to learn representations of molecular graphs.
    """

    def __init__(self, atom_fdim: int, bond_fdim: int, atom_messages: bool, depth: int = 3,
                 dropout: float = 0.1, hidden_dim: int = 128, use_batch_norm: bool = True,
                 activation: str = 'relu', bias: bool = True, booster: str = 'sum', comm_mode: str = 'add'):
        """
        """
        assert comm_mode in ['add', 'mlp', 'gru',
                             'ip'], f"Invalid comm_mode: {comm_mode}. Must be one of ['add', 'mlp', 'gru', 'ip']"
        assert booster in ['sum', 'sum_max', 'attention',
                           'mean'], f"Invalid booster: {booster}. Must be one of ['sum', 'sum_max', 'attention', 'mean']"

        self.booster = booster
        self.comm_mode = comm_mode
        super().__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = atom_messages
        self.depth = depth
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act_func = get_activation_fn(activation)

        # Input layers
        self.W_i_atom = nn.Linear(atom_fdim, hidden_dim, bias=bias)
        self.W_i_bond = nn.Linear(bond_fdim, hidden_dim, bias=bias)

        # Additional Layers for aggregation and readout
        self.W_o = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.lr = nn.Linear(hidden_dim * 3, hidden_dim, bias=bias)
        self.gru = BatchGRU(hidden_dim)

        # Communication Mode
        if comm_mode == 'mlp':
            self.atom_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                get_activation_fn(activation)
            )
        elif comm_mode == 'gru':
            self.atom_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Create stack of message passing layers
        self.W_h = nn.ModuleList([nn.Linear(hidden_dim, self.hidden_dim, bias=bias)
                                  for _ in range(depth - 1)])

    def initialize(self, f_atoms: torch.Tensor, f_bonds: torch.Tensor):
        """
        Initialize the atom and bond representations.
        """
        input_atom = self.act_func(self.W_i_atom(f_atoms))
        message_atom = input_atom.clone()

        input_bond = self.act_func(self.W_i_bond(f_bonds))
        message_bond = input_bond.clone()
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
            scores = torch.softmax(torch.sum(agg, dim=2), dim=1).unsqueeze(-1)  # (n_atoms, max_bonds, 1)
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
        updated = self.W_h[depth_index](updated)
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

    def forward(self, f_atoms: torch.Tensor, f_bonds: torch.Tensor,
                a2b: torch.Tensor, b2a: torch.Tensor, b2revb: torch.Tensor,
                a_scope: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        """
        # Initialize atom and bond messages
        input_atom, message_atom, input_bond, message_bond = self.initialize(f_atoms, f_bonds)

        # Message passing
        for depth in range(self.depth - 1):
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
