import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def get_activation_fn(activation: str) -> nn.Module:
    """
    Returns an instance of the activation function based on the provided string.
    """
    if activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.lower() == 'tanh':
        return nn.Tanh()
    elif activation.lower() == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation.lower() == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target[index == 0] = 0
    return target


def initialize_weights(model: nn.Module):
    """
    Initializes the weights of a model in place.

    :param model: An nn.Module.
    """
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # Skip bias terms and single-value parameters like gamma
        if 'weight' in name:
            nn.init.xavier_normal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)


def plot_train_val_scatter(train_preds, train_true, val_preds, val_true, save_dir, epoch):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Training set
    axes[0].scatter(train_true, train_preds, alpha=0.6)
    axes[0].plot([-180, 180], [-180, 180], color='red')
    axes[0].set_xlabel("True Angle (deg)")
    axes[0].set_ylabel("Predicted Angle (deg)")
    axes[0].set_title("Train Scatter")

    # Validation set
    axes[1].scatter(val_true, val_preds, alpha=0.6, color='orange')
    axes[1].plot([-180, 180], [-180, 180], color='red')
    axes[1].set_xlabel("True Angle (deg)")
    axes[1].set_ylabel("Predicted Angle (deg)")
    axes[1].set_title("Validation Scatter")

    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"scatter_train_val_epoch_{epoch:04d}.png"), dpi=300)
    plt.close()

def apply_small_coordinate_noise(molecule, std=0.02):
    """
    Applies a small random noise to a molecule's atomic coordinates.

    Args:
        molecule: A molecule object with .coords attribute (Nx3 tensor).
        std: Standard deviation of the Gaussian noise in Ångstroms.
    """
    if hasattr(molecule, "coords"):  # assuming your Molecule class has .coords
        noise = torch.randn_like(molecule.coords) * std
        molecule.coords += noise


def compute_angle_difference(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Returns angular difference in degrees, shape (N,)
    Assumes inputs are (N,2): sin/cos format
    """
    pred_angle = torch.atan2(preds[:, 0], preds[:, 1])  # OK if preds[:, 0] = sin, preds[:, 1] = cos
    true_angle = torch.atan2(targets[:, 0], targets[:, 1])
    diff = (pred_angle - true_angle + torch.pi) % (2 * torch.pi) - torch.pi
    return diff.abs() * (180.0 / torch.pi)