import torch
from torch import nn
from typing import Iterable, Callable, List


class ArrheniusLayer(nn.Module):
    r"""Compute :math:`\ln k(T)` from Arrhenius parameters.

    Parameters
    ----------
    temps:
        Iterable of temperatures in Kelvin at which the rate constant should be
        evaluated.
    use_kJ:
        Whether activation energies are provided in kJ/mol.  If ``False`` the
        values are assumed to be in J/mol.
    lnk_mean / lnk_scale:
        Optional mean and standard deviation used to (de-)standardise
        :math:`\ln k(T)`.  These are typically obtained from a
        ``StandardScaler``.
    """

    def __init__(
        self,
        temps: Iterable[float],
        *,
        use_kJ: bool = True,
        lnk_mean: torch.Tensor | None = None,
        lnk_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.register_buffer("T", torch.tensor(list(temps), dtype=torch.float32))
        self.R = 8.31446261815324e-3 if use_kJ else 8.31446261815324
        self.register_buffer("T0", torch.tensor(1.0, dtype=torch.float32))

        if lnk_mean is not None and lnk_scale is not None:
            self.register_buffer("lnk_mu", lnk_mean.float())
            self.register_buffer("lnk_sig", lnk_scale.float())
        else:
            self.lnk_mu = self.lnk_sig = None

    def forward(
        self, params: torch.Tensor, sampled_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Evaluate :math:`\ln k` for a batch of Arrhenius parameters.

        Parameters
        ----------
        params:
            Tensor of shape ``(B, 3)`` with columns ``[A, n, Ea]``.  ``A`` is in
            ``s⁻¹`` and ``Ea`` in ``kJ mol⁻¹`` (or ``J mol⁻¹`` when
            ``use_kJ=False``).
        sampled_indices:
            Optional indices selecting which temperatures to use.  If ``None``
            all temperatures passed at construction time are used.
        """

        A, n, Ea = params.unbind(1)

        T = self.T if sampled_indices is None else self.T[sampled_indices]
        T = T.to(A.device).unsqueeze(0)

        ln_k = (
            torch.log(torch.clamp(A, min=1e-30)).unsqueeze(1)
            + n.unsqueeze(1) * torch.log(T / self.T0)
            - Ea.unsqueeze(1) / (self.R * T)
        )

        if self.lnk_mu is not None:
            mu = (
                self.lnk_mu if sampled_indices is None else self.lnk_mu[sampled_indices]
            )
            sig = (
                self.lnk_sig
                if sampled_indices is None
                else self.lnk_sig[sampled_indices]
            )
            ln_k = (ln_k - mu) / sig

        return ln_k
