import torch
from cmpnn.models.arrhenius import ArrheniusLayer


def test_arrhenius_basic():
    temps = [300.0, 400.0]
    layer = ArrheniusLayer(temps)
    params = torch.tensor([[1.0e12, 0.5, 50.0]])  # A, n, Ea
    out = layer(params)
    R = 8.31446261815324e-3
    T = torch.tensor(temps)
    expected = torch.log(torch.tensor(1.0e12)) + 0.5 * torch.log(T) - 50.0 / (R * T)
    assert torch.allclose(out.squeeze(0), expected, atol=1e-5)


def test_arrhenius_standardised():
    temps = [300.0, 400.0]
    mean = torch.tensor([1.0, 2.0])
    scale = torch.tensor([2.0, 4.0])
    layer = ArrheniusLayer(temps, lnk_mean=mean, lnk_scale=scale)
    params = torch.tensor([[1.0e12, 0.0, 0.0]])
    raw = layer(params)
    # Without standardisation the values are:
    base_layer = ArrheniusLayer(temps)
    base = base_layer(params)
    expected = (base - mean) / scale
    assert torch.allclose(raw, expected, atol=1e-5)
