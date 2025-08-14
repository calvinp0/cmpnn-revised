import numpy as np
import torch
import pytest

from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from cmpnn.scaler.unscalers import ColumnUnscaler


def _rng_data(n=3000, seed=0):
    """
    Returns a matrix with 4 columns:
    - col1: real-valued (Yeo–Johnson ok)
    - col2: strictly positive (Box–Cox ok)
    - col3: strictly positive (for other transforms)
    - col4: heavy-tailed (to test robustness)
    """
    rng = np.random.default_rng(seed)
    col1 = rng.normal(size=(n, 1))
    col2 = rng.lognormal(mean=0.0, sigma=0.6, size=(n, 1))
    col3 = np.clip(rng.normal(loc=3.0, scale=1.0, size=(n, 1)), 1e-3, None)
    col4 = rng.standard_t(df=3, size=(n, 1))
    return np.hstack([col1, col2, col3, col4]).astype(np.float64)


@pytest.mark.parametrize("standardize", [True, False])
def test_powertransformer_inverse_matches_sklearn(standardize):
    """
    Check that ColumnwiseUnscaler reproduces sklearn's inverse for:
    - Yeo–Johnson on col1
    - Box-Cox on col2
    """
    Y = _rng_data()
    pt_yj = PowerTransformer(method="yeo-johnson", standardize=standardize).fit(
        Y[:, [0]]
    )

    pt_bc = PowerTransformer(method="box-cox", standardize=standardize).fit(Y[:, [1]])

    Yt = np.hstack([pt_yj.transform(Y[:, [0]]), pt_bc.transform(Y[:, [1]])])

    un = ColumnUnscaler([pt_yj, pt_bc])

    Y_back = un(torch.from_numpy(Yt).float()).cpu().numpy()

    np.testing.assert_allclose(Y_back, Y[:, [0, 1]], rtol=1e-5, atol=1e-6)


def test_mixed_transformers_roundtrip():
    """
    Mix affine torch-native paths and PowerTransformers.
    """
    Y = _rng_data()
    tfs = [
        StandardScaler().fit(Y[:, [0]]),
        MinMaxScaler().fit(Y[:, [1]]),
        RobustScaler().fit(Y[:, [2]]),
        PowerTransformer(method="yeo-johnson", standardize=True).fit(Y[:, [3]]),
    ]
    Yt = np.hstack([t.transform(Y[:, [i]]) for i, t in enumerate(tfs)])

    un = ColumnUnscaler(tfs)
    Y_back = un(torch.from_numpy(Yt).float()).cpu().numpy()

    np.testing.assert_allclose(Y_back, Y[:, :4], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("method", ["yeo-johnson", "box-cox"])
@pytest.mark.parametrize("standardize", [True, False])
def test_powertransformer_multi_column(method, standardize):
    """
    Fit one PowerTransformer per column on a 3-col matrix
    and verify the module handles per-column λ and std flags.
    """
    Y = _rng_data()[:, :3]
    # Ensure positivity if Box–Cox across all cols
    if method == "box-cox":
        Y = np.clip(Y, 1e-3, None)

    tfs = [
        PowerTransformer(method=method, standardize=standardize).fit(Y[:, [i]])
        for i in range(Y.shape[1])
    ]
    Yt = np.hstack([t.transform(Y[:, [i]]) for i, t in enumerate(tfs)])

    un = ColumnUnscaler(tfs)
    Y_back = un(torch.from_numpy(Yt).float()).cpu().numpy()

    np.testing.assert_allclose(Y_back, Y, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_behavior(device):
    """
    The unscaler should accept inputs on CPU or CUDA and
    emit outputs on the same device.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this runner.")

    Y = _rng_data()[:, :2]
    pt_yj = PowerTransformer(method="yeo-johnson", standardize=True).fit(Y[:, [0]])
    pt_bc = PowerTransformer(method="box-cox", standardize=True).fit(
        np.clip(Y[:, [1]], 1e-3, None)
    )

    Yt = np.hstack([pt_yj.transform(Y[:, [0]]), pt_bc.transform(Y[:, [1]])])

    un = ColumnUnscaler([pt_yj, pt_bc]).to(device)
    Yt_t = torch.from_numpy(Yt).float().to(device)
    out = un(Yt_t)

    assert out.device.type == device
    np.testing.assert_allclose(
        out.detach().cpu().numpy(), Y[:, :2], rtol=1e-5, atol=1e-6
    )


def test_state_dict_roundtrip(tmp_path):
    """
    Check that buffers (means, scales, λ, μ, σ, masks) survive save/load,
    and outputs remain identical.
    """
    Y = _rng_data()[:, :2]
    pt_yj = PowerTransformer(method="yeo-johnson", standardize=True).fit(Y[:, [0]])
    pt_bc = PowerTransformer(method="box-cox", standardize=True).fit(
        np.clip(Y[:, [1]], 1e-3, None)
    )
    tfs = [pt_yj, pt_bc]
    Yt = np.hstack([t.transform(Y[:, [i]]) for i, t in enumerate(tfs)])
    Yt_t = torch.from_numpy(Yt).float()

    un1 = ColumnUnscaler(tfs)
    out1 = un1(Yt_t).detach().cpu().numpy()

    p = tmp_path / "un.pt"
    torch.save(un1.state_dict(), p)

    # Rebuild with same structure (methods/standardize flags) then load
    un2 = ColumnUnscaler(tfs)
    un2.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))

    out2 = un2(Yt_t).detach().cpu().numpy()
    np.testing.assert_allclose(out2, out1, rtol=0, atol=0)
