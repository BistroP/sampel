import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral import SpectralOptimizer, freon_map, kaon_map, muon_map, split_params, truncsgd_map
from spectral.gpt import GPT, GPTConfig
from spectral.maps import kaon_scalar_map


def sv(M):
    return torch.linalg.svdvals(M.double())


def make_G(m=64, n=192, seed=0):
    g = torch.Generator().manual_seed(seed)
    # decaying spectrum like a real gradient
    U, _ = torch.linalg.qr(torch.randn(m, m, generator=g))
    V, _ = torch.linalg.qr(torch.randn(n, m, generator=g))
    S = torch.arange(1, m + 1, dtype=torch.float32) ** -1.0
    return (U * S) @ V.T


@pytest.mark.parametrize("shape", [(64, 192), (192, 64), (64, 64)])
def test_muon_spectrum_is_flat(shape):
    G = make_G(*shape) if shape[0] <= shape[1] else make_G(shape[1], shape[0]).T
    s = sv(muon_map(G))
    assert 0.5 < s.min() and s.max() < 1.5  # Newton-Schulz "approximately 1" band
    assert abs(s.mean() - 0.85) < 0.2


def test_freon_half_is_polar_factor():
    G = make_G()
    U, _, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    D = freon_map(G, c=0.5).double()
    assert torch.allclose(D, U @ Vh, atol=1e-4)


def test_freon_zero_is_scaled_gradient():
    G = make_G()
    D = freon_map(G, c=0.0)
    r = min(G.shape)
    assert torch.allclose(D / D.norm(), G / G.norm(), atol=1e-5)
    assert abs(D.norm().item() - math.sqrt(r)) < 1e-3  # mean-2-norm == 1


def test_freon_one_is_pseudoinverse_like():
    G = make_G()
    D = freon_map(G, c=1.0).double()
    M = D @ G.double().T  # (GG^T)^{-1} G G^T = I up to scale
    assert torch.allclose(M / M[0, 0], torch.eye(M.shape[0], dtype=M.dtype), atol=1e-3)


@pytest.mark.parametrize("c", [0.0, 0.25, 0.5, 2 / 3, 0.75, 1.0])
def test_freon_mean_q_norm_is_one_and_transpose_equivariant(c):
    G = make_G()
    D = freon_map(G, c)
    s = sv(D)
    q = 2 * (1 - c)
    # D has spectrum (sigma/mu)^(q-1) with mu the mean-q power mean of sigma
    S = sv(G)
    mu = S.log().mean().exp() if abs(q) < 1e-6 else (S**q).mean() ** (1 / q)
    expected = (S / mu) ** (1 - 2 * c)
    # for c > 1/2 the exponent is negative, so the spectrum order reverses: compare sorted
    assert torch.allclose(s.sort().values, expected.sort().values, rtol=1e-3, atol=1e-5)
    assert torch.allclose(freon_map(G.T.contiguous(), c), D.T, atol=1e-4)


def test_freon_quasi_norm_exponent_orders_spectrum():
    # larger c -> flatter-then-inverted spectrum: sigma^(1-2c) flips ordering for c > 1/2
    G = make_G()
    s_half, s_23 = sv(freon_map(G, 0.5)), sv(freon_map(G, 2 / 3))
    assert s_half.max() / s_half.min() < 1 + 1e-4
    assert s_23.max() / s_23.min() > 1.5  # small gradient directions are now *boosted*


def test_kaon_matches_scalar_map_on_singular_values():
    G = make_G()
    S = sv(G)
    expected = kaon_scalar_map((S / S.norm()).float(), steps=5) / 1.175
    got = sv(kaon_map(G))
    assert torch.allclose(got.sort().values, expected.double().sort().values, atol=1e-4)
    assert got.max() <= 1.0 + 1e-5 and got.min() >= 0


def test_kaon_is_deterministic_and_spread():
    G = make_G()
    assert torch.equal(kaon_map(G), kaon_map(G))
    s = sv(kaon_map(G))
    assert s.std() / s.mean() > 0.2  # nothing like a flat Muon spectrum


def test_truncsgd_zeroes_top_and_keeps_norm():
    G = make_G()
    D = truncsgd_map(G, 0.1)
    s = sv(D)
    k = math.ceil(0.1 * 64)
    assert (s < 1e-5).sum() == k  # exactly k singular values zeroed
    assert abs(D.norm().item() - G.norm().item()) < 1e-4
    assert torch.equal(truncsgd_map(G, 0.0), G.float())


def test_optimizer_decreases_quadratic_for_every_method():
    torch.manual_seed(0)
    target = torch.randn(32, 48)
    for method, kw in [("muon", {}), ("kaon", {}), ("freon", {"c": 2 / 3}), ("sgd", {}), ("truncsgd", {"frac": 0.05})]:
        W = torch.nn.Parameter(torch.zeros(32, 48))
        opt = SpectralOptimizer([W], lr=0.1 if method not in ("sgd", "truncsgd") else 0.5,
                                momentum=0.9, method=method, method_kwargs=kw)
        l0 = ((W - target) ** 2).sum().item()
        for _ in range(60):
            opt.zero_grad()
            ((W - target) ** 2).sum().backward()
            opt.step()
        assert ((W - target) ** 2).sum().item() < 0.7 * l0, method


def test_optimizer_handles_conv_filters():
    W = torch.nn.Parameter(torch.randn(16, 3, 3, 3))
    opt = SpectralOptimizer([W], lr=0.01, method="freon", method_kwargs={"c": 0.6})
    (W**2).sum().backward()
    opt.step()
    assert W.shape == (16, 3, 3, 3) and torch.isfinite(W).all()


def test_gpt_matches_proposal_size_and_param_split():
    cfg = GPTConfig()
    model = GPT(cfg)
    assert 10.4e6 < model.num_params() < 10.9e6, model.num_params()  # "about 10.7M"
    spectral, decay, no_decay = split_params(model)
    assert len(spectral) == 4 * cfg.n_layer
    assert {tuple(p.shape) for p in spectral} == {(3 * 384, 384), (384, 384), (4 * 384, 384), (384, 4 * 384)}
    assert len(decay) == 2  # wte (tied lm_head) + wpe
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    tgt = torch.randint(0, cfg.vocab_size, (2, 16))  # independent of idx (tied embeddings leak idx)
    _, loss = model(idx, tgt)
    assert math.isfinite(loss.item()) and abs(loss.item() - math.log(cfg.vocab_size)) < 0.3
