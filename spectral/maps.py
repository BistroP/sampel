"""Spectral maps: G -> D, where D keeps G's singular vectors but changes the spectrum.

All maps take a 2D gradient/momentum matrix and return a float32 matrix of the same
shape. Internally the matrix is oriented "wide" (rows <= cols) so that the Gram matrix
X X^T is the small one.

Reference: Shumaylov et al., "Muon is Not That Special" (arXiv 2605.11181).
    muon_map       Algorithm 1 (quintic Newton-Schulz, Keller Jordan's coefficients)
    kaon_map       Algorithm 2 (chaotic cubic-in-X^2 iteration; random-like spectrum)
    freon_map      D = (G G^T)^{-c} G, mean-q-norm normalised (Sec 2.4, App H.3)
    truncsgd_map   zero the top p% singular values, rescale to ||G||_F (Sec 2.2)
"""
from __future__ import annotations

import math

import torch

# Keller Jordan's quintic Newton-Schulz coefficients (Muon). Output singular values
# land roughly in [0.7, 1.2] rather than exactly 1, which is what everyone calls "Muon".
NS_COEFFS = (3.4445, -4.7750, 2.0315)

# Kaon: x <- lam * x * (1 - x^2)^2 with lam = 4.1 is chaotic with support (0, 1.175).
KAON_LAMBDA = 4.1
KAON_SUPPORT = 1.175


def _orient(G: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """float32 copy with rows <= cols; returns (X, transposed)."""
    X = G.detach().float()
    transposed = X.size(0) > X.size(1)
    return (X.T if transposed else X), transposed


def _unorient(X: torch.Tensor, transposed: bool) -> torch.Tensor:
    return X.T if transposed else X


def _svd(X: torch.Tensor):
    """Thin SVD. torch.linalg.svd is not implemented on MPS, so fall back to CPU."""
    if X.device.type == "mps":
        U, S, Vh = torch.linalg.svd(X.cpu(), full_matrices=False)
        return U.to(X.device), S.to(X.device), Vh.to(X.device)
    return torch.linalg.svd(X, full_matrices=False)


# --------------------------------------------------------------------------- Muon
def muon_map(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    X, tr = _orient(G)
    X = X / (X.norm() + eps)
    a, b, c = NS_COEFFS
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return _unorient(X, tr)


# --------------------------------------------------------------------------- Kaon
def kaon_map(
    G: torch.Tensor,
    steps: int = 5,
    lam: float = KAON_LAMBDA,
    support: float = KAON_SUPPORT,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Algorithm 2. Matrix form of x -> lam * x * (1 - x^2)^2 on the singular values,
    iterated `steps` times from sigma/||G||_F, then divided by the support edge so the
    output spectrum lies in (0, 1]."""
    X, tr = _orient(G)
    X = X / (X.norm() + eps)
    I = torch.eye(X.size(0), device=X.device, dtype=X.dtype)
    for _ in range(steps):
        B = I - X @ X.T
        X = lam * (B @ B @ X)
    return _unorient(X / support, tr)


def kaon_scalar_map(x: torch.Tensor, steps: int = 5, lam: float = KAON_LAMBDA) -> torch.Tensor:
    """Scalar version of the Kaon iteration (used in tests / for plotting spectra)."""
    for _ in range(steps):
        x = lam * x * (1 - x**2) ** 2
    return x


# --------------------------------------------------------------------------- Freon
def freon_map(G: torch.Tensor, c: float, rel_eps: float = 1e-10) -> torch.Tensor:
    """Exact (SVD-based) Freon direction, D = (G G^T)^{-c} G = U diag(sigma^{1-2c}) V^T,
    normalised so that the update has mean-q-norm 1 with q = 2(1-c) (paper Sec 2.4.1 and
    App H.3):  sigma_tilde = sigma / mu_c,  mu_c = (mean sigma^q)^{1/q}.

        c = 0    -> normalised SGD direction  (spectrum ∝ sigma)
        c = 1/2  -> exact polar factor U V^T  (Muon's ideal target)
        c in (1/2, 1] -> the quasi-norm regime (q in [0, 1)); c = 1 is the q -> 0 limit
                         where mu_c is the geometric mean of sigma.
    """
    X, tr = _orient(G)
    U, S, Vh = _svd(X)
    S = S.clamp_min(S.max().item() * rel_eps + 1e-30)
    q = 2.0 * (1.0 - c)
    if abs(q) < 1e-6:  # c = 1: geometric mean
        mu = torch.exp(torch.log(S).mean())
    else:
        mu = (S**q).mean() ** (1.0 / q)
    S_new = (S / mu) ** (1.0 - 2.0 * c)
    return _unorient((U * S_new) @ Vh, tr)


# --------------------------------------------------------------------------- TruncatedSGD
def truncsgd_map(G: torch.Tensor, frac: float) -> torch.Tensor:
    """Zero the top `frac` fraction of singular values (frac in [0, 1)), then rescale so
    the Frobenius norm matches the untruncated gradient (paper App B)."""
    X, tr = _orient(G)
    if frac <= 0:
        return _unorient(X, tr)
    U, S, Vh = _svd(X)
    k = int(math.ceil(frac * S.numel()))
    S = S.clone()
    S[:k] = 0.0
    out = (U * S) @ Vh
    out = out * (X.norm() / (out.norm() + 1e-30))
    return _unorient(out, tr)


def apply_map(G: torch.Tensor, method: str, **kw) -> torch.Tensor:
    """Dispatch by name. Methods: muon | kaon | freon | truncsgd | sgd."""
    if method == "muon":
        return muon_map(G, **kw)
    if method == "kaon":
        return kaon_map(G, **kw)
    if method == "freon":
        return freon_map(G, **kw)
    if method == "truncsgd":
        return truncsgd_map(G, **kw)
    if method == "sgd":
        return G.detach().float()
    raise ValueError(f"unknown method {method!r}")
