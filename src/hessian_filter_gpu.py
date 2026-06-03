"""
GPU Jerman / vesselness3D — matches T. Jerman MATLAB reference (Kroon Hessian + Yang–Cheng mask).

Tensor layout: [B, C, D, H, W] (MONAI-style).
  - dim W (index 4) ≈ MATLAB ``x`` (1st index)
  - dim H (index 3) ≈ MATLAB ``y``
  - dim D (index 2) ≈ MATLAB ``z`` (3rd index)

``spacing`` must align with **D, H, W** in that order. If your volume stores patient Z, Y, X
along D, H, W respectively, pass ``spacing=[sz, sy, sx]`` (z, y, x).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Union

import torch
import torch.nn.functional as F


def vesselness3D(
    I, # NOTE: Z Y X order
    sigmas: Sequence[float] = (0.9, 1.6, 2.3, 3.0),
    spacing: Union[Sequence[float], torch.Tensor] = (1.0, 1.0, 1.0), # NOTE: Z Y X order
    tau: float = 1.0,
    brightondark: bool = True,
    device: Union[str, torch.device] = "cuda",
    verbose: bool = True,
):
    """
    Jerman 3D vesselness (max over scales), aligned with ``vesselness3D.m``.

    Non-finite voxels are zeroed; computation uses float32 (as MATLAB ``single``).
    """
    if not isinstance(I, torch.Tensor):
        I = torch.tensor(I, dtype=torch.float32, device=device)
    else:
        I = I.to(device).float()

    I = torch.where(torch.isfinite(I), I, torch.zeros_like(I))

    if I.ndim == 3:
        I = I.unsqueeze(0).unsqueeze(0)

    if not torch.is_tensor(spacing):
        spacing_t = torch.tensor(spacing, device=I.device, dtype=torch.float32)
    else:
        spacing_t = spacing.to(device=I.device, dtype=torch.float32)

    vesselness = torch.zeros_like(I)

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.preferred_linalg_library("magma")
        except Exception:
            pass

    for sigma in sigmas:
        if verbose:
            print(f"Current Filter Sigma: {sigma}")

        lambda2, lambda3 = volume_eigenvalues(I, float(sigma), spacing_t, brightondark)

        if brightondark:
            lambda2 = -lambda2
            lambda3 = -lambda3

        max_lambda3 = torch.max(lambda3)
        lambda_rho = lambda3.clone()
        mask = (lambda3 > 0) & (lambda3 <= tau * max_lambda3)
        lambda_rho[mask] = tau * max_lambda3
        lambda_rho[lambda3 <= 0] = 0

        denom = lambda2 + lambda_rho
        response = (lambda2**2 * (lambda_rho - lambda2) * 27.0) / (denom**3)
        response[(lambda2 >= lambda_rho / 2) & (lambda_rho > 0)] = 1.0
        response[(lambda2 <= 0) | (lambda_rho <= 0)] = 0.0
        response[~torch.isfinite(response)] = 0.0

        vesselness = torch.max(vesselness, response)

    # MATLAB: vesselness ./ max(vesselness(:)) — clamp avoids NaNs if the map is all zeros.
    vesselness = vesselness / torch.clamp(torch.max(vesselness), min=1e-30)
    vesselness[vesselness < 1e-2] = 0

    return vesselness.squeeze()


def _swap_lambda_cols_inplace(dd: torch.Tensor, m: torch.Tensor, a: int, b: int) -> None:
    """dd (N,3), m (N,) — swap columns a,b where m is True (eig3volume-style swaps)."""
    ta = dd[:, a].clone()
    tb = dd[:, b].clone()
    dd[m, a] = tb[m]
    dd[m, b] = ta[m]


def reorder_eigenvalues_like_eig3volume(eig: torch.Tensor) -> torch.Tensor:
    """
    Match MATLAB ``eig3volume``: ``eigvalsh`` ascending is first passed through the same
    reordering as ``eigen_decomposition`` after ``tql2`` (``eig3volume_reference.c``
    lines 221–239). Eigenvalues become ``|λ|:`` small → medium → large along columns
    ``[...,0],[...,1],[...,2]``, each column still holding the **signed** value.

    This differs from plain ``torch.linalg.eigvalsh``, which sorts only algebraically.
    """
    prefix = eig.shape[:-1]
    dd = eig.reshape(-1, 3).clone()
    da = dd.abs()
    da0, da1, da2 = da[:, 0], da[:, 1], da[:, 2]

    m1 = (da0 >= da1) & (da0 > da2)
    _swap_lambda_cols_inplace(dd, m1, 0, 2)

    da = dd.abs()
    da0, da1, da2 = da[:, 0], da[:, 1], da[:, 2]
    m2 = (~m1) & (da1 >= da0) & (da1 > da2)
    _swap_lambda_cols_inplace(dd, m2, 1, 2)

    da = dd.abs()
    da0, da1 = da[:, 0], da[:, 1]
    m3 = da0 > da1
    _swap_lambda_cols_inplace(dd, m3, 0, 1)

    return dd.reshape(prefix + (3,))


def volume_eigenvalues(
    V: torch.Tensor,
    sigma: float,
    spacing: torch.Tensor,
    brightondark: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Eigenvalues Λ1, Λ2, Λ3 per voxel (Λ2, Λ3 used by Jerman); Yang–Cheng mask and
    small-|λ| pruning as in MATLAB ``volumeEigenvalues``. Eigen ordering matches
    ``eig3volume`` (sort by magnitude tier), not raw ``eigvalsh``.
    """
    Hxx, Hyy, Hzz, Hxy, Hxz, Hyz = hessian3D(V, sigma, spacing)

    c = sigma**2
    Hxx, Hyy, Hzz = Hxx * c, Hyy * c, Hzz * c
    Hxy, Hxz, Hyz = Hxy * c, Hxz * c, Hyz * c

    B1 = -(Hxx + Hyy + Hzz)
    B2 = Hxx * Hyy + Hxx * Hzz + Hyy * Hzz - Hxy * Hxy - Hxz * Hxz - Hyz * Hyz
    B3 = (
        Hxx * Hyz * Hyz
        + Hxy * Hxy * Hzz
        + Hxz * Hyy * Hxz
        - Hxx * Hyy * Hzz
        - Hxy * Hyz * Hxz
        - Hxz * Hxy * Hyz
    )

    T = torch.ones_like(B1, dtype=torch.bool)
    if brightondark:
        T &= B1 > 0
        T &= ~((B2 <= 0) & (B3 == 0))
        T &= ~((B1 > 0) & (B2 > 0) & (B1 * B2 < B3))
    else:
        T &= B1 < 0
        T &= ~((B2 >= 0) & (B3 == 0))
        T &= ~((B1 < 0) & (B2 < 0) & ((-B1) * (-B2) < (-B3)))

    flat_idx = torch.nonzero(T.reshape(-1), as_tuple=False).squeeze(-1)
    n = int(flat_idx.numel())

    lambda1 = torch.zeros_like(Hxx)
    lambda2 = torch.zeros_like(Hxx)
    lambda3 = torch.zeros_like(Hxx)

    if n == 0:
        return lambda2, lambda3

    hxx = Hxx.reshape(-1)[flat_idx]
    hyy = Hyy.reshape(-1)[flat_idx]
    hzz = Hzz.reshape(-1)[flat_idx]
    hxy = Hxy.reshape(-1)[flat_idx]
    hxz = Hxz.reshape(-1)[flat_idx]
    hyz = Hyz.reshape(-1)[flat_idx]

    stack = torch.stack(
        [
            torch.stack([hxx, hxy, hxz], dim=-1),
            torch.stack([hxy, hyy, hyz], dim=-1),
            torch.stack([hxz, hyz, hzz], dim=-1),
        ],
        dim=-2,
    )

    eig_alg = torch.linalg.eigvalsh(stack)
    eig_m = reorder_eigenvalues_like_eig3volume(eig_alg)
    l1i, l2i, l3i = eig_m[:, 0], eig_m[:, 1], eig_m[:, 2]

    for L, Li in ((lambda1, l1i), (lambda2, l2i), (lambda3, l3i)):
        Lf = L.reshape(-1)
        Lf[flat_idx] = Li

    for L in (lambda1, lambda2, lambda3):
        L[~torch.isfinite(L)] = 0
        L[L.abs() < 1e-4] = 0

    return lambda2, lambda3


def hessian3D(V: torch.Tensor, sigma: float, spacing: torch.Tensor):
    """Gaussian-smoothed Kroon Hessian"""
    V_smoothed = imgaussian(V, sigma, spacing)

    def gradient(f: torch.Tensor, dim: int) -> torch.Tensor:
        diff = torch.zeros_like(f)
        slice_post = [slice(None)] * 5
        slice_pre = [slice(None)] * 5
        slice_mid = [slice(None)] * 5

        slice_post[dim] = slice(2, None)
        slice_pre[dim] = slice(0, -2)
        slice_mid[dim] = slice(1, -1)

        diff[tuple(slice_mid)] = (f[tuple(slice_post)] - f[tuple(slice_pre)]) / 2.0

        slice_0 = [slice(None)] * 5
        slice_0[dim] = 0
        slice_1 = [slice(None)] * 5
        slice_1[dim] = 1
        diff[tuple(slice_0)] = f[tuple(slice_1)] - f[tuple(slice_0)]

        slice_last = [slice(None)] * 5
        slice_last[dim] = -1
        slice_penu = [slice(None)] * 5
        slice_penu[dim] = -2
        diff[tuple(slice_last)] = f[tuple(slice_last)] - f[tuple(slice_penu)]

        return diff

    dx = gradient(V_smoothed, 4)
    dy = gradient(V_smoothed, 3)
    dz = gradient(V_smoothed, 2)

    dxx = gradient(dx, 4)
    dyy = gradient(dy, 3)
    dzz = gradient(dz, 2)

    dxy = gradient(dx, 3)
    dxz = gradient(dx, 2)
    dyz = gradient(dy, 2)

    return dxx, dyy, dzz, dxy, dxz, dyz


def imgaussian(I: torch.Tensor, sigma: float, spacing: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian with Kroon/Jerman kernel support and replicate padding (MATLAB ``imfilter`` replicate)."""
    if sigma <= 0:
        return I

    s_list: List[float] = spacing.flatten().tolist()
    res = I
    siz = float(sigma) * 6.0

    for i, s in enumerate(s_list):
        sig = float(sigma) / float(s)
        hw = int(math.ceil(siz / float(s) / 2.0))
        t = torch.arange(-hw, hw + 1, device=res.device, dtype=torch.float32)
        ker_1d = torch.exp(-(t**2) / (2 * sig**2))
        ker_1d = ker_1d / ker_1d.sum()
        klen = int(ker_1d.numel())
        pad = klen // 2

        shape = [1, 1, 1, 1, 1]
        shape[2 + i] = klen
        ker = ker_1d.view(*shape)

        # i=0 (Depth/Z dimension)
        # Tuple: (W_pad, W_pad, H_pad, H_pad, D_pad, D_pad)
        # We only want to pad Depth, so we use (0, 0, 0, 0, pad, pad)
        if i == 0:
            res = F.pad(res, (0, 0, 0, 0, pad, pad), mode="replicate")
        # i=1 (Height/Y dimension)
        # We only want to pad Height, so we use (0, 0, pad, pad, 0, 0)
        elif i == 1:
            res = F.pad(res, (0, 0, pad, pad, 0, 0), mode="replicate")
        # i=2 (Width/X dimension)
        # We only want to pad Width, so we use (pad, pad, 0, 0, 0, 0)
        else:
            res = F.pad(res, (pad, pad, 0, 0, 0, 0), mode="replicate")

        res = F.conv3d(res, ker, padding=0)

    return res