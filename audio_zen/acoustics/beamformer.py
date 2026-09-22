#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/16 17:16
# @Author  : Guinan Li
# @File    : beamformer.py
import torch
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor


def get_power_spectral_density_matrix(xs: ComplexTensor,
                                      mask: ComplexTensor = None,
                                      averaging=True,
                                      normalization=True,
                                      eps: float = 1e-6,
                                      mode: str = 'mask'  # 'mask' or 'direct'
                                      ) -> ComplexTensor:
    """Return cross-channel power spectral density (PSD) matrix

    Args:
        xs (ComplexTensor): (..., F, C, T) - input complex spectrogram
        mask (ComplexTensor, optional): (..., F, C, T) - mask for mask-based method
        averaging (bool): whether to average over time dimension
        normalization (bool): whether to normalize
        eps (float): epsilon for numerical stability
        mode (str): 'mask' for mask-based method, 'direct' for direct calculation
    Returns
        psd (ComplexTensor): (..., F, C, C)
    """
    if mode == 'direct':
        # Direct calculation from input (e.g., noise speech)
        return get_psd_direct(xs, averaging, normalization, eps)
    else:
        # Original mask-based method
        return get_psd_with_mask(xs, mask, averaging, normalization, eps)


def get_psd_direct(xs: ComplexTensor,
                   normalization=True,
                   eps: float = 1e-6
                   ) -> ComplexTensor:
    """Direct PSD calculation from input spectrogram"""
    # outer product: (..., C, T) x (..., C, T) -> (..., T, C, C)
    psd = FC.einsum('...ct,...et->...tce', [xs, xs.conj()])

    # normalize psd with T
    T = xs.shape[-1]
    psd = psd / (T + eps)

    psd = psd.mean(dim=-3)

    return psd


def get_psd_with_mask(xs: ComplexTensor,
                      mask: ComplexTensor,
                      eps: float = 1e-6
                      ) -> ComplexTensor:
    """Original mask-based PSD calculation"""
    # outer product: (..., C, T) x (..., C, T) -> (..., T, C, C)
    psd = FC.einsum('...ct,...et->...tce', [xs, xs.conj()])

    # normalize psd with mask
    mask_power = mask.real * mask.real + mask.imag * mask.imag
    norm_mask = mask_power.sum(dim=-1, keepdim=True) + eps
    psd = psd / norm_mask[..., None, None]

    # Average over time
    psd = psd.sum(dim=-3)

    return psd


def get_mvdr_vector_stable(
    psd_s: ComplexTensor,
    psd_n: ComplexTensor,
    reference_vector: torch.Tensor,
    use_diag_loading: bool = True,
    diag_loading_ratio: float = 1e-5,
    use_torch_solver: bool = True,
    eps = 1e-5
) -> ComplexTensor:
    """Return the MVDR(Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (ComplexTensor): (..., F, C, C)
        psd_n (ComplexTensor): (..., F, C, C)
        reference_vector (torch.Tensor): (..., C)
        eps (float):
    Returns:
        beamform_vector (ComplexTensor)r: (..., F, C)
    """
    # Add eps
    B, F = psd_n.shape[:2]
    C = psd_n.size(-1)
    eye = torch.eye(C, dtype=psd_n.dtype, device=psd_n.device)
    shape = [1 for _ in range(psd_n.dim() - 2)] + [C, C]
    eye = eye.view(*shape).repeat(B, F, 1, 1)
    epsilon = None
    if use_diag_loading:
        with torch.no_grad():
            epsilon = FC.trace(psd_n).real.abs()[..., None, None] * diag_loading_ratio
            # in case that correlation_matrix is all-zero
            # import pdb; pdb.set_trace()
            epsilon += diag_loading_ratio
    else:
        epsilon = diag_loading_ratio

    if use_torch_solver:
        # import pdb; pdb.set_trace()
        # print("use_torch_solver")
        numerator = FC.solve(psd_s, psd_n + epsilon * eye)
    else:
        psd_n += epsilon * eye
        try:
            psd_n_i = psd_n.inverse()
        except:
            try:
                reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real), torch.rand_like(psd_n.real)) * 1e-2
                psd_n = psd_n / 10e+4
                psd_s = psd_s / 10e+4
                psd_n += reg_coeff_tensor
                psd_n_i = psd_n.inverse()
            except:
                try:
                    reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real), torch.rand_like(psd_n.real)) * 1e-1
                    psd_n = psd_n / 10e+10
                    psd_s = psd_s / 10e+10
                    psd_n += reg_coeff_tensor
                    psd_n_i = psd_n.inverse()
                except:
                    try:
                        reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                                         torch.rand_like(psd_n.real)) * 1e-1
                        psd_n = psd_n / 10e+10
                        psd_s = psd_s / 10e+10
                        psd_n += reg_coeff_tensor
                        psd_n_i = psd_n.inverse()
                    except:
                        reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real), torch.rand_like(psd_n.real))
                        psd_n = psd_n / 10e+10
                        psd_s = psd_s / 10e+10
                        psd_n += reg_coeff_tensor
                        psd_n_i = psd_n.inverse()

        # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
        # [32, 257, 15, 15]
        numerator = FC.einsum('...ec,...cd->...ed', [psd_n_i, psd_s])

    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = FC.einsum("...fec,...c->...fe", [ws, reference_vector])
    return beamform_vector


def apply_beamforming_vector(beamform_vector: ComplexTensor,
                             mix: ComplexTensor) -> ComplexTensor:
    # (..., C) x (..., C, T) -> (..., T)
    es = FC.einsum('...c,...ct->...t', [beamform_vector.conj(), mix])
    return es
