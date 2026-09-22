#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multichn_librispeech_v2 DOA scheme:

Phase 1: DOA-only model from scratch (no separation modules on GPU).
Phase 2: merge phase1 DOA (+ ln_LPS_doa) with GT-DOA pretrained sep
         (+ ln_LPS_sep); pred-AF; joint SI-SNR + CE.

Dual LPS LayerNorm:
  - ln_LPS_doa: trained in phase1, kept for DOA path in phase2
  - ln_LPS_sep: loaded from GT-DOA pretrained Model for separation path
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_zen.model.base_model import BaseModel
from audio_feature_doa import DFComputer, iSTFT, ChannelWiseLayerNorm
from tacnet import TAC
from model import Conv1D, Conv1DBlock, build_norm


def angle_rad_to_bin(angle_rad, num_bins, angle_range_deg=180.0):
    """Continuous rad -> bin index over [0, angle_range_deg)."""
    angle_deg = angle_rad * 180.0 / math.pi
    bin_width = float(angle_range_deg) / num_bins
    if torch.is_tensor(angle_deg):
        angle_deg = angle_deg % 360.0
        angle_deg = angle_deg.clamp(0.0, float(angle_range_deg) - 1e-4)
        return (angle_deg / bin_width).long().clamp(max=num_bins - 1)
    angle_deg = float(angle_deg) % 360.0
    angle_deg = min(max(angle_deg, 0.0), float(angle_range_deg) - 1e-4)
    return min(int(angle_deg / bin_width), num_bins - 1)


def bin_to_angle_rad(bin_idx, num_bins, angle_range_deg=180.0):
    angle_deg = (bin_idx.float() + 0.5) * (float(angle_range_deg) / num_bins)
    return angle_deg * math.pi / 180.0


def logits_to_angle(logits, num_bins, hard=False, angle_range_deg=180.0):
    if hard:
        bin_idx = logits.argmax(dim=-1)
        return bin_to_angle_rad(bin_idx, num_bins, angle_range_deg)
    bin_centers = torch.arange(num_bins, device=logits.device, dtype=logits.dtype)
    bin_centers = bin_to_angle_rad(bin_centers, num_bins, angle_range_deg)
    probs = F.softmax(logits, dim=-1)
    if float(angle_range_deg) >= 360.0 - 1e-6:
        cos_e = torch.sum(probs * torch.cos(bin_centers), dim=-1)
        sin_e = torch.sum(probs * torch.sin(bin_centers), dim=-1)
        return torch.atan2(sin_e, cos_e) % (2.0 * math.pi)
    return torch.sum(probs * bin_centers, dim=-1)


def build_target_directions(est_angle, max_nspk):
    B = est_angle.shape[0]
    directions = torch.full(
        (B, max_nspk), -1.0, device=est_angle.device, dtype=est_angle.dtype
    )
    directions[:, 0] = est_angle
    return directions


def _strip_module_prefix(state):
    return {k.replace("module.", ""): v for k, v in state.items()}


def _extract_state_dict(ckpt):
    if "model" in ckpt:
        return ckpt["model"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


class ModelDOA(BaseModel):
    """
    training_phase:
        1 - DOA-only (no sep modules constructed)
        2 - DOA + sep; dual ln_LPS; pred-AF; joint SI-SNR + CE

    onset_pool_ratio:
        DOA temporal pooling: average only the first `ratio` of frames
        (target often leads interferers in simulation). 1.0 = full utterance.
    """

    def __init__(
        self,
        norm,
        out_spk,
        non_linear,
        causal,
        input_features,
        cosIPD,
        sinIPD,
        N, V, X, B, H, P, R,
        HOP_SIZE,
        FFT_SIZE,
        NFFT,
        merge_mode,
        speaker_feature_dim,
        sr,
        AF_premasking,
        num_doa_bins=18,
        doa_angle_range_deg=180.0,
        max_nspk=3,
        training_phase=1,
        use_pred_doa_for_af=False,
        onset_pool_ratio=1.0,
    ):
        super().__init__()
        self.num_doa_bins = num_doa_bins
        self.doa_angle_range_deg = float(doa_angle_range_deg)
        self.max_nspk = max_nspk
        self.training_phase = int(training_phase)
        self.use_pred_doa_for_af = bool(use_pred_doa_for_af)
        self.onset_pool_ratio = onset_pool_ratio
        self.B = B
        self.out_spk = out_spk
        self._build_kwargs = dict(
            norm=norm, out_spk=out_spk, causal=causal,
            HOP_SIZE=HOP_SIZE, FFT_SIZE=FFT_SIZE, NFFT=NFFT,
            X=X, B=B, H=H, P=P,
        )

        # Feature front-end (STFT / IPD / AF). Internal ln_LPS unused when
        # we pass ln_LPS_doa / ln_LPS_sep explicitly.
        self.df_computer = DFComputer(
            frame_hop=HOP_SIZE,
            frame_len=FFT_SIZE,
            NFFT=NFFT,
            in_feature=input_features,
            merge_mode=merge_mode,
            cosIPD=cosIPD,
            sinIPD=sinIPD,
            speaker_feature_dim=speaker_feature_dim,
            sr=sr,
            AF_premasking=AF_premasking,
        )
        # Freeze unused default LN inside df_computer (we use explicit LNs)
        if hasattr(self.df_computer, "ln_LPS"):
            for p in self.df_computer.ln_LPS.parameters():
                p.requires_grad = False

        dim_no_af = self.df_computer.df_dim_no_af

        # DOA-path LPS LayerNorm (phase1 from-scratch; phase2 keep phase1 weights)
        self.ln_LPS_doa = ChannelWiseLayerNorm(self.df_computer.num_bins)

        # -------- DOA branch --------
        self.conv1x1_pre = Conv1D(dim_no_af, B, 1)
        self.audio_block_1_doa = self._build_repeats(1, X, B, H, P, norm, causal)
        self.tac1_doa = TAC(B, B, B)
        self.audio_block_2_doa = self._build_repeats(1, X, B, H, P, norm, causal)
        self.tac2_doa = TAC(B, B, B)
        self.doa_head = nn.Linear(B, num_doa_bins)

        # -------- separation branch: only in phase 2 --------
        self.has_sep = False
        if self.training_phase == 2:
            self._build_separation_branch()

        self._apply_training_phase()

    def _build_separation_branch(self):
        kw = self._build_kwargs
        dim_full = self.df_computer.df_dim
        B, H, P, X = kw["B"], kw["H"], kw["P"], kw["X"]
        norm, causal = kw["norm"], kw["causal"]
        out_spk = kw["out_spk"]
        FFT_SIZE, HOP_SIZE = kw["FFT_SIZE"], kw["HOP_SIZE"]

        # Sep-path LPS LayerNorm (loaded from GT-DOA pretrained Model)
        self.ln_LPS_sep = ChannelWiseLayerNorm(self.df_computer.num_bins)

        self.conv1x1_1 = Conv1D(dim_full, B, 1)
        self.audio_block_1 = self._build_repeats(1, X, B, H, P, norm, causal)
        self.tac1 = TAC(B, B, B)
        self.audio_block_2 = self._build_repeats(1, X, B, H, P, norm, causal)
        self.tac2 = TAC(B, B, B)
        self.audio_block_3 = self._build_repeats(1, X, B, H, P, norm, causal)
        self.tac3 = TAC(B, B, B)
        self.fusion_block = self._build_repeats(3, X, B, H, P, norm, causal)
        self.conv1d_real = Conv1D(B, out_spk * self.df_computer.num_bins, 1)
        self.conv1d_imag = Conv1D(B, out_spk * self.df_computer.num_bins, 1)
        self.istft = iSTFT(frame_len=FFT_SIZE, frame_hop=HOP_SIZE, num_fft=FFT_SIZE)
        self.has_sep = True

    def _build_repeats(self, num_repeats, num_blocks, in_channels, conv_channels,
                       kernel_size, norm, causal):
        repeats = [
            nn.Sequential(*[
                Conv1DBlock(
                    in_channels=in_channels,
                    conv_channels=conv_channels,
                    kernel_size=kernel_size,
                    dilation=(2 ** b),
                    norm=norm,
                    causal=causal,
                )
                for b in range(num_blocks)
            ])
            for _ in range(num_repeats)
        ]
        return nn.Sequential(*repeats)

    def _sep_modules(self):
        if not self.has_sep:
            return []
        return [
            self.ln_LPS_sep,
            self.conv1x1_1, self.audio_block_1, self.tac1,
            self.audio_block_2, self.tac2,
            self.audio_block_3, self.tac3, self.fusion_block,
            self.conv1d_real, self.conv1d_imag,
        ]

    def _doa_modules(self):
        return [
            self.ln_LPS_doa,
            self.conv1x1_pre, self.audio_block_1_doa, self.tac1_doa,
            self.audio_block_2_doa, self.tac2_doa, self.doa_head,
        ]

    def _set_requires_grad(self, modules, flag):
        for mod in modules:
            for p in mod.parameters():
                p.requires_grad = flag

    def _apply_training_phase(self):
        doa_mods = self._doa_modules()
        if self.training_phase == 1:
            if self.has_sep:
                raise RuntimeError("Phase1 must not contain separation modules.")
            self._set_requires_grad(doa_mods, True)
            self.use_pred_doa_for_af = False
        elif self.training_phase == 2:
            if not self.has_sep:
                raise RuntimeError("Phase2 requires separation modules.")
            self._set_requires_grad(doa_mods + self._sep_modules(), True)
            self.use_pred_doa_for_af = True
        else:
            raise ValueError(f"Unsupported training_phase={self.training_phase}")

    def load_doa_from_checkpoint(self, ckpt_path, map_location="cpu"):
        """Load DOA branch + ln_LPS_doa from phase1 ckpt."""
        state = _strip_module_prefix(
            _extract_state_dict(torch.load(ckpt_path, map_location=map_location))
        )
        doa_prefixes = (
            "ln_LPS_doa.",
            "conv1x1_pre.",
            "audio_block_1_doa.",
            "tac1_doa.",
            "audio_block_2_doa.",
            "tac2_doa.",
            "doa_head.",
        )
        doa_keys = {k: v for k, v in state.items() if k.startswith(doa_prefixes)}
        # Backward compat: old phase1 may have saved df_computer.ln_LPS
        if "ln_LPS_doa.weight" not in doa_keys and "df_computer.ln_LPS.weight" in state:
            doa_keys["ln_LPS_doa.weight"] = state["df_computer.ln_LPS.weight"]
            doa_keys["ln_LPS_doa.bias"] = state["df_computer.ln_LPS.bias"]
            print("[load_doa_from_checkpoint] mapped df_computer.ln_LPS -> ln_LPS_doa")
        if not doa_keys:
            raise RuntimeError(f"No DOA keys found in {ckpt_path}.")
        missing, unexpected = self.load_state_dict(doa_keys, strict=False)
        print(
            f"[load_doa_from_checkpoint] loaded {len(doa_keys)} tensors from "
            f"{ckpt_path}; missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def load_sep_from_pretrained(self, ckpt_path, map_location="cpu"):
        """Load sep backbone + map pretrained df_computer.ln_LPS -> ln_LPS_sep."""
        if not self.has_sep:
            raise RuntimeError("Separation branch not built.")
        state = _strip_module_prefix(
            _extract_state_dict(torch.load(ckpt_path, map_location=map_location))
        )
        sep_module_names = {
            "conv1x1_1",
            "audio_block_1",
            "audio_block_2",
            "audio_block_3",
            "tac1",
            "tac2",
            "tac3",
            "fusion_block",
            "conv1d_real",
            "conv1d_imag",
        }
        sep_keys = {}
        for k, v in state.items():
            top = k.split(".", 1)[0]
            if top in sep_module_names:
                sep_keys[k] = v
        # Map pretrained LPS LN onto sep-path LN
        if "df_computer.ln_LPS.weight" in state:
            sep_keys["ln_LPS_sep.weight"] = state["df_computer.ln_LPS.weight"]
            sep_keys["ln_LPS_sep.bias"] = state["df_computer.ln_LPS.bias"]
        if not sep_keys:
            raise RuntimeError(f"No separation keys found in {ckpt_path}.")
        missing, unexpected = self.load_state_dict(sep_keys, strict=False)
        has_ln = "ln_LPS_sep.weight" in sep_keys
        print(
            f"[load_sep_from_pretrained] loaded {len(sep_keys)} tensors "
            f"(ln_LPS_sep={'yes' if has_ln else 'NO'}) from {ckpt_path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if not has_ln:
            print(
                "[load_sep_from_pretrained] WARNING: pretrained ckpt has no "
                "df_computer.ln_LPS; ln_LPS_sep stays random."
            )

    def merge_phase1_doa_and_sep_pretrained(
        self, phase1_ckpt, sep_pretrained_ckpt, map_location="cpu"
    ):
        self.load_doa_from_checkpoint(phase1_ckpt, map_location=map_location)
        self.load_sep_from_pretrained(sep_pretrained_ckpt, map_location=map_location)
        self._apply_training_phase()
        print(
            f"[merge] DOA+ln_LPS_doa <- {phase1_ckpt}; "
            f"sep+ln_LPS_sep <- {sep_pretrained_ckpt}"
        )

    def _pool_for_doa(self, emb):
        """
        emb: [B, C, F, T] after tac2_doa.
        onset_pool_ratio in (0,1): mean-pool only first ratio*T frames
        (target-lead prior). 1.0 / None: full-utterance mean.
        """
        emb = emb.mean(dim=1)  # [B, F, T]
        T = emb.shape[-1]
        ratio = self.onset_pool_ratio
        if ratio is not None and 0.0 < float(ratio) < 1.0:
            t_keep = max(1, int(T * float(ratio)))
            emb = emb[..., :t_keep]
        return emb.mean(dim=-1)  # [B, F]

    def _forward_doa_branch(self, base_df):
        B, C, F_base, T = base_df.shape
        emb = self.conv1x1_pre(base_df.reshape(B * C, F_base, T))
        emb = self.audio_block_1_doa(emb)
        emb = self.tac1_doa(emb.view(B, C, emb.shape[-2], T))
        emb = self.audio_block_2_doa(emb.view(B * C, emb.shape[-2], T))
        emb = self.tac2_doa(emb.view(B, C, emb.shape[-2], T))
        pooled = self._pool_for_doa(emb)
        return self.doa_head(pooled)

    def _forward_separation(self, audio_fea, mag, phase):
        B, C, F_all, T = audio_fea.shape
        emb = self.conv1x1_1(audio_fea.reshape(B * C, F_all, T))
        emb = self.audio_block_1(emb)
        emb = self.tac1(emb.view(B, C, emb.shape[-2], T))
        emb = self.audio_block_2(emb.view(B * C, emb.shape[-2], T))
        emb = self.tac2(emb.view(B, C, emb.shape[-2], T))
        emb = self.audio_block_3(emb.view(B * C, emb.shape[-2], T))
        emb = self.tac3(emb.view(B, C, emb.shape[-2], T))
        emb = emb.mean(dim=1)
        emb = self.fusion_block(emb)

        mask_real = self.conv1d_real(emb)
        mask_imag = self.conv1d_imag(emb)

        real = mag[:, 0] * torch.cos(phase[:, 0])
        imag = mag[:, 0] * torch.sin(phase[:, 0])
        est_real = mask_real * real - mask_imag * imag
        est_imag = mask_real * imag + mask_imag * real
        est_imag = est_imag + 1e-10
        est_mag = (est_real ** 2 + est_imag ** 2) ** 0.5
        est_phase = torch.atan2(est_imag, est_real)
        enhanced = self.istft(est_mag, est_phase, squeeze=True)
        return [enhanced]

    def check_forward_args(self, all_x):
        x = all_x[0]
        directions = all_x[1] if len(all_x) > 1 else None
        spk_num = all_x[2]
        seq_len = all_x[3]
        geometry = all_x[4]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if directions is not None and directions.dim() == 1:
            directions = directions.unsqueeze(0)
        if spk_num.dim() == 0:
            spk_num = spk_num.unsqueeze(0)
        return x, directions, spk_num, seq_len, geometry

    def forward(self, all_x):
        x, directions, spk_num, seq_len, geometry = self.check_forward_args(all_x)

        if self.training_phase == 1:
            base_df, mag, phase, ipd_r, ipd_i = self.df_computer.forward_base(
                x, geometry, ln_lps=self.ln_LPS_doa
            )
            doa_logits = self._forward_doa_branch(base_df)
            dummy = x.new_zeros(x.shape[0], x.shape[-1])
            return [dummy], doa_logits

        # Phase 2: dual LN, one STFT
        base_doa, base_sep, mag, phase, ipd_r, ipd_i = (
            self.df_computer.forward_base_dual_ln(
                x, geometry, self.ln_LPS_doa, self.ln_LPS_sep
            )
        )
        doa_logits = self._forward_doa_branch(base_doa)

        est_angle = logits_to_angle(
            doa_logits,
            self.num_doa_bins,
            hard=not self.training,
            angle_range_deg=self.doa_angle_range_deg,
        )
        directions_for_af = build_target_directions(est_angle, self.max_nspk)
        AF = self.df_computer.compute_AF(
            directions_for_af, ipd_r, ipd_i, spk_num, geometry
        )
        audio_fea = torch.cat([base_sep, AF], dim=2)
        enhanced_wav = self._forward_separation(audio_fea, mag, phase)
        return enhanced_wav, doa_logits
