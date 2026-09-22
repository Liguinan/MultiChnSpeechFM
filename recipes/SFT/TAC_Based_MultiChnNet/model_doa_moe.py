#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ModelDOAMoE: v2 DOA branch (pred-AF) + MoE separation backbone.

Forward:
  dual LN STFT -> DOA logits -> pred angle AF -> MoE TAC sep -> wav
Returns:
  (enhanced_list, adapter_outputs, geometry_logits, doa_logits)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_zen.model.base_model import BaseModel
from audio_feature_doa import DFComputer, iSTFT, ChannelWiseLayerNorm
from adapter_moe import SpeakerAdapter, Moe_Weight
from tacnet import TAC
from model import Conv1D, Conv1DBlock, build_norm


def angle_rad_to_bin(angle_rad, num_bins, angle_range_deg=180.0):
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


class ModelDOAMoE(BaseModel):
    def __init__(
        self,
        norm,
        out_spk,
        non_linear,
        causal,
        input_features,
        cosIPD,
        sinIPD,
        N,
        V,
        X,
        B,
        H,
        P,
        R,
        HOP_SIZE,
        FFT_SIZE,
        NFFT,
        merge_mode,
        speaker_feature_dim,
        sr,
        AF_premasking,
        TCN_layers,
        add_adapter_pos,
        ada_num,
        geometry_num,
        ce_loss,
        num_doa_bins=12,
        doa_angle_range_deg=180.0,
        max_nspk=3,
        onset_pool_ratio=1.0,
        use_pred_doa_for_af=True,
    ):
        super().__init__()
        self.num_doa_bins = int(num_doa_bins)
        self.doa_angle_range_deg = float(doa_angle_range_deg)
        self.max_nspk = int(max_nspk)
        self.onset_pool_ratio = onset_pool_ratio
        self.use_pred_doa_for_af = bool(use_pred_doa_for_af)
        self.out_spk = out_spk
        self.add_adapter_pos = add_adapter_pos
        self.ce_loss = ce_loss
        self.ada_num = ada_num

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
        if hasattr(self.df_computer, "ln_LPS"):
            for p in self.df_computer.ln_LPS.parameters():
                p.requires_grad = False

        dim_no_af = self.df_computer.df_dim_no_af
        dim_full = self.df_computer.df_dim

        # ---- DOA branch (from v2) ----
        self.ln_LPS_doa = ChannelWiseLayerNorm(self.df_computer.num_bins)
        self.conv1x1_pre = Conv1D(dim_no_af, B, 1)
        self.audio_block_1_doa = self._build_repeats(1, X, B, H, P, norm, causal)
        self.tac1_doa = TAC(B, B, B)
        self.audio_block_2_doa = self._build_repeats(1, X, B, H, P, norm, causal)
        self.tac2_doa = TAC(B, B, B)
        self.doa_head = nn.Linear(B, self.num_doa_bins)

        # ---- Sep branch + MoE adapters ----
        self.ln_LPS_sep = ChannelWiseLayerNorm(self.df_computer.num_bins)
        self.conv1x1_1 = Conv1D(dim_full, B, 1)

        if self.add_adapter_pos == -1:
            self.expert_adapter_1 = SpeakerAdapter(256, 128, ada_num=ada_num)
            self.moe_weights_1 = Moe_Weight(geometry_num=geometry_num, ada_num=ada_num)
            self.expert_adapter_2 = SpeakerAdapter(256, 128, ada_num=ada_num)
            self.moe_weights_2 = Moe_Weight(geometry_num=geometry_num, ada_num=ada_num)
            self.expert_adapter_3 = SpeakerAdapter(256, 128, ada_num=ada_num)
            self.moe_weights_3 = Moe_Weight(geometry_num=geometry_num, ada_num=ada_num)
            self.expert_adapter_4 = SpeakerAdapter(256, 128, ada_num=ada_num)
            self.moe_weights_4 = Moe_Weight(geometry_num=geometry_num, ada_num=ada_num)
            self.expert_adapter_5 = SpeakerAdapter(256, 128, ada_num=ada_num)
            self.moe_weights_5 = Moe_Weight(geometry_num=geometry_num, ada_num=ada_num)
            self.expert_adapter_6 = SpeakerAdapter(256, 128, ada_num=ada_num)
            self.moe_weights_6 = Moe_Weight(geometry_num=geometry_num, ada_num=ada_num)

        if self.ce_loss:
            for i in range(1, 7):
                setattr(self, f"geometry_projector_{i}", nn.Linear(256, 128))
                setattr(self, f"geometry_head_{i}", nn.Linear(128, ada_num))

        self.audio_block_1 = self._build_repeats(1, X, B, H, P, norm, causal)
        self.tac1 = TAC(B, B, B)
        self.audio_block_2 = self._build_repeats(1, X, B, H, P, norm, causal)
        self.tac2 = TAC(B, B, B)
        self.audio_block_3 = self._build_repeats(1, X, B, H, P, norm, causal)
        self.tac3 = TAC(B, B, B)
        self.fusion_block_4 = self._build_repeats(1, X, B, H, P, norm, causal)
        self.fusion_block_5 = self._build_repeats(1, X, B, H, P, norm, causal)
        self.fusion_block_6 = self._build_repeats(1, X, B, H, P, norm, causal)
        self.conv1d_real = Conv1D(B, out_spk * self.df_computer.num_bins, 1)
        self.conv1d_imag = Conv1D(B, out_spk * self.df_computer.num_bins, 1)
        self.istft = iSTFT(frame_len=FFT_SIZE, frame_hop=HOP_SIZE, num_fft=FFT_SIZE)

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

    def _pool_for_doa(self, emb):
        emb = emb.mean(dim=1)
        T = emb.shape[-1]
        ratio = self.onset_pool_ratio
        if ratio is not None and 0.0 < float(ratio) < 1.0:
            t_keep = max(1, int(T * float(ratio)))
            emb = emb[..., :t_keep]
        return emb.mean(dim=-1)

    def _forward_doa_branch(self, base_df):
        B, C, F_base, T = base_df.shape
        emb = self.conv1x1_pre(base_df.reshape(B * C, F_base, T))
        emb = self.audio_block_1_doa(emb)
        emb = self.tac1_doa(emb.view(B, C, emb.shape[-2], T))
        emb = self.audio_block_2_doa(emb.view(B * C, emb.shape[-2], T))
        emb = self.tac2_doa(emb.view(B, C, emb.shape[-2], T))
        return self.doa_head(self._pool_for_doa(emb))

    def _geometry_logits_from(self, projector, head, audio_emb, seq_len, B, C, after_mean=False):
        geometry_type_hidden_states = projector(audio_emb.permute(0, 2, 1))
        if seq_len is None:
            pooled = geometry_type_hidden_states.mean(dim=1)
        else:
            if after_mean:
                sl = seq_len
            else:
                sl = seq_len.repeat_interleave(C, dim=0)
            T = geometry_type_hidden_states.size(1)
            mask = torch.arange(T, device=geometry_type_hidden_states.device).unsqueeze(0) < sl.unsqueeze(1)
            masked = geometry_type_hidden_states * mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / sl.float().unsqueeze(-1)
        logits = head(pooled)
        if not after_mean:
            logits = logits.reshape(B, C, -1).mean(dim=1)
        return logits

    def _forward_moe_sep(self, audio_fea, mag, phase, seq_len, expert_id):
        B, C, F_all, T = audio_fea.shape
        audio_emb = self.conv1x1_1(audio_fea.reshape(B * C, F_all, T))
        geometry_logits = []

        audio_emb = self.audio_block_1(audio_emb)
        moe_param = self.moe_weights_1(expert_id, audio_emb.dtype, audio_emb.device)
        audio_emb, a1 = self.expert_adapter_1(audio_emb, moe_param)
        if self.ce_loss:
            geometry_logits.append(
                self._geometry_logits_from(
                    self.geometry_projector_1, self.geometry_head_1, audio_emb, seq_len, B, C
                )
            )
        audio_emb = self.tac1(audio_emb.view(B, C, audio_emb.shape[-2], -1))
        audio_emb = audio_emb.view(B * C, audio_emb.shape[-2], -1)

        audio_emb = self.audio_block_2(audio_emb)
        moe_param = self.moe_weights_2(expert_id, audio_emb.dtype, audio_emb.device)
        audio_emb, a2 = self.expert_adapter_2(audio_emb, moe_param)
        if self.ce_loss:
            geometry_logits.append(
                self._geometry_logits_from(
                    self.geometry_projector_2, self.geometry_head_2, audio_emb, seq_len, B, C
                )
            )
        audio_emb = self.tac2(audio_emb.view(B, C, audio_emb.shape[-2], -1))
        audio_emb = audio_emb.view(B * C, audio_emb.shape[-2], -1)

        audio_emb = self.audio_block_3(audio_emb)
        moe_param = self.moe_weights_3(expert_id, audio_emb.dtype, audio_emb.device)
        audio_emb, a3 = self.expert_adapter_3(audio_emb, moe_param)
        if self.ce_loss:
            geometry_logits.append(
                self._geometry_logits_from(
                    self.geometry_projector_3, self.geometry_head_3, audio_emb, seq_len, B, C
                )
            )
        audio_emb = self.tac3(audio_emb.view(B, C, audio_emb.shape[-2], -1))
        audio_emb = audio_emb.mean(dim=1)

        audio_emb = self.fusion_block_4(audio_emb)
        moe_param = self.moe_weights_4(expert_id, audio_emb.dtype, audio_emb.device)
        audio_emb, a4 = self.expert_adapter_4(audio_emb, moe_param)
        if self.ce_loss:
            geometry_logits.append(
                self._geometry_logits_from(
                    self.geometry_projector_4, self.geometry_head_4, audio_emb, seq_len, B, C, after_mean=True
                )
            )

        audio_emb = self.fusion_block_5(audio_emb)
        moe_param = self.moe_weights_5(expert_id, audio_emb.dtype, audio_emb.device)
        audio_emb, a5 = self.expert_adapter_5(audio_emb, moe_param)
        if self.ce_loss:
            geometry_logits.append(
                self._geometry_logits_from(
                    self.geometry_projector_5, self.geometry_head_5, audio_emb, seq_len, B, C, after_mean=True
                )
            )

        audio_emb = self.fusion_block_6(audio_emb)
        moe_param = self.moe_weights_6(expert_id, audio_emb.dtype, audio_emb.device)
        audio_emb, a6 = self.expert_adapter_6(audio_emb, moe_param)
        if self.ce_loss:
            geometry_logits.append(
                self._geometry_logits_from(
                    self.geometry_projector_6, self.geometry_head_6, audio_emb, seq_len, B, C, after_mean=True
                )
            )

        mask_real = self.conv1d_real(audio_emb)
        mask_imag = self.conv1d_imag(audio_emb)
        real = mag[:, 0] * torch.cos(phase[:, 0])
        imag = mag[:, 0] * torch.sin(phase[:, 0])
        est_real = mask_real * real - mask_imag * imag
        est_imag = mask_real * imag + mask_imag * real + 1e-10
        est_mag = (est_real ** 2 + est_imag ** 2) ** 0.5
        est_phase = torch.atan2(est_imag, est_real)
        enhanced = [self.istft(est_mag, est_phase, squeeze=True)]
        adapter_outputs = [a1, a2, a3, a4, a5, a6]
        return enhanced, adapter_outputs, (geometry_logits if self.ce_loss else None)

    def check_forward_args(self, all_x):
        x = all_x[0]
        directions = all_x[1] if len(all_x) > 1 else None
        spk_num = all_x[2]
        seq_len = all_x[3]
        geometry = all_x[4]
        expert_id = all_x[5] if len(all_x) > 5 else torch.tensor(0, device=x.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if directions is not None and directions.dim() == 1:
            directions = directions.unsqueeze(0)
        if spk_num.dim() == 0:
            spk_num = spk_num.unsqueeze(0)
        return x, directions, spk_num, seq_len, geometry, expert_id

    def forward(self, all_x):
        x, directions, spk_num, seq_len, geometry, expert_id = self.check_forward_args(all_x)

        base_doa, base_sep, mag, phase, ipd_r, ipd_i = self.df_computer.forward_base_dual_ln(
            x, geometry, self.ln_LPS_doa, self.ln_LPS_sep
        )
        doa_logits = self._forward_doa_branch(base_doa)

        if self.use_pred_doa_for_af:
            est_angle = logits_to_angle(
                doa_logits,
                self.num_doa_bins,
                hard=not self.training,
                angle_range_deg=self.doa_angle_range_deg,
            )
            directions_for_af = build_target_directions(est_angle, self.max_nspk)
        else:
            directions_for_af = directions

        AF = self.df_computer.compute_AF(
            directions_for_af, ipd_r, ipd_i, spk_num, geometry
        )
        audio_fea = torch.cat([base_sep, AF], dim=2)
        enhanced, adapter_outputs, geometry_logits = self._forward_moe_sep(
            audio_fea, mag, phase, seq_len, expert_id
        )
        return enhanced, adapter_outputs, geometry_logits, doa_logits
