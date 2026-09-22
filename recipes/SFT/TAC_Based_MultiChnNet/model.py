#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/1
# @Author  : Guinan Li
# @File    : model.py

import torch
from torch.nn import functional
from audio_zen.acoustics.feature import drop_band
from audio_zen.model.base_model import BaseModel
from audio_zen.model.module.sequence_model import SequenceModel
import torch.nn as nn
import torch.nn.functional as F
from audio_feature import DFComputer, iSTFT
from adapter_moe import SpeakerAdapter, Moe_Weight
from tacnet import TAC


class Model(BaseModel):
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
        ce_loss
    ):
        super(Model, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax,
            "linear": None
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear  # string

        self.df_computer = DFComputer(frame_hop=HOP_SIZE,
                                      frame_len=FFT_SIZE,
                                      NFFT=NFFT,
                                      in_feature=input_features,
                                      merge_mode=merge_mode,
                                      cosIPD=cosIPD,
                                      sinIPD=sinIPD,
                                      speaker_feature_dim=speaker_feature_dim,
                                      sr=sr,
                                      AF_premasking=AF_premasking)

        dim_conv = self.df_computer.df_dim
        self.conv1x1_1 = Conv1D(dim_conv, B, 1)

        self.add_adapter_pos = add_adapter_pos
        if self.add_adapter_pos == -1:
            # self.expert_adapter = SpeakerAdapter(hidden_size=256, intermediate_size=128, ada_num=ada_num)
            self.expert_adapter_1 = SpeakerAdapter(hidden_size=256, intermediate_size=128, ada_num=ada_num)
            self.moe_weights_1 = Moe_Weight(geometry_num=geometry_num, ada_num=ada_num)
            self.expert_adapter_2 = SpeakerAdapter(hidden_size=256, intermediate_size=128, ada_num=ada_num)
            self.moe_weights_2 = Moe_Weight(geometry_num=geometry_num, ada_num=ada_num)
            self.expert_adapter_3 = SpeakerAdapter(hidden_size=256, intermediate_size=128, ada_num=ada_num)
            self.moe_weights_3 = Moe_Weight(geometry_num=geometry_num, ada_num=ada_num)
            self.expert_adapter_4 = SpeakerAdapter(hidden_size=256, intermediate_size=128, ada_num=ada_num)
            self.moe_weights_4 = Moe_Weight(geometry_num=geometry_num, ada_num=ada_num)
            self.expert_adapter_5 = SpeakerAdapter(hidden_size=256, intermediate_size=128, ada_num=ada_num)
            self.moe_weights_5 = Moe_Weight(geometry_num=geometry_num, ada_num=ada_num)
            self.expert_adapter_6 = SpeakerAdapter(hidden_size=256, intermediate_size=128, ada_num=ada_num)
            self.moe_weights_6 = Moe_Weight(geometry_num=geometry_num, ada_num=ada_num)
        
        self.ce_loss = ce_loss
        if self.ce_loss:
            self.geometry_projector_1 = nn.Linear(256, 128)
            self.geometry_head_1 = nn.Linear(128, ada_num)

            self.geometry_projector_2 = nn.Linear(256, 128)
            self.geometry_head_2 = nn.Linear(128, ada_num)

            self.geometry_projector_3 = nn.Linear(256, 128)
            self.geometry_head_3 = nn.Linear(128, ada_num)

            self.geometry_projector_4 = nn.Linear(256, 128)
            self.geometry_head_4 = nn.Linear(128, ada_num)

            self.geometry_projector_5 = nn.Linear(256, 128)
            self.geometry_head_5 = nn.Linear(128, ada_num)

            self.geometry_projector_6 = nn.Linear(256, 128)
            self.geometry_head_6 = nn.Linear(128, ada_num)

        
        # if self.add_adapter_pos == -1:
        #     self.adapter_pos = torch.zeros(TCN_layers)
        # else:
        #     self.adapter_pos = torch.nn.functional.one_hot(torch.tensor(add_adapter_pos), num_classes=TCN_layers)

        self.audio_block_1 = self._build_repeats(
            num_repeats=1,
            num_blocks=X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)
        self.tac1 = TAC(B, B, B)

        self.audio_block_2 = self._build_repeats(
            num_repeats=1,
            num_blocks=X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)
        self.tac2 = TAC(B, B, B)

        self.audio_block_3 = self._build_repeats(
            num_repeats=1,
            num_blocks=X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)
        self.tac3 = TAC(B, B, B)

        self.fusion_block_4 = self._build_repeats(
            num_repeats=1,
            num_blocks=X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)

        self.fusion_block_5 = self._build_repeats(
            num_repeats=1,
            num_blocks=X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)

        self.fusion_block_6 = self._build_repeats(
            num_repeats=1,
            num_blocks=X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)

        self.conv1d_real = Conv1D(B, out_spk * self.df_computer.num_bins, 1)
        self.conv1d_imag = Conv1D(B, out_spk * self.df_computer.num_bins, 1)

        self.non_linear = supported_nonlinear[non_linear]  # activation function instance

        self.istft = iSTFT(frame_len=FFT_SIZE, frame_hop=HOP_SIZE, num_fft=FFT_SIZE)

        self.out_spk = out_spk

    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [Conv1DBlock(dilation=(2 ** b), **block_kwargs) for b in range(num_blocks)]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        """
        Build Conv1D block repeats
        """
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)
        ]
        return nn.Sequential(*repeats)

    def check_forward_args(self, all_x):
        x = all_x[0]
        directions = all_x[1]
        spk_num = all_x[2]
        seq_len = all_x[3]
        geometry = all_x[4]
        expert_id = all_x[5]


        if x.dim() == 2:
            x = torch.unsqueeze(x, 0)
        if directions.dim() == 1:
            directions = torch.unsqueeze(directions, 0)
        if spk_num.dim() == 0:
            spk_num = torch.unsqueeze(spk_num, 0)

        return x, directions, spk_num, seq_len, geometry, expert_id

    def forward(self, all_x):
        '''
        Input params: all_x:
        [0] x - multi-channel mixture wav, shape: [B, C, t]
        [1] directions - all speakers' directions relative to microphone array, shape: [B, spk_num]
        [2] spk_num - actual speaker number in current wav shape: [B]
        [3] seq_len - frame length of wav, shape: [B]
        [4] geometry - array geometry of each batch, shape: [selected_microphone_num_in_array_geometry]

        Output: 
            wav: single-channel separated wav, [B, t]
        '''
        # import pdb; pdb.set_trace()
        x, directions, spk_num, seq_len, geometry, expert_id = self.check_forward_args(all_x)

        audio_fea, mag, phase = self.df_computer(x, directions, spk_num, geometry)
    
        # audio_fea: [B, geometry, 257*3, T]
        B, C, F_all, T = audio_fea.shape

        audio_fea = audio_fea.reshape(B * C, F_all, T)
        # [B*C, F_all, T] -> [B*C, F=256, T]
        audio_emb = self.conv1x1_1(audio_fea)

        # [B*C, F, T] -> [B*C, F, T]
        audio_emb = self.audio_block_1(audio_emb)

        # import pdb; pdb.set_trace()
        # if self.add_adapter_pos == 0:
        #     print(f"self.add_adapter_pos:{self.add_adapter_pos}")
        #     # import pdb; pdb.set_trace()
        moe_param = self.moe_weights_1(expert_id, audio_emb.dtype, audio_emb.device)
        audio_emb, adapter_output_1 = self.expert_adapter_1(audio_emb, moe_param)

        if self.ce_loss:
            geometry_type_hidden_states = self.geometry_projector_1(audio_emb.permute(0,2,1))
            if seq_len is None:
                pooled_output = geometry_type_hidden_states.mean(dim=1)
            else:
                seq_len_tmp = seq_len.repeat_interleave(C, dim=0)
                T = geometry_type_hidden_states.size(1)
                # 构造 mask: (B, T)，有效位置为 True
                mask = torch.arange(T, device=geometry_type_hidden_states.device).unsqueeze(0) < seq_len_tmp.unsqueeze(1)
                # 将无效位置置零，求和后除以有效长度
                masked_hidden = geometry_type_hidden_states * mask.unsqueeze(-1)  # (B, T, F)
                sum_hidden = masked_hidden.sum(dim=1)                            # (B, F)
                valid_len = seq_len_tmp.float().unsqueeze(-1)                        # (B, 1)
                pooled_output = sum_hidden / valid_len
            geometry_logits_1 =self.geometry_head_1(pooled_output)
            geometry_logits_1 = geometry_logits_1.reshape(B, C, -1).mean(dim=1)
        # import pdb; pdb.set_trace()
        # [B*C, F, T] -> [B, C, F, T] -> [B, C, F, T]
        audio_emb = self.tac1(audio_emb.view(B, C, audio_emb.shape[-2], -1))
       
        # [B, C, F, T] -> [B*C, F, T]
        audio_emb = audio_emb.view(B * C, audio_emb.shape[-2], -1)

        # # import pdb; pdb.set_trace()
        # if self.add_adapter_pos == 1:
        #     print(f"self.add_adapter_pos:{self.add_adapter_pos} after TAC")
        #     # moe_param = self.moe_weights(expert_id, audio_emb.dtype, audio_emb.device)
        #     # audio_emb = self.ada_transf(moe_param, moe_param)
        #     audio_emb = self.expert_adapter(audio_emb, expert_id)
        #     # import pdb; pdb.set_trace()

        # [B*C, F, T] -> [B*C, F, T]
        audio_emb = self.audio_block_2(audio_emb)

        moe_param = self.moe_weights_2(expert_id, audio_emb.dtype, audio_emb.device)
        audio_emb, adapter_output_2 = self.expert_adapter_2(audio_emb, moe_param)
 
        if self.ce_loss:
            geometry_type_hidden_states = self.geometry_projector_2(audio_emb.permute(0,2,1))
            if seq_len is None:
                pooled_output = geometry_type_hidden_states.mean(dim=1)
            else:
                # import pdb; pdb.set_trace()
                seq_len_tmp = seq_len.repeat_interleave(C, dim=0)
                T = geometry_type_hidden_states.size(1)
                # 构造 mask: (B, T)，有效位置为 True
                mask = torch.arange(T, device=geometry_type_hidden_states.device).unsqueeze(0) < seq_len_tmp.unsqueeze(1)
                # 将无效位置置零，求和后除以有效长度
                masked_hidden = geometry_type_hidden_states * mask.unsqueeze(-1)  # (B, T, F)
                sum_hidden = masked_hidden.sum(dim=1)                            # (B, F)
                valid_len = seq_len_tmp.float().unsqueeze(-1)                        # (B, 1)
                pooled_output = sum_hidden / valid_len
            geometry_logits_2 =self.geometry_head_2(pooled_output)
            geometry_logits_2 = geometry_logits_2.reshape(B, C, -1).mean(dim=1)

        # [B*C, F, T] -> [B, C, F, T] -> [B, C, F, T]
        audio_emb = self.tac2(audio_emb.view(B, C, audio_emb.shape[-2], -1))
        
        # [B, C, F, T] -> [B*C, F, T]
        audio_emb = audio_emb.view(B * C, audio_emb.shape[-2], -1)
        # [B*C, F, T] -> [B*C, F, T]
        audio_emb = self.audio_block_3(audio_emb)

        moe_param = self.moe_weights_3(expert_id, audio_emb.dtype, audio_emb.device)
        audio_emb, adapter_output_3 = self.expert_adapter_3(audio_emb, moe_param)

        if self.ce_loss:
            geometry_type_hidden_states = self.geometry_projector_3(audio_emb.permute(0,2,1))
            if seq_len is None:
                pooled_output = geometry_type_hidden_states.mean(dim=1)
            else:
                seq_len_tmp = seq_len.repeat_interleave(C, dim=0)
                T = geometry_type_hidden_states.size(1)
                # 构造 mask: (B, T)，有效位置为 True
                mask = torch.arange(T, device=geometry_type_hidden_states.device).unsqueeze(0) < seq_len_tmp.unsqueeze(1)
                # 将无效位置置零，求和后除以有效长度
                masked_hidden = geometry_type_hidden_states * mask.unsqueeze(-1)  # (B, T, F)
                sum_hidden = masked_hidden.sum(dim=1)                            # (B, F)
                valid_len = seq_len_tmp.float().unsqueeze(-1)                        # (B, 1)
                pooled_output = sum_hidden / valid_len
            geometry_logits_3 =self.geometry_head_3(pooled_output)
            geometry_logits_3 = geometry_logits_3.reshape(B, C, -1).mean(dim=1)


        # [B*C, F, T] -> [B, C, F, T] -> [B, C, F, T]
        audio_emb = self.tac3(audio_emb.view(B, C, audio_emb.shape[-2], -1))
        
        # [B, C, F, T] -> [B, F, T]
        audio_emb = audio_emb.mean(dim=1)
        
        # [B, F, T] -> [B, F, T] 
        audio_emb = self.fusion_block_4(audio_emb)
        moe_param = self.moe_weights_4(expert_id, audio_emb.dtype, audio_emb.device)
        audio_emb, adapter_output_4 = self.expert_adapter_4(audio_emb, moe_param)

        if self.ce_loss:
            geometry_type_hidden_states = self.geometry_projector_4(audio_emb.permute(0,2,1))
            if seq_len is None:
                pooled_output = geometry_type_hidden_states.mean(dim=1)
            else:
                # seq_len = seq_len.repeat_interleave(C, dim=0)
                T = geometry_type_hidden_states.size(1)
                # 构造 mask: (B, T)，有效位置为 True
                mask = torch.arange(T, device=geometry_type_hidden_states.device).unsqueeze(0) < seq_len.unsqueeze(1)
                # 将无效位置置零，求和后除以有效长度
                masked_hidden = geometry_type_hidden_states * mask.unsqueeze(-1)  # (B, T, F)
                sum_hidden = masked_hidden.sum(dim=1)                            # (B, F)
                valid_len = seq_len.float().unsqueeze(-1)                        # (B, 1)
                pooled_output = sum_hidden / valid_len
            geometry_logits_4 =self.geometry_head_4(pooled_output)

       
        audio_emb = self.fusion_block_5(audio_emb)
        moe_param = self.moe_weights_5(expert_id, audio_emb.dtype, audio_emb.device)
        audio_emb, adapter_output_5 = self.expert_adapter_5(audio_emb, moe_param)

        if self.ce_loss:
            geometry_type_hidden_states = self.geometry_projector_5(audio_emb.permute(0,2,1))
            if seq_len is None:
                pooled_output = geometry_type_hidden_states.mean(dim=1)
            else:
                # seq_len = seq_len.repeat_interleave(C, dim=0)
                T = geometry_type_hidden_states.size(1)
                # 构造 mask: (B, T)，有效位置为 True
                mask = torch.arange(T, device=geometry_type_hidden_states.device).unsqueeze(0) < seq_len.unsqueeze(1)
                # 将无效位置置零，求和后除以有效长度
                masked_hidden = geometry_type_hidden_states * mask.unsqueeze(-1)  # (B, T, F)
                sum_hidden = masked_hidden.sum(dim=1)                            # (B, F)
                valid_len = seq_len.float().unsqueeze(-1)                        # (B, 1)
                pooled_output = sum_hidden / valid_len
            geometry_logits_5 =self.geometry_head_5(pooled_output)
        
        audio_emb = self.fusion_block_6(audio_emb)
        moe_param = self.moe_weights_6(expert_id, audio_emb.dtype, audio_emb.device)
        audio_emb, adapter_output_6 = self.expert_adapter_6(audio_emb, moe_param)

        if self.ce_loss:
            geometry_type_hidden_states = self.geometry_projector_6(audio_emb.permute(0,2,1))
            if seq_len is None:
                pooled_output = geometry_type_hidden_states.mean(dim=1)
            else:
                # seq_len = seq_len.repeat_interleave(C, dim=0)
                T = geometry_type_hidden_states.size(1)
                # 构造 mask: (B, T)，有效位置为 True
                mask = torch.arange(T, device=geometry_type_hidden_states.device).unsqueeze(0) < seq_len.unsqueeze(1)
                # 将无效位置置零，求和后除以有效长度
                masked_hidden = geometry_type_hidden_states * mask.unsqueeze(-1)  # (B, T, F)
                sum_hidden = masked_hidden.sum(dim=1)                            # (B, F)
                valid_len = seq_len.float().unsqueeze(-1)                        # (B, 1)
                pooled_output = sum_hidden / valid_len
            geometry_logits_6 =self.geometry_head_6(pooled_output)

        # # import pdb; pdb.set_trace()
        # if self.add_adapter_pos == 5:
        #     print(f"self.add_adapter_pos:{self.add_adapter_pos}")
        #     # moe_param = self.moe_weights(expert_id, audio_emb.dtype, audio_emb.device)
        #     # audio_emb = self.ada_transf(moe_param, moe_param)
        #     audio_emb = self.expert_adapter(audio_emb, expert_id)
        #     # import pdb; pdb.set_trace()

        # [B, F, T] -> [B, F+1, T]
        mask_real = self.conv1d_real(audio_emb)
        mask_imag = self.conv1d_imag(audio_emb)

        # mixture * mask
        # use the averaged single-channel mixture (idx=-1) as the input for masking calculation
        real = mag[:, 0] * torch.cos(phase[:, 0])
        imag = mag[:, 0] * torch.sin(phase[:, 0])

        # complex masking calculation
        est_real_part = mask_real * real - mask_imag * imag
        est_imag_part = mask_real * imag + mask_imag * real
        est_imag_part = est_imag_part + 1.0e-10

        # magnitude and phase of enhanced wav
        est_mag = (est_real_part ** 2 + est_imag_part ** 2) ** 0.5
        est_phase = torch.atan2(est_imag_part, est_real_part)
        # [B, F+1=257, T] -> [B, t]
        enhanced_wav = [self.istft(est_mag, est_phase, squeeze=True)]

        adapter_outputs = [adapter_output_1, adapter_output_2, adapter_output_3, adapter_output_4, adapter_output_5, adapter_output_6]

        if self.ce_loss:
            geometry_logits = [geometry_logits_1, geometry_logits_2, geometry_logits_3, geometry_logits_4, geometry_logits_5, geometry_logits_6]
        else:
            geometry_logits = None


        return enhanced_wav, adapter_outputs, geometry_logits




class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super(Conv1D, self).forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


# for each 1-D convolutional block (depth-wise separable convolution)
class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=256,  # in_channels=B,
                 conv_channels=512, # conv_channels=H,
                 kernel_size=3,  # kernel_size=P
                 dilation=1,
                 norm="BN",  # norm=norm,
                 causal=False):
        super(Conv1DBlock, self).__init__()

        # <1> 1x1-conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        # <2> prelu
        self.prelu1 = nn.PReLU()
        # <3> normalization
        self.lnorm1 = build_norm(norm, conv_channels)
        # <4> D-conv
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        # <5> prelu
        self.prelu2 = nn.PReLU()
        # <6> normalization
        self.lnorm2 = build_norm(norm, conv_channels)
        # <7> 1x1-conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y  # identity connection
        return x


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: BS x N x K
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # BS x N x K => BS x K x N
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = torch.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
               "elementwise_affine={elementwise_affine}".format(**self.__dict__)


if __name__ == "__main__":
    pass
