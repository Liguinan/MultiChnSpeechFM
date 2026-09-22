#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/6 20:08
# @Author  : Guinan Li
# @File    : pt_tacnet.py

import torch.nn as nn
import torch


class TAC(nn.Module):
    """
   transform-average-concatenate (TAC) applied to each layer/block.

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(TAC, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.ch_transform = nn.Sequential(nn.Linear(input_size, hidden_size * 3),
                                          nn.PReLU()
                                          )
        self.ch_average = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size * 3),
                                        nn.PReLU()
                                        )
        self.ch_concat = nn.Sequential(nn.Linear(hidden_size * 6, output_size),
                                       nn.PReLU()
                                       )
        self.ch_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        """
        input:
        x: shape: (B, C, F, T)

        """
        B, C, F, T = x.shape
        # (B, C, F, T) -> (B, C, T, F)
        x_flat = x.permute(0, 1, 3, 2).contiguous()

        # 1. Transform each channel
        # (B, C, T, F) -> (B, C, T, 3*F)
        transformed = self.ch_transform(x_flat)

        # 2. Average (mean pooling) across channels
        # (B, C, T, 3*F) -> (B, 1, T, 3*F)
        avg_pool = transformed.mean(dim=1, keepdim=True)
        # (B, 1, T, 3*F) -> (B, 1, T, 3*F)
        avg_pool = self.ch_average(avg_pool)

        # 3. Concat
        # (B, 1, T, 3*F) -> (B, C, T, 3*F)
        avg_expanded = avg_pool.expand(-1, C, -1, -1)
        # (B, C, T, 3*F) -> (B, C, T, 3*F*2)
        concat_feat = torch.cat([transformed, avg_expanded], dim=-1)
        # (B, C, T, 3*F*2) -> (B, C, T, F)
        concat_feat = self.ch_concat(concat_feat)

        # Norm
        out = self.ch_norm(concat_feat)
        # (B, C, T, F) -> (B, C, F, T)
        out = out.permute(0, 1, 3, 2).contiguous()

        return x + out
