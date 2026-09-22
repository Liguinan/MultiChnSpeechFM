#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/1/15 14:52
# @Author  : Guinan Li
# @File    : adapter_moe.py
import torch
import copy
import math
from torch import nn
from activations import ACT2FN


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward with Wav2Vec2->WavLM
class WavLMFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act='gelu'):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(0.1)

        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(0.1)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

    def forward(self, hidden_states):
        # import pdb; pdb.set_trace()
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# class SpeakerAdapter(nn.Module):
#     def __init__(self, hidden_size, intermediate_size, ada_num):
#         super().__init__()
#         self.intermediate_size = intermediate_size
#         self.ada_feed_forward_list = nn.ModuleList([WavLMFeedForward(hidden_size, intermediate_size) for _ in range(ada_num)])
#         self.ada_layer_norm_list = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-5) for _ in range(ada_num)])
#
#     def forward(self, hidden_states, expert_id):
#         '''
#         hidden_states: B x F x T
#         return output: B x Fx T
#         '''
#         # import pdb; pdb.set_trace()
#         hidden_states = hidden_states.permute(0, 2, 1)
#         output = self.ada_layer_norm_list[expert_id](self.ada_feed_forward_list[expert_id](hidden_states))
#         # or resudual connection
#         output = hidden_states + output
#
#         output = output.permute(0, 2, 1)
#
#         return output
#
#     def reset_parameters(self):
#         # 重置所有的HubertFeedForward层
#         for module in self.ada_feed_forward_list:
#             module.reset_parameters()


class SpeakerAdapter(nn.Module):
    def __init__(self, hidden_size, intermediate_size, ada_num):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.ada_feed_forward_list = nn.ModuleList(
            [WavLMFeedForward(hidden_size, intermediate_size) for _ in range(ada_num)])
        self.ada_layer_norm_list = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-5) for _ in range(ada_num)])
        self.ada_num = ada_num

    def forward(self, hidden_states, moe_weights):
        '''
        hidden_states: B x F x T
        moe_weights: 1 x ada_num
        return B x F x T
        '''
        # import pdb; pdb.set_trace()
        hidden_states = hidden_states.permute(0, 2, 1)
        output = []
        output_inner = []
        for i in range(self.ada_num):
            i_output_1 = self.ada_layer_norm_list[i](self.ada_feed_forward_list[i](hidden_states))
            # todo check 是否要去掉
            i_output = hidden_states + i_output_1
            output_inner.append(i_output_1)
            output.append(i_output)
            
        # -> ada_num x B x T x F
        # import pdb; pdb.set_trace()
        # print(f"i_output_1")
        all_output_inner = torch.stack(output_inner)
        all_output = torch.stack(output)
        moe_weights = moe_weights.unsqueeze(-1).unsqueeze(-1)
        moe_weights = moe_weights.transpose(0, 1) # ada_num x B x 1 x 1
        weighted_hidden_states = moe_weights * all_output
        weighted_sum_hidden_states = weighted_hidden_states.sum(dim=0)
        weighted_sum_hidden_states = weighted_sum_hidden_states.permute(0, 2, 1)
        return weighted_sum_hidden_states, all_output_inner

    def reset_parameters(self):
        # 重置所有的HubertFeedForward层
        for module in self.ada_feed_forward_list:
            module.reset_parameters()

class Moe_Weight(nn.Module):
    def __init__(self, geometry_num, ada_num):
        super().__init__()
        self.geometry_num = geometry_num
        self.moe_paramter = nn.Parameter(torch.zeros(geometry_num, ada_num))
        nn.init.kaiming_uniform_(self.moe_paramter, a=math.sqrt(5))
        print(f"self.moe_paramter shape: {self.moe_paramter.shape}")

    def forward(self, geometry_id, dtype, device):
        # import pdb; pdb.set_trace()
        # spk_one_hot = torch.nn.functional.one_hot(torch.tensor(geometry_id), num_classes=self.geometry_num).to(dtype).to(device)
        spk_one_hot = torch.nn.functional.one_hot(geometry_id.clone().detach(), num_classes=self.geometry_num).to(dtype).to(device)
        spk_one_hot = spk_one_hot.reshape(1, -1)
        '''
        if self.training:
            selectd_hidden_states = nn.functional.gumbel_softmax(spk_one_hot @ self.moe_paramter, tau=temperature, hard=False, dim=1)
        else:
            selectd_hidden_states = nn.functional.softmax((spk_one_hot @ self.moe_paramter) / temperature, dim=1)
        '''
        # selectd_hidden_states = spk_one_hot @ self.moe_paramter
        # print(f"no softmax for moe")
        # if geometry_id == 1:
        #     import pdb; pdb.set_trace()
            
        selectd_hidden_states = nn.functional.softmax((spk_one_hot @ self.moe_paramter), dim=1)


        return selectd_hidden_states