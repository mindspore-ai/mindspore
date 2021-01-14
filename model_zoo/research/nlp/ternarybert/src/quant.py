# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Quantization function."""

from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import nn


class QuantizeWeightCell(nn.Cell):
    """
    The ternary fake quant op for weight.

    Args:
        num_bits (int): The bit number of quantization, supporting 2 to 8 bits. Default: 2.
        compute_type (:class:`mindspore.dtype`): Compute type in QuantizeWeightCell. Default: mstype.float32.
        clip_value (float): Clips weight to be in [-clip_value, clip_value].
        per_channel (bool): Quantization granularity based on layer or on channel. Default: False.

    Inputs:
        - **weight** (Parameter) - Parameter of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Parameter of shape :math:`(N, C_{out}, H_{out}, W_{out})`.
    """

    def __init__(self, num_bits=8, compute_type=mstype.float32, clip_value=1.0, per_channel=False):
        super(QuantizeWeightCell, self).__init__()
        self.num_bits = num_bits
        self.compute_type = compute_type
        self.clip_value = clip_value
        self.per_channel = per_channel

        self.clamp = C.clip_by_value
        self.abs = P.Abs()
        self.sum = P.ReduceSum()
        self.nelement = F.size
        self.div = P.Div()
        self.cast = P.Cast()
        self.max = P.ReduceMax()
        self.min = P.ReduceMin()
        self.round = P.Round()

    def construct(self, weight):
        """quantize weight cell"""
        tensor = self.clamp(weight, -self.clip_value, self.clip_value)
        if self.num_bits == 2:
            if self.per_channel:
                n = self.nelement(tensor[0])
                m = self.div(self.sum(self.abs(tensor), 1), n)
                thres = 0.7 * m
                pos = self.cast(tensor[:] > thres[0], self.compute_type)
                neg = self.cast(tensor[:] < -thres[0], self.compute_type)
                mask = self.cast(self.abs(tensor)[:] > thres[0], self.compute_type)
                alpha = self.reshape(self.sum(self.abs(mask * tensor), 1) / self.sum(mask, 1), (-1, 1))
                output = alpha * pos - alpha * neg
            else:
                n = self.nelement(tensor)
                m = self.div(self.sum(self.abs(tensor)), n)
                thres = 0.7 * m
                pos = self.cast(tensor > thres, self.compute_type)
                neg = self.cast(tensor < -thres, self.compute_type)
                mask = self.cast(self.abs(tensor) > thres, self.compute_type)
                alpha = self.sum(self.abs(mask * self.cast(tensor, self.compute_type))) / self.sum(mask)
                output = alpha * pos - alpha * neg
        else:
            tensor_max = self.cast(self.max(tensor), self.compute_type)
            tensor_min = self.cast(self.min(tensor), self.compute_type)
            s = (tensor_max - tensor_min) / (2 ** self.cast(self.num_bits, self.compute_type) - 1)
            output = self.round(self.div(tensor - tensor_min, s)) * s + tensor_min
        return output


class QuantizeWeight:
    """
    Quantize weight into specified bit.

    Args:
        num_bits (int): The bit number of quantization, supporting 2 to 8 bits. Default: 2.
        compute_type (:class:`mindspore.dtype`): Compute type in QuantizeWeightCell. Default: mstype.float32.
        clip_value (float): Clips weight to be in [-clip_value, clip_value].
        per_channel (bool): Quantization granularity based on layer or on channel. Default: False.

    Inputs:
        - **weight** (Parameter) - Parameter of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Parameter of shape :math:`(N, C_{out}, H_{out}, W_{out})`.
    """

    def __init__(self, num_bits=2, compute_type=mstype.float32, clip_value=1.0, per_channel=False):
        self.num_bits = num_bits
        self.compute_type = compute_type
        self.clip_value = clip_value
        self.per_channel = per_channel

        self.clamp = C.clip_by_value
        self.abs = P.Abs()
        self.sum = P.ReduceSum()
        self.nelement = F.size
        self.div = P.Div()
        self.cast = P.Cast()
        self.max = P.ReduceMax()
        self.min = P.ReduceMin()
        self.floor = P.Floor()

    def construct(self, weight):
        """quantize weight"""
        tensor = self.clamp(weight, -self.clip_value, self.clip_value)
        if self.num_bits == 2:
            if self.per_channel:
                n = self.nelement(tensor[0])
                m = self.div(self.sum(self.abs(tensor), 1), n)
                thres = 0.7 * m
                pos = self.cast(tensor[:] > thres[0], self.compute_type)
                neg = self.cast(tensor[:] < -thres[0], self.compute_type)
                mask = self.cast(self.abs(tensor)[:] > thres[0], self.compute_type)
                alpha = self.reshape(self.sum(self.abs(mask * tensor), 1) / self.sum(mask, 1), (-1, 1))
                output = alpha * pos - alpha * neg
            else:
                n = self.nelement(tensor)
                m = self.div(self.sum(self.abs(tensor)), n)
                thres = 0.7 * m
                pos = self.cast(tensor > thres, self.compute_type)
                neg = self.cast(tensor < -thres, self.compute_type)
                mask = self.cast(self.abs(tensor) > thres, self.compute_type)
                alpha = self.sum(self.abs(mask * tensor)) / self.sum(mask)
                output = alpha * pos - alpha * neg
        else:
            tensor_max = self.max(tensor)
            tensor_min = self.min(tensor)
            s = (tensor_max - tensor_min) / (2 ** self.num_bits - 1)
            output = self.floor(self.div((tensor - tensor_min), s) + 0.5) * s + tensor_min
        return output


def convert_network(network, embedding_bits=2, weight_bits=2, clip_value=1.0):
    quantize_embedding = QuantizeWeight(num_bits=embedding_bits, clip_value=clip_value)
    quantize_weight = QuantizeWeight(num_bits=weight_bits, clip_value=clip_value)
    for name, param in network.parameters_and_names():
        if 'bert_embedding_lookup' in name and 'min' not in name and 'max' not in name:
            quantized_param = quantize_embedding.construct(param)
            param.set_data(quantized_param)
        elif 'weight' in name and 'dense_1' not in name:
            quantized_param = quantize_weight.construct(param)
            param.set_data(quantized_param)


def save_params(network):
    return {name: Parameter(param, 'saved_params') for name, param in network.parameters_and_names()}


def restore_params(network, params_dict):
    for name, param in network.parameters_and_names():
        param.set_data(params_dict[name])
