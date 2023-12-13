# Copyright 2023 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""Mask"""
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.nn as nn


class LayerNormProcess(nn.Cell):
    def __init__(self,):
        super(LayerNormProcess, self).__init__()
        self.layernorm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)

    def construct(self, msa_act, query_norm_gamma, query_norm_beta):
        output, _, _ = self.layernorm(msa_act, query_norm_gamma, query_norm_beta)
        return output


class MaskedLayerNorm(nn.Cell):
    '''masked_layer_norm'''

    def __init__(self):
        super(MaskedLayerNorm, self).__init__()
        self.norm = LayerNormProcess()

    def construct(self, act, gamma, beta, mask=None):
        '''construct'''
        act = act
        gamma = gamma
        beta = beta

        ones = P.Ones()(act.shape[:-1] + (1,), act.dtype)
        if mask is not None:
            mask = F.expand_dims(mask, -1)
            mask = mask * ones
        else:
            mask = ones

        act = act * mask
        act = self.norm(act, gamma, beta)
        act = act * mask
        return act
