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
"""Transition"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore import Parameter
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from .initializer import lecun_init
from .mask import MaskedLayerNorm
from tests.st.mindscience.mindsponge.mindsponge.common.utils import _memory_reduce


class Transition(nn.Cell):
    r"""
    This is 2-layer MLP where the intermediate layer expands number of channels
    of the input by a factor(num_intermediate_factor).
    """

    def __init__(self, num_intermediate_factor, input_dim, batch_size=None, slice_num=0):
        super(Transition, self).__init__()
        self.matmul = P.MatMul(transpose_b=True)
        self.input_dim = input_dim
        self.num_intermediate = int(input_dim * num_intermediate_factor)
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.relu = nn.ReLU()
        self.idx = Tensor(0, mstype.int32)
        self.masked_layer_norm = MaskedLayerNorm()
        self._init_parameter()

    def construct(self, act, index=None, mask=None):
        '''Compute transition'''
        if self.batch_size:
            input_layer_norm_gamma = P.Gather()(self.input_layer_norm_gammas, index, 0)
            input_layer_norm_beta = P.Gather()(self.input_layer_norm_betas, index, 0)
            transition1_weight = P.Gather()(self.transition1_weights, index, 0)
            transition1_bias = P.Gather()(self.transition1_biases, index, 0)
            transition2_weight = P.Gather()(self.transition2_weights, index, 0)
            transition2_bias = P.Gather()(self.transition2_biases, index, 0)
        else:
            input_layer_norm_gamma = self.input_layer_norm_gammas
            input_layer_norm_beta = self.input_layer_norm_betas
            transition1_weight = self.transition1_weights
            transition1_bias = self.transition1_biases
            transition2_weight = self.transition2_weights
            transition2_bias = self.transition2_biases
        act = self.masked_layer_norm(act, input_layer_norm_gamma, input_layer_norm_beta, mask=mask)
        batched_inputs = (act,)
        nonbatched_inputs = (transition1_weight, transition1_bias, transition2_weight, transition2_bias)
        act = _memory_reduce(self._compute, batched_inputs, nonbatched_inputs, self.slice_num)
        return act

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.input_layer_norm_gammas = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim)), mstype.float32))
            self.input_layer_norm_betas = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim)), mstype.float32))
            self.transition1_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate, self.input_dim)), mstype.float32))
            self.transition1_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate)), mstype.float32))
            self.transition2_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim, self.num_intermediate)), mstype.float32))
            self.transition2_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim)), mstype.float32))
        else:
            self.input_layer_norm_gammas = Parameter(Tensor(np.ones((self.input_dim)), mstype.float32))
            self.input_layer_norm_betas = Parameter(Tensor(np.zeros((self.input_dim)), mstype.float32))
            self.transition1_weights = Parameter(initializer(lecun_init(self.input_dim, initializer_name='relu'),
                                                             [self.num_intermediate, self.input_dim]))
            self.transition1_biases = Parameter(Tensor(np.zeros((self.num_intermediate)), mstype.float32))
            self.transition2_weights = Parameter(
                Tensor(np.zeros((self.input_dim, self.num_intermediate)), mstype.float32))
            self.transition2_biases = Parameter(Tensor(np.zeros((self.input_dim)), mstype.float32))

    def _compute(self, act, transition1_weight, transition1_bias, transition2_weight, transition2_bias):
        '''compute transition.'''

        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        act = self.relu(P.BiasAdd()(self.matmul(act, transition1_weight), transition1_bias))
        act = P.BiasAdd()(self.matmul(act, transition2_weight), transition2_bias)
        act = P.Reshape()(act, act_shape)
        return act
