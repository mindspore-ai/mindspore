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
"""MSA"""
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from tests.st.mindscience.mindsponge.mindsponge.common.utils import _memory_reduce
from .basic import Attention, GlobalAttention
from .mask import MaskedLayerNorm

class MSARowAttentionWithPairBias(nn.Cell):
    r"""
    MSA row attention. Information from pair action value is made as the bias of the matrix of MSARowAttention,
    in order to update the state of MSA using pair information.
    """

    def __init__(self, num_head, key_dim, gating, msa_act_dim, pair_act_dim, batch_size=None, slice_num=0):
        super(MSARowAttentionWithPairBias, self).__init__()
        self.num_head = num_head
        self.batch_size = batch_size
        self.matmul = P.MatMul(transpose_b=True)
        self.attn_mod = Attention(num_head, key_dim, gating, msa_act_dim, msa_act_dim, msa_act_dim, batch_size)
        self.msa_act_dim = msa_act_dim
        self.pair_act_dim = pair_act_dim
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.idx = Tensor(0, mstype.int32)
        self.masked_layer_norm = MaskedLayerNorm()
        self._init_parameter()

    def construct(self, msa_act, msa_mask, pair_act, index=None):
        '''construct'''
        norm_msa_mask = None
        norm_pair_mask = None
        res_idx = None
        if self.batch_size:
            query_norm_gamma = P.Gather()(self.query_norm_gammas, index, 0)
            query_norm_beta = P.Gather()(self.query_norm_betas, index, 0)
            feat_2d_norm_gamma = P.Gather()(self.feat_2d_norm_gammas, index, 0)
            feat_2d_norm_beta = P.Gather()(self.feat_2d_norm_betas, index, 0)
            feat_2d_weight = P.Gather()(self.feat_2d_weights, index, 0)
        else:
            query_norm_gamma = self.query_norm_gammas
            query_norm_beta = self.query_norm_betas
            feat_2d_norm_gamma = self.feat_2d_norm_gammas
            feat_2d_norm_beta = self.feat_2d_norm_betas
            feat_2d_weight = self.feat_2d_weights

        q, k, _ = pair_act.shape
        input_bias = 1e9 * (msa_mask - 1.0)
        input_bias = P.ExpandDims()(P.ExpandDims()(input_bias, 1), 2)

        msa_act = self.masked_layer_norm(msa_act, query_norm_gamma, query_norm_beta, mask=norm_msa_mask)
        pair_act = self.masked_layer_norm(pair_act, feat_2d_norm_gamma, feat_2d_norm_beta, mask=norm_pair_mask)
        pair_act = P.Reshape()(pair_act, (-1, pair_act.shape[-1]))
        nonbatched_bias = P.Transpose()(P.Reshape()(self.matmul(pair_act, feat_2d_weight), (q, k, self.num_head)),
                                        (2, 0, 1))
        batched_inputs = (msa_act, input_bias)
        if res_idx is not None:
            nonbatched_inputs = (nonbatched_bias, res_idx)
        else:
            nonbatched_inputs = (index, nonbatched_bias)
        msa_act = _memory_reduce(self._compute, batched_inputs, nonbatched_inputs, self.slice_num)
        return msa_act

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim]), mstype.float32))
            self.feat_2d_norm_gammas = Parameter(
                Tensor(np.zeros([self.batch_size, self.pair_act_dim]), mstype.float32))
            self.feat_2d_norm_betas = Parameter(
                Tensor(np.zeros([self.batch_size, self.pair_act_dim]), mstype.float32))
            self.feat_2d_weights = Parameter(
                Tensor(np.zeros([self.batch_size, self.num_head, self.pair_act_dim]), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones([self.msa_act_dim]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.msa_act_dim]), mstype.float32))
            self.feat_2d_norm_gammas = Parameter(Tensor(np.ones([self.pair_act_dim]), mstype.float32))
            self.feat_2d_norm_betas = Parameter(Tensor(np.zeros([self.pair_act_dim]), mstype.float32))
            self.feat_2d_weights = Parameter(
                Tensor(np.random.normal(scale=1 / np.sqrt(self.pair_act_dim), size=[self.num_head, self.pair_act_dim]),
                       mstype.float32))

    def _compute(self, msa_act, mask, index, nonbatched_bias):
        """
        compute.
        """
        msa_act = self.attn_mod(msa_act, msa_act, mask, index, nonbatched_bias)
        return msa_act


class MSAColumnAttention(nn.Cell):
    """
    MSA column-wise gated self attention.
    The column-wise attention lets the elements that belong to the same target residue exchange information.
    """

    def __init__(self, num_head, key_dim, gating, msa_act_dim, batch_size=None, slice_num=0):
        super(MSAColumnAttention, self).__init__()
        self.query_norm = MaskedLayerNorm()
        self.attn_mod = Attention(num_head, key_dim, gating, msa_act_dim, msa_act_dim, msa_act_dim, batch_size)
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.msa_act_dim = msa_act_dim
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def construct(self, msa_act, msa_mask, index=None):
        '''construct'''
        if self.batch_size:
            query_norm_gamma = P.Gather()(self.query_norm_gammas, index, 0)
            query_norm_beta = P.Gather()(self.query_norm_betas, index, 0)
        else:
            query_norm_gamma = self.query_norm_gammas
            query_norm_beta = self.query_norm_betas
        msa_act = P.Transpose()(msa_act, (1, 0, 2))
        msa_mask = P.Transpose()(msa_mask, (1, 0))

        input_mask = 1e9 * (msa_mask - 1.)
        input_mask = P.ExpandDims()(P.ExpandDims()(input_mask, 1), 2)
        msa_act = self.query_norm(msa_act, query_norm_gamma, query_norm_beta)
        batched_inputs = (msa_act, input_mask)
        nonbatched_inputs = (index,)
        msa_act = _memory_reduce(self._compute, batched_inputs, nonbatched_inputs, self.slice_num)
        msa_act = P.Transpose()(msa_act, (1, 0, 2))
        return msa_act

    def _init_parameter(self):
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim]), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones([self.msa_act_dim]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.msa_act_dim]), mstype.float32))

    def _compute(self, msa_act, input_mask, index):
        '''compute'''
        msa_act = self.attn_mod(msa_act, msa_act, input_mask, index)
        return msa_act


class MSAColumnGlobalAttention(nn.Cell):
    r"""
    MSA column global attention. Transpose MSA information at sequence axis and residue axis, then use `GlobalAttention
    <https://www.mindspore.cn/mindsponge/docs/zh-CN/master/cell/mindsponge.cell.GlobalAttention.html>` to
    do Attention between input sequences without dealing with the relationship between residues in sequence.
    Comparing with MSAColumnAttention, it uses GlobalAttention to deal with longer input sequence.
    """

    def __init__(self, num_head, gating, msa_act_dim, batch_size=None, slice_num=0):
        super(MSAColumnGlobalAttention, self).__init__()
        self.attn_mod = GlobalAttention(num_head, gating, msa_act_dim, msa_act_dim, batch_size)
        self.query_norm = MaskedLayerNorm()
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.msa_act_dim = msa_act_dim
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def construct(self, msa_act, msa_mask, index=None):
        '''construct'''
        if self.batch_size:
            query_norm_gamma = P.Gather()(self.query_norm_gammas, index, 0)
            query_norm_beta = P.Gather()(self.query_norm_betas, index, 0)
            msa_act = P.Transpose()(msa_act, (1, 0, 2))
            msa_mask = P.Transpose()(msa_mask, (1, 0))
        else:
            query_norm_gamma = self.query_norm_gammas
            query_norm_beta = self.query_norm_betas
            msa_act = P.Transpose()(msa_act, (1, 0, 2))
            msa_mask = P.Transpose()(msa_mask, (1, 0))

        input_mask = 1e9 * (msa_mask - 1.)
        input_mask = P.ExpandDims()(P.ExpandDims()(input_mask, 1), 2)

        msa_act = self.query_norm(msa_act,
                                  query_norm_gamma,
                                  query_norm_beta)
        msa_mask = P.ExpandDims()(msa_mask, -1)
        batched_inputs = (msa_act, msa_mask)
        nonbatched_inputs = (index,)
        msa_act = _memory_reduce(self._compute, batched_inputs, nonbatched_inputs, self.slice_num)
        msa_act = P.Transpose()(msa_act, (1, 0, 2))
        return msa_act

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros((self.batch_size, self.msa_act_dim)), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros((self.batch_size, self.msa_act_dim)), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones((self.msa_act_dim)), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros((self.msa_act_dim)), mstype.float32))

    def _compute(self, msa_act, msa_mask, index):
        """
        compute.

        Args:
            msa_act (Tensor):       Tensor of msa_act.
            msa_mask (Tensor):      The mask for msa_act matrix.
            index (Tensor):         The index of while loop, only used in case of while
                                    control flow. Default: ``None``.

        Outputs:
            - **msa_act** (Tensor)- Tensor, the float tensor of the msa_act of the attention layer.
        """
        msa_act = self.attn_mod(msa_act, msa_act, msa_mask, index)
        return msa_act
