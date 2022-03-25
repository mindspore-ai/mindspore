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
'''Utils for RNNs CPU version, like Reverse operators'''
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.ops as P
from mindspore.ops.primitive import constexpr
from mindspore.nn.cell import Cell
from mindspore.common.tensor import Tensor


@constexpr
def arange(start, stop, step):
    return Tensor(np.arange(start, stop, step), mstype.int32)


class _Reverse(Cell):
    """Reverse operator, like Reverse in mindspore"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def construct(self, input_x):
        dim_size = input_x.shape[self.dim]
        reversed_indexes = arange(dim_size-1, -1, -1)
        output = P.Gather()(input_x, reversed_indexes, self.dim)
        return output


class _ReverseSequence(Cell):
    """Reverse sequence operator, like ReverseSequenceV2 in mindspore"""
    def __init__(self, seq_dim, batch_dim=0):
        super().__init__()
        self.seq_dim = seq_dim
        self.batch_dim = batch_dim

    def construct(self, x, seq_lengths):
        """Defines the ReverseSequence operator computation performed."""
        batch_size = x.shape[self.batch_dim]
        max_seq_len = x.shape[self.seq_dim]
        seq_lens_type = seq_lengths.dtype

        back = P.Sub()(seq_lengths, P.OnesLike()(seq_lengths))

        batch_idx = self.make_shape((batch_size, max_seq_len), seq_lens_type, 0)
        forward_idx = self.make_shape((batch_size, max_seq_len), seq_lens_type, 1)

        back = back.view(-1, 1)
        reverse_idx = P.Sub()(back, forward_idx)

        condition = P.Less()(reverse_idx, P.ZerosLike()(reverse_idx))
        reverse_idx = P.Select()(condition, forward_idx, reverse_idx)

        reverse_idx = P.ExpandDims()(reverse_idx, 2)
        batch_idx = P.ExpandDims()(batch_idx, 2)

        if self.batch_dim > self.seq_dim:
            batch_idx = P.Transpose()(batch_idx, (1, 0, 2))
            reverse_idx = P.Transpose()(reverse_idx, (1, 0, 2))
            x = P.Transpose()(x, (1, 0, 2))
        start_indices = P.Concat(2)((batch_idx, reverse_idx))

        output = P.GatherNd()(x, start_indices)

        return output


    @staticmethod
    def make_shape(shape, dtype, range_dim):
        """Calculates the shape according by the inputs."""
        output = P.Ones()(shape, mstype.float32)
        output = P.CumSum()(output, range_dim)
        output = P.Cast()(output, dtype)
        output = output - 1
        return output
