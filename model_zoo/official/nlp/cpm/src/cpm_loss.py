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
"""Loss."""
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor


class Cross_entropy(nn.Cell):
    """
    Calculate loss of Training mode and zero-shot mode.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input tensor sequence.
        vocab_size (int): Size of the dictionary of embeddings.
        config: The config of networks.
        is_training (bool): Whether is training.
    Returns:
        Tensor, shape of (batch_size,).
    """

    def __init__(self, batch_size, seq_length, vocab_size, config, is_training=False):
        super(Cross_entropy, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.argmax = P.ArgMaxWithValue(axis=-1).shard(((config.dp, 1, 1),))
        self.expanddim = P.ExpandDims().shard(((config.dp, 1),))
        self.sub = P.Sub().shard(((config.dp, 1), (config.dp, 1)))
        self.sub_logist = P.Sub().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.exp = P.Exp().shard(((config.dp, 1, 1),))
        self.till = P.Tile().shard(((1, 1),))
        self.till_expand = P.Tile().shard(((config.dp, 1, 1),))
        self.reduce_sum = P.ReduceSum().shard(((config.dp, 1, 1),))
        self.reshape = P.Reshape()
        self.is_training = is_training
        if self.is_training:
            self.seq_length = 1
        else:
            self.seq_length = seq_length

        self.first_dim = self.batch_size * self.seq_length
        self.start = Tensor(np.zeros((self.batch_size, self.seq_length), dtype=np.int32))
        self.zero = Tensor(np.zeros((self.batch_size, self.seq_length), dtype=np.float32))
        self.end = Tensor(np.array([[self.vocab_size]], dtype=np.int32))
        self.arange = Tensor(np.expand_dims(np.arange(0, self.first_dim), -1), mstype.int32)

        self.greater = P.GreaterEqual().shard(((config.dp, 1), (config.dp, 1)))
        self.logicalor = P.LogicalOr().shard(((config.dp, 1), (config.dp, 1)))
        self.less = P.Less().shard(((config.dp, 1), (config.dp, 1)))
        self.squeeze = P.Squeeze(axis=0)
        self.log = P.Log().shard(((config.dp, 1),))
        self.cast = P.Cast()
        self.select = P.Select().shard(((config.dp, 1), (config.dp, 1), (config.dp, 1)))
        self.select_target = P.Select().shard(((config.dp, 1), (config.dp, 1), (config.dp, 1)))
        self.concat = P.Concat(axis=-1).shard(((config.dp, 1), (config.dp, 1)))
        self.gathernd = P.GatherNd().shard(((1, 1), (1, 1)))
        self.sub_last = P.Sub().shard(((config.dp, 1), (config.dp, 1)))

        self.realdiv = P.RealDiv().shard(((1,), (1,)))
        self.mul = P.Mul().shard(((1, 1), (1, 1)))
        self.reduce_sum2 = P.ReduceSum().shard(((1, 1),))
        self.reduce_sum3 = P.ReduceSum().shard(((1, 1),))

    def construct(self, logits, target, loss_mask=None):
        r"""
        Compute loss using logits, target and loss mask.
        """
        # [8, 1, 30000]
        _, logits_max = self.argmax(logits)
        # [8 1]
        logits_max_expand = self.expanddim(logits_max, -1)
        logits_max_expand = self.till_expand(logits_max_expand, (1, 1, self.vocab_size))
        # [8, 1, 30000] | [8, 1, 30000]
        logits_sub = self.sub_logist(logits, logits_max_expand)
        logits_exp = self.exp(logits_sub)
        # [8, 1, 30000] ->[8,30000]
        sum_exp_logits = self.reduce_sum(logits_exp, -1)
        # create a mask of a valid vocab ids
        ends = self.end
        ends = self.till(ends, (self.batch_size, self.seq_length))

        vocab_start_res = self.less(target, self.start)
        vocab_end_res = self.greater(target, ends)
        # training mode: [batch 1].
        target_mask = self.logicalor(vocab_start_res, vocab_end_res)
        masked_target = self.sub(target, self.start)
        masked_target = self.select_target(target_mask, self.zero, self.cast(masked_target, mstype.float32))
        # [batch, vocab]
        logits_2d = self.reshape(logits_sub, (-1, self.vocab_size))
        masked_target_1d = self.reshape(masked_target, (-1, 1))
        masked_target_1d = self.cast(masked_target_1d, mstype.int32)

        zeros = self.zero
        # the next stack/concat means: predicted_logits_1d, logits_2d[self.arange, masked_target_1d]
        stack_out = self.concat((self.arange, masked_target_1d))

        predicted_logits_1d = self.gathernd(logits_2d, stack_out)
        predicted_logits = self.reshape(predicted_logits_1d, (self.batch_size, -1))
        predicted_logits_masked = self.select(target_mask, zeros, self.cast(predicted_logits, mstype.float32))
        losses = self.sub_last(self.log(sum_exp_logits), predicted_logits_masked)

        if (not self.is_training) and (loss_mask is not None):
            # loss calculate.
            loss_mask_sum = self.reduce_sum2(loss_mask, -1)
            loss_with_mask = self.mul(losses, loss_mask)
            loss_with_mask_sum = self.reduce_sum3(loss_with_mask, -1)
            loss = self.realdiv(loss_with_mask_sum, loss_mask_sum)
            return loss
        return losses


class Cross_entropy_eval(nn.Cell):
    """
    Calculate loss of validation mode.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input tensor sequence.
        vocab_size (int): Size of the dictionary of embeddings.
        config: The config of networks.

    Returns:
        Tensor, shape of (batch_size,).
    """

    def __init__(self, batch_size, seq_length, vocab_size, config):
        super(Cross_entropy_eval, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.argmax = P.ArgMaxWithValue(axis=-1).shard(((config.dp, 1),))
        self.squeeze = P.Squeeze()
        self.expanddims = P.ExpandDims().shard(((config.dp, 1),))
        self.expanddims1 = P.ExpandDims().shard(((config.dp,),))
        self.tile = P.Tile().shard(((config.dp, 1, 1),))
        self.reducesum = P.ReduceSum().shard(((config.dp, 1, 1),))
        self.reducesum2 = P.ReduceSum().shard(((config.dp, 1),))
        self.readdiv = P.RealDiv().shard(((config.dp, 1), (config.dp, 1)))
        self.mul = P.Mul().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.reshape = P.Reshape()

    def construct(self, logist, loss_mask):
        r"""
        Compute loss using logits and loss mask.
        """
        loss_mask_expand = self.expanddims(loss_mask, -1)
        loss_masks = self.tile(loss_mask_expand, (1, 1, self.vocab_size))
        loss_mask_sum = self.expanddims1(self.reducesum2(loss_mask, -1), -1)
        logist_mask_mul = self.mul(logist, loss_masks)
        logist_mask_sum = self.reducesum(logist_mask_mul, 1)
        output = self.readdiv(logist_mask_sum, loss_mask_sum)

        return output
