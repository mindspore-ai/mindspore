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
"""
Parallel Loss for the Parallel Training
This is an experimental interface that is subject to change and/or deletion.
"""
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn import Cell
from mindspore.nn.loss.loss import _check_is_tensor
from mindspore.parallel.nn.layers import _check_input_dtype, _check_input_shape
from .op_parallel_config import default_dpmp_config, OpParallelConfig

__all__ = ["CrossEntropyLoss"]


class CrossEntropyLoss(Cell):
    """
    Calculate the cross entropy loss.

    Args:
        parallel_config (OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                                           an instance of `OpParallelConfig` with default args.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. the output logits of
          the backbone.

        - **labels** (Tensor) - Tensor of shape (N, ). The ground truth label of the sample.

        - **input_mask** (Tensor) - Tensor of shape (N, ). input_mask indicates whether there is padded inputs and for
          padded inputs it will not be counted into loss.

    Outputs:
        Tensor. the corresponding cross entropy loss

    Examples:
        >>> from mindspore.parallel.nn import CrossEntropyLoss
        >>> loss = CrossEntropyLoss()
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mindspore.float32)
        >>> labels_np = np.array([1]).astype(np.int32)
        >>> input_mask = Tensor(np.ones(1).astype(np.float32))
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels, input_mask)
        >>> print(output.shape)
        (1,)
    """

    def __init__(self, parallel_config=default_dpmp_config):
        super(CrossEntropyLoss, self).__init__()
        if not isinstance(parallel_config, OpParallelConfig):
            raise TypeError("Input args parallel_config must be the type OpParallelConfig.")
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.sum = P.ReduceSum().shard(((dp, mp),))
        self.onehot = P.OneHot().shard(((dp, mp), (), ()))
        # on/off value for onehot, for smooth labeling, modify the off_value
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.max = P.ArgMaxWithValue(axis=-1, keep_dims=True).shard(
            ((dp, mp),))
        self.eps_const = Tensor(1e-24, mstype.float32)
        self.sub = P.Sub().shard(((dp, mp), (dp, 1)))
        self.exp = P.Exp().shard(((dp, mp),))
        self.div = P.RealDiv().shard(((dp, mp), (dp, 1)))
        self.log = P.Log().shard(((dp, mp),))
        self.add = P.Add().shard(((dp, mp), ()))
        self.mul = P.Mul().shard(
            ((dp, mp), (dp, mp)))
        self.neg = P.Neg().shard(((dp, mp),))
        self.sum2 = P.ReduceSum().shard(((1,),))

        self.mul2 = P.Mul().shard(((1,), (1,)))
        self.add2 = P.Add()
        self.div2 = P.RealDiv()

    def construct(self, logits, label, input_mask):
        self._check_input(logits, label, input_mask)

        # the shape is [bs*seq_length, vocab_size]
        logits = F.cast(logits, mstype.float32)
        # LogSoftmax for logits over last dimension
        _, logit_max = self.max(logits)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1)
        exp_sum = P.Reshape()(exp_sum, (F.shape(exp_sum)[0], 1))
        softmax_result = self.div(logit_exp, exp_sum)
        log_softmax_result = self.log(self.add(softmax_result, self.eps_const))

        # Flatten label to [bs*seq_length]
        label = P.Reshape()(label, (-1,))
        # Get onehot label [bs*seq_length, vocab_size]
        one_hot_label = self.onehot(label, F.shape(logits)[-1], self.on_value,
                                    self.off_value)
        # Cross-Entropy loss
        loss = self.mul(log_softmax_result, one_hot_label)
        loss_unsum = self.neg(loss)
        loss_reduce = self.sum(loss_unsum, -1)
        # input_mask indicates whether there is padded inputs and for padded inputs it will not be counted into loss
        input_mask = P.Reshape()(input_mask, (-1,))
        numerator = self.sum2(self.mul2(loss_reduce, input_mask))

        denominator = self.add2(
            self.sum2(input_mask),
            P.Cast()(F.tuple_to_array((1e-5,)), mstype.float32))
        loss = self.div2(numerator, denominator)
        return loss

    def _check_input(self, logits, label, input_mask):
        r"""Check the input tensor shape and type"""
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('label', label, self.cls_name)
        _check_is_tensor('input_mask', input_mask, self.cls_name)
        _check_input_dtype(F.dtype(logits), "logits", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(F.dtype(label), "label", [mstype.int32], self.cls_name)
        _check_input_dtype(F.dtype(input_mask), "input_mask", [mstype.float32], self.cls_name)
        _check_input_shape(F.shape(logits), "logits", self.cls_name, 2)
        _check_input_shape(F.shape(label), "label", self.cls_name, 1)
        _check_input_shape(F.shape(input_mask), "input_mask", self.cls_name, 1)
        return True
