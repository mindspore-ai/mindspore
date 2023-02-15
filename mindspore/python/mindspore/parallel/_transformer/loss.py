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
This is an experimental interface that is subject to change or deletion.
"""
from __future__ import absolute_import

from mindspore.parallel import set_algo_parameters
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn import Cell
from mindspore.nn.loss.loss import _check_is_tensor
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_device_num, _get_pipeline_stages
from mindspore.log import _LogActionOnce
from mindspore import log as logger
from mindspore.parallel._transformer.layers import _check_input_dtype
from mindspore.parallel._transformer.op_parallel_config import default_dpmp_config, OpParallelConfig

__all__ = ["CrossEntropyLoss"]


class _Softmax(Cell):
    """
    Calculate the softmax results with given logits.

    Note:
        The bprop of the cell is rewritten, just returns the accepted dout as returns. This cell should be used
        together with _NLLoss, to optimize the bprop of the cross entroy loss.

    Args:
        parallel_config (OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
            an instance of `OpParallelConfig` with default args.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. The output logits of
          the backbone.


    Outputs:
        Tensor. The corresponding softmax results.
    """
    def __init__(self, parallel_config=default_dpmp_config):
        super(_Softmax, self).__init__()
        if not isinstance(parallel_config, OpParallelConfig):
            raise TypeError("For 'CrossEntropyLoss', the class variable 'parallel_config' must be OpParallelConfig"
                            ", but got the type: {}.".format(type(parallel_config)))
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        # on/off value for onehot, for smooth labeling, modify the off_value
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

        self.sum = P.ReduceSum().shard(((dp, mp),))
        self.max = P.ArgMaxWithValue(axis=-1, keep_dims=True).shard(
            ((dp, mp),))
        self.sub = P.Sub().shard(((dp, mp), (dp, 1)))
        self.exp = P.Exp().shard(((dp, mp),))
        self.div = P.RealDiv().shard(((dp, mp), (dp, 1)))
        self.onehot = P.OneHot().shard(((dp, mp), (), ()))

    def construct(self, logits, label):
        # LogSoftmax for logits over last dimension
        logits = F.cast(logits, mstype.float32)
        _, logit_max = self.max(logits)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1)
        exp_sum = P.Reshape()(exp_sum, (F.shape(exp_sum)[0], 1))
        softmax_result = self.div(logit_exp, exp_sum)

        one_hot_label = self.onehot(label, F.shape(logits)[-1], self.on_value, self.off_value)
        return softmax_result, one_hot_label

    def bprop(self, logits, label, out, dout):
        """just return the loss of the dout. Note this should be used together with _NLLLoss"""
        d_logits = F.cast(dout[0], F.dtype(logits))
        return d_logits, F.zeros_like(label)


class _NLLLoss(Cell):
    """
    Calculate the NLLLoss results with given softmax results and the label.

    Note:
        The bprop of the cell is rewritten. This cell should be used
        together with _Softmax, to optimize the bprop of the cross entroy loss.

    Args:
        parallel_config (OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
            an instance of `OpParallelConfig` with default args.

    Inputs:
        - **loss** (Tensor) - Tensor of shape (N, C). Data type is float32.

    Outputs:
        Tensor. The corresponding loss results.
    """
    def __init__(self, parallel_config=default_dpmp_config):
        super(_NLLLoss, self).__init__()
        if not isinstance(parallel_config, OpParallelConfig):
            raise TypeError("For 'CrossEntropyLoss', the class variable 'parallel_config' must be OpParallelConfig"
                            ", but got the type: {}.".format(type(parallel_config)))
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.repeat_loss = 1
        self.eps_const = Tensor(1e-24, mstype.float32)
        # In auto parallel, there will be a virtual div in the back propagation begins. As we use custom bprop function
        # we need to eliminate this virtual div by adding a factor "mp".
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL):
            self.repeat_loss = mp
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sum = P.ReduceSum()
            self.mul = P.Mul()
            self.neg = P.Neg()
            self.log = P.Log()
            self.add = P.Add().shard(((dp, mp), ()))
        else:
            self.sum = P.ReduceSum().shard(((dp, mp),))
            self.mul = P.Mul().shard(((dp, mp), (dp, mp)))
            self.neg = P.Neg().shard(((dp, mp),))
            self.log = P.Log().shard(((dp, mp),))
            self.add = P.Add().shard(((dp, mp), ()))

    def construct(self, softmax_result, one_hot_label):
        log_softmax_result = self.log(self.add(softmax_result, self.eps_const))
        loss = self.mul(log_softmax_result, one_hot_label)
        loss_unsum = self.neg(loss)
        loss_reduce = self.sum(loss_unsum, -1)
        return loss_reduce

    def bprop(self, softmax_result, one_hot_label, out, dout):
        """A simplified function. Note this should be used together with _Softmax"""
        logits = softmax_result - one_hot_label
        logits = logits * P.ExpandDims()(dout, -1) * self.repeat_loss

        return logits, F.zeros_like(one_hot_label)


class CrossEntropyLoss(Cell):
    """
    Calculate the cross entropy loss.

    Args:
        parallel_config (OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
            an instance of `OpParallelConfig` with default args.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. The output logits of
          the backbone.

        - **labels** (Tensor) - Tensor of shape (N, ). The ground truth label of the sample.

        - **input_mask** (Tensor) - Tensor of shape (N, ). input_mask indicates whether there are padded inputs and for
          padded inputs it will not be counted into loss.

    Outputs:
        Tensor. The corresponding cross entropy loss.

    Examples:
        >>> import numpy as np
        >>> from mindspore import dtype as mstype
        >>> from mindspore.nn.transformer import CrossEntropyLoss
        >>> from mindspore import Tensor
        >>> loss = CrossEntropyLoss()
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mstype.float32)
        >>> labels_np = np.array([1]).astype(np.int32)
        >>> input_mask = Tensor(np.ones(1).astype(np.float32))
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels, input_mask)
        >>> print(output.shape)
        (1,)
    """
    @_LogActionOnce(logger=logger, key='CrossEntropyLoss',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    def __init__(self, parallel_config=default_dpmp_config):
        super(CrossEntropyLoss, self).__init__()
        if not isinstance(parallel_config, OpParallelConfig):
            raise TypeError("For 'CrossEntropyLoss', the class variable 'parallel_config' must be OpParallelConfig"
                            ", but got the type: {}.".format(type(parallel_config)))
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.enable_force_redistribute = False
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL):
            self.enable_force_redistribute = True
            self.add = P.Add().shard(((dp, mp), ())).add_prim_attr("keep_alive", True)
            self.add_label = P.Add().shard(((dp,), ())).add_prim_attr("keep_alive", True)
            self._check_and_modify_sharding_context(dp)
        self.sum2 = P.ReduceSum().shard(((1,),))
        self.mul2 = P.Mul().shard(((1,), (1,)))
        self.add2 = P.Add()
        self.div2 = P.RealDiv()
        self.relu = P.ReLU().shard(((1,),))

        self._softmax = _Softmax(parallel_config)
        self._nllloss = _NLLLoss(parallel_config)

    @staticmethod
    def _check_and_modify_sharding_context(dp):
        device_num = _get_device_num()
        stages = _get_pipeline_stages()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and dp * stages != device_num:
            set_algo_parameters(fully_use_devices=False)

    def construct(self, logits, label, input_mask):
        self._check_input(logits, label, input_mask)

        # The add is used for forcing the redistribution before stepping in sub graphs, when semi/auto parallel enabled.
        if self.enable_force_redistribute:
            logits = self.add(logits, 0)
            label = self.add_label(label, 0)
        softmax, one_hot_label = self._softmax(logits, label)
        loss_reduce = self._nllloss(softmax, one_hot_label)

        # Using input_mask to mask the loss
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
        return True
