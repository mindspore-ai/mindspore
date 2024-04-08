# Copyright 2024 Huawei Technologies Co., Ltd
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
"""MindFormer Self-Define Loss."""

from mindspore import nn, Tensor
from mindspore import ops as P
from mindspore.common import dtype as mstype
from mindspore.context import ParallelMode
from mindspore.ops import functional as F
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._utils import _get_device_num, _get_pipeline_stages, _get_parallel_mode, \
    _is_sharding_propagation

from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
__all__ = ['CrossEntropyLoss']


class _LogSoftmax(nn.Cell):
    """
    Calculate the log softmax results with given logits. The bprop of the cell is rewritten,
    just returns the accepted dout as returns. This cell should be used together with _NLLoss,
    to optimize the bprop of the cross entroy loss.

    Args:
        parallel_config (OpParallelConfig): The parallel configuration. Default `default_dpmp_config`,
            an instance of `OpParallelConfig` with default args.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. The output logits of
          the backbone.

        - **label** (Tensor) - Tensor of shape (N, 1). The ground truth label of the sample.

    Returns:
        The corresponding log softmax results.
    """

    def __init__(self, parallel_config=default_dpmp_config):
        super(_LogSoftmax, self).__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        # on/off value for onehot, for smooth labeling, modify the off_value
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

        self.sum = P.ReduceSum(keep_dims=True).shard(((dp, mp),))
        self.max = P.ReduceMax(keep_dims=True).shard(
            ((dp, mp),))
        self.sub = P.Sub().shard(((dp, mp), (dp, 1)))
        self.exp = P.Exp().shard(((dp, mp),))
        self.log = P.Log().shard(((dp, 1),))
        self.onehot = P.OneHot().shard(((dp, mp), (), ()))

    def construct(self, logits, label):
        """Forward process"""
        logits = F.cast(logits, mstype.float32)
        logit_max = self.max(logits, 1)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1)
        log_exp_sum = self.log(exp_sum)
        log_softmax_result = self.sub(logit_sub, log_exp_sum)

        one_hot_label = self.onehot(label, F.shape(logits)[-1], self.on_value, self.off_value)
        return log_softmax_result, one_hot_label

    def bprop(self, logits, label, _, dout):
        """just return the loss of the dout. Note this should be used together with _NLLLoss"""
        d_logits = F.cast(dout[0], F.dtype(logits))
        return d_logits, F.zeros_like(label)


class _NLLLoss(nn.Cell):
    """
    Calculate the NLLLoss results with given log softmax results and the label. The bprop of the cell is rewritten.
    This cell should be used together with _Log_softmax, to optimize the bprop of the cross entroy loss.

    Args:
        parallel_config (OpParallelConfig): The parallel configuration. Default `default_dpmp_config`,
            an instance of `OpParallelConfig` with default args.

    Inputs:
        - **log_softmax_result** (Tensor) - Tensor of shape (N, C). Data type is float32.
        - **one_hot_label** (Tensor) - Tensor of shape (N, C). The ground truth label in one-hot format of the sample.

    Returns:
        The corresponding loss results.
    """

    def __init__(self, parallel_config=default_dpmp_config):
        super(_NLLLoss, self).__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.repeat_loss = 1
        self.gather_d = P.GatherD()
        self.expand_dims = P.ExpandDims()
        # In auto parallel, there will be a virtual div in the back propagation begins. As we use custom bprop function
        # we need to eliminate this virtual div by adding a factor "mp".
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL):
            self.repeat_loss = mp
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sum = P.ReduceSum()
            self.mul = P.Mul()
            self.neg = P.Neg()
        else:
            self.sum = P.ReduceSum().shard(((dp, mp),))
            self.mul = P.Mul().shard(((dp, mp), (dp, mp)))
            self.neg = P.Neg().shard(((dp, mp),))

    def construct(self, log_softmax_result, one_hot_label):
        """Forward process"""
        loss = self.mul(log_softmax_result, one_hot_label)
        loss_unsum = self.neg(loss)
        loss_reduce = self.sum(loss_unsum, -1)
        return loss_reduce

    def bprop(self, log_softmax_result, one_hot_label, _, dout):
        """A simplified function. Note this should be used together with _Softmax"""
        softmax_result = P.Exp()(log_softmax_result)
        logits = softmax_result - one_hot_label
        logits = logits * P.ExpandDims()(dout, -1) * self.repeat_loss

        return logits, F.zeros_like(one_hot_label)


class CrossEntropyLoss(nn.Cell):
    def __init__(self, parallel_config=default_dpmp_config, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.kwargs = kwargs
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

        self._log_softmax = _LogSoftmax(parallel_config)
        self._nllloss = _NLLLoss(parallel_config)

    @staticmethod
    def _check_and_modify_sharding_context(dp):
        device_num = _get_device_num()
        stages = _get_pipeline_stages()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and dp * stages != device_num:
            set_algo_parameters(fully_use_devices=False)

    def construct(self, logits, label, input_mask):
        """Forward process"""
        # The add is used for forcing the redistribution before stepping in sub graphs, when semi/auto parallel enabled.
        if self.enable_force_redistribute:
            logits = self.add(logits, 0)
            label = self.add_label(label, 0)
        log_softmax, one_hot_label = self._log_softmax(logits, label)
        loss_reduce = self._nllloss(log_softmax, one_hot_label)

        # Using input_mask to mask the loss
        input_mask = P.Reshape()(input_mask, (-1,))
        numerator = self.sum2(self.mul2(loss_reduce, input_mask))

        denominator = self.add2(
            self.sum2(input_mask),
            P.Cast()(F.tuple_to_array((1e-8,)), mstype.float32))
        loss = self.div2(numerator, denominator)

        return loss
