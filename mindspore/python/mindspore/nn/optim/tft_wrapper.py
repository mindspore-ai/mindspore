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
"""OptTFTWrapper"""
from __future__ import absolute_import

import os
from mindspore.common.tensor import Tensor
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.ops.operations.manually_defined._inner import TensorReport
from mindspore import ops, context
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype

class OptTFTWrapper(Optimizer):
    r"""
    Implements TFT optimizer wrapper, this wrapper is used to report status to MindIO TFT before optimizer updating.

    Note:
        This optimizer is depend on MindIO TFT feature. Currently only support ascend graph mode and
        sink_size must be less than 1.

    Args:
        opt (Optimizer): Must be sub-class of Optimizer.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of opt's `params`, the shape is the same as opt's `params`.

    Outputs:
        Tensor, result of executing optimizer 'opt'.

    Raises:
        TypeError: If the parameter opt is not an subclass of Optimizer.
        ValueError: If the platform is not Ascend graph mode, or customer doesn't switch on TFT feature.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.SGD(params=net.trainable_params())
        >>> optim_wrapper = nn.OptTFTWrapper(optim)
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.train.Model(net, loss_fn=loss, optimizer=optim)
    """

    def __init__(self, opt, **kwargs):
        if not isinstance(opt, Optimizer):
            raise TypeError(f"For 'OptTFTWrapper', the argument 'opt' must be Optimizer type, " f"but got {type(opt)}.")
        super(OptTFTWrapper, self).__init__(opt.learning_rate, opt._parameters) # pylint: disable=W0212
        tft_env = os.getenv("MS_ENABLE_TFT", "")
        if ("TTP:1" not in tft_env) and ("UCE:1" not in tft_env) and ("ARF:1" not in tft_env):
            raise ValueError("MindIO TFT regitster need custom switch on[MS_ENABLE_TFT='{TTP:1,UCE:1,ARF:1}']!")
        device_target = context.get_context("device_target")
        if device_target != "Ascend":
            raise ValueError("MindIO adataper only support on Ascend device!")
        self.opt = opt
        self.report = TensorReport()
        self.report_end = TensorReport()
        self.report_end.add_prim_attr("side_effect_mem", True).add_prim_attr("optimizer_end", True)
        self.depend = ops.Depend()
        self.allreduce_sum = ops.AllReduce()
        self.allreduce_sum.add_prim_attr("tft_report_before", True)
        self.tft_g_one_flag = Parameter(Tensor([1], dtype=mstype.int32))

        self.param_rank = opt.param_rank
        self.optim_filter = opt.optim_filter
        self.loss_scale = opt.loss_scale
        self.dynamic_weight_decay = opt.dynamic_weight_decay
        self.grad_centralization = opt.grad_centralization

        self.dynamic_lr = opt.dynamic_lr
        self.global_step = opt.global_step
        self.is_group = opt.is_group
        self.is_group_lr = opt.is_group_lr
        self.is_group_params_ordered = opt.is_group_params_ordered
        self.use_parallel = opt.use_parallel
        if self.is_group:
            self.group_params = opt.group_params
            self.group_lr = opt.group_lr
            self.group_weight_decay = opt.group_weight_decay
            self.group_grad_centralization = opt.group_grad_centralization
            self.grad_centralization_flags = opt.grad_centralization_flags

        self.skip_auto_parallel_compile = opt.skip_auto_parallel_compile

        self.learning_rate = opt.learning_rate
        self.parameters = opt.parameters
        self.decay_flags = opt.decay_flags
        self.dynamic_decay_flags = opt.dynamic_decay_flags
        self.weight_decay = opt.weight_decay
        self.exec_weight_decay = opt.exec_weight_decay
        self.cache_enable = opt.cache_enable
        self.reciprocal_scale = opt.reciprocal_scale
        self.need_scale = opt.need_scale
        self.global_step_increase_tensor = opt.global_step_increase_tensor
        self.param_length = opt.param_length
        self.enable_tuple_broaden = opt.enable_tuple_broaden

    def construct(self, gradients):
        tft_g_one_flag = self.depend(self.tft_g_one_flag, gradients)
        self.tft_g_one_flag = self.allreduce_sum(tft_g_one_flag)

        grads = self.depend(gradients, self.report("tft_report", self.tft_g_one_flag))
        opt_ret = self.opt(grads)
        self.report_end("tft_report", self.tft_g_one_flag)
        return opt_ret
