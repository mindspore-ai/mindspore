# Copyright 2023 Huawei Technologies Co., Ltd
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
"""rprop"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common import Tensor, Parameter
import mindspore.common.dtype as mstype
from mindspore import _checkparam as validator
from mindspore.experimental.optim.optimizer import Optimizer, check_not_less_than_without_equal
from mindspore import ops
from mindspore import jit

_rprop_opt = C.MultitypeFuncGraph("rprop_opt")

op_sign = P.Sign()
op_fill = P.FillV2()
op_assign = P.Assign()
op_assignadd = P.AssignAdd()
op_cast = P.Cast()
op_select = P.Select()
op_oneslike = P.OnesLike()


@_rprop_opt.register("Tensor", "Tensor", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt(etaminus, etaplus, step_size_min, step_size_max, step, lr, param, prev, step_size, gradient):
    """Apply rprop optimizer to the weight parameter."""
    if step == 1:
        step_size_value = op_oneslike(step_size) * lr
    else:
        step_size_value = step_size.value()

    sign = op_sign(gradient * prev)

    sign[sign.gt(0)] = etaplus
    sign[sign.lt(0)] = etaminus
    sign[sign.eq(0)] = 1

    step_size_clip = ops.clip_by_value(step_size_value * sign, step_size_min, step_size_max)
    op_assign(step_size, step_size_clip)

    gradient_update = op_select(sign == etaminus, op_fill(sign.shape, op_cast(0., mstype.float32)), gradient)

    op_assign(prev, gradient_update)
    next_param = param - op_sign(gradient_update) * step_size_clip

    op_assign(param, next_param)

    return True


class Rprop(Optimizer):
    r"""
    Implements Rprop algorithm.

    .. warning::
        This is an experimental optimizer API that is subject to change.
        This module must be used with lr scheduler module in `LRScheduler Class
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#lrscheduler-class>`_ .

    Args:
        params (Union[list(Parameter), list(dict)]): list of parameters to optimize or dicts defining
            parameter groups.
        lr (Union[int, float, Tensor], optional): learning rate. Default: ``1e-2``.
        etas (Tuple[float, float], optional): pair of (etaminus, etaplus), that
            are multiplicative increase and decrease factors. Default:``(0.5, 1.2)``
        step_sizes (Tuple[float, float], optional): a pair of minimal and
            maximal allowed step sizes. Default:``(1e-6, 50)``

    Keyword Args:
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: ``False``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the `etas[1]` is less than or equal to 1.0.
        ValueError: If the `etas[0]` not in the range of 0-1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> optimizer = optim.Rprop(net.trainable_params(), lr=0.1)
        >>> def forward_fn(data, label):
        ...     logits = net(data)
        ...     loss = loss_fn(logits, label)
        ...     return loss, logits
        >>> grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        >>> def train_step(data, label):
        ...     (loss, _), grads = grad_fn(data, label)
        ...     optimizer(grads)
        ...     return loss
    """

    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50), *, maximize=False):
        check_not_less_than_without_equal(lr, "lr", self.cls_name)
        check_not_less_than_without_equal(etas[1], "etas[1]", self.cls_name, 1.)
        validator.check_float_range(etas[0], 0., 1., validator.INC_NEITHER, "etas[0]", self.cls_name)

        defaults = dict(
            lr=lr,
            etas=etas,
            step_sizes=step_sizes,
            maximize=maximize,
        )
        super(Rprop, self).__init__(params, defaults)
        self.prev = self.parameters.clone(prefix="prev", init='zeros')
        self.step_size = self.parameters.clone(prefix="step_size", init='zeros')
        self.step_t = Parameter(Tensor(0, mstype.int32), "step_t")
        self.increase_tensor = Tensor(1, mstype.int32)
        self.op_cast = P.Cast()

    @jit
    def implementation(self, etaminus, etaplus, group_id, lr, gradients, maximize, step_size_min, step_size_max):
        """Extract the common computing part for acceleration"""
        etaminus, etaplus = op_cast(etaminus, mstype.float32), op_cast(etaplus, mstype.float32)
        start_id = self.group_start_id[group_id]
        end_id = self.group_start_id[group_id + 1]

        params = self.parameters[start_id: end_id]
        grads = tuple([grad if not maximize else F.neg(grad) for grad in gradients[start_id: end_id]])
        prev = self.prev[start_id: end_id]
        step_size = self.step_size[start_id: end_id]
        self.hyper_map(F.partial(_rprop_opt, etaminus, etaplus, step_size_min, step_size_max, self.step_t, lr),
                       params, prev, step_size, grads)
        return True

    def construct(self, gradients):
        op_assignadd(self.step_t, self.increase_tensor)
        for group_id, group in enumerate(self.param_groups):
            lr = self.lrs[group_id]
            if isinstance(group.get("lr"), float):
                lr = self.op_cast(group.get("lr"), mstype.float32)
            maximize = group.get("maximize")

            etaminus, etaplus = group["etas"]
            step_size_min, step_size_max = group["step_sizes"]

            self.implementation(etaminus, etaplus, group_id, lr, gradients, maximize, step_size_min, step_size_max)

        return True
