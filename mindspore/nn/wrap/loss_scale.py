# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Loss scale cell for loss scale training."""
import mindspore.context as context
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.train.parallel_utils import ParallelMode
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_mirror_mean
from ..cell import Cell
from ...common import Tensor, ParameterTuple
from ...common.parameter import Parameter
from ...ops import functional as F
from ...ops import composite as C
from ...ops import operations as P
from ...ops.operations import NPUGetFloatStatus, NPUAllocFloatStatus, NPUClearFloatStatus, ReduceSum, LessEqual, \
    ControlDepend
from ...common import dtype as mstype

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)

class DynamicLossScaleUpdateCell(Cell):
    r"""
    Dynamic Loss scale update cell.

    For loss scaling training, the initial loss scaling value will be set to be `loss_scale_value`.
    In every training step, the loss scaling value  will be updated by loss scaling value/`scale_factor`
    when there is overflow. And it will be increased by loss scaling value * `scale_factor` if there is no
    overflow for a continuous `scale_window` steps. This cell is used for Graph mode training in which all
    logic will be executed on device side(Another training mode is non-sink mode in which some logic will be
    executed on host).

    Args:
        loss_scale_value (float): Init loss scale.
        scale_factor (int): Coefficient of increase and decrease.
        scale_window (int): Maximum continuous training steps that do not have overflow.

    Inputs:
        - **inputs** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a scalar Tensor with shape :math:`()`.

    Examples:
        >>> net_with_loss = Net()
        >>> optimizer = nn.Momentum(net_with_loss.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
        >>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_update_cell=manager)
        >>> train_network.set_train()
        >>>
        >>> inputs = Tensor(np.ones([16, 16]).astype(np.float32))
        >>> label = Tensor(np.zeros([16, 16]).astype(np.float32))
        >>> scaling_sens = Tensor(np.full((1), np.finfo(np.float32).max), dtype=mindspore.float32)
        >>> output = train_network(inputs, label, scaling_sens)
    """

    def __init__(self,
                 loss_scale_value,
                 scale_factor,
                 scale_window):
        super(DynamicLossScaleUpdateCell, self).__init__()

        self.scale_window = Tensor(scale_window, dtype=mstype.int32)
        self.scale_factor = Tensor(scale_factor, dtype=mstype.float32)
        self.loss_scale_value = loss_scale_value

        self.cur_iter = Parameter(Tensor(1, dtype=mstype.int32), name="current_iterator_step")
        self.last_overflow_iter = Parameter(Tensor(0, dtype=mstype.int32), name="last_overflow_iterator_step")
        self.select = P.Select()
        self.max = P.Maximum()
        self.minimum_loss_scale = Tensor(1.0, dtype=mstype.float32)
        self.reciprocal = P.Reciprocal()
        self.less_equal = P.LessEqual()
        self.logic_and = P.LogicalAnd()
        self.logic_not = P.LogicalNot()
        self.logic_or = P.LogicalOr()
        self.const_true = Tensor(True, dtype=mstype.bool_)

    def get_loss_scale(self):
        return self.loss_scale_value

    def construct(self, loss_scale, overflow):
        overflow_cond = overflow
        loss_scale_on_overflow = self.select(overflow_cond, self.max(loss_scale * self.reciprocal(self.scale_factor),
                                                                     self.minimum_loss_scale), loss_scale)
        should_inc = self.less_equal(self.scale_window, self.cur_iter - self.last_overflow_iter)
        last_iter_cond = self.logic_or(overflow_cond, should_inc)
        last_overflow_iter = self.select(last_iter_cond, self.cur_iter, self.last_overflow_iter)
        assign_last_iter = F.assign(self.last_overflow_iter, last_overflow_iter)
        update_scale_cond = self.logic_and(should_inc, self.logic_not(overflow_cond))
        scale_mul_res = loss_scale_on_overflow * self.scale_factor
        scaled_loss_scale = self.select(update_scale_cond, scale_mul_res, loss_scale_on_overflow)
        assign_scaled_loss_scale = F.assign(loss_scale, scaled_loss_scale)
        inc_cur_iter = self.cur_iter + 1
        assing_cur_iter = F.assign(self.cur_iter, inc_cur_iter)
        t = (assign_last_iter, assign_scaled_loss_scale, assing_cur_iter)
        F.control_depend(assign_last_iter, assing_cur_iter)
        return F.depend(overflow, t)


class FixedLossScaleUpdateCell(Cell):
    """
    Static scale update cell, the loss scaling value will not be updated.

    For usage please refer to `DynamicLossScaleUpdateCell`.

    Args:
        loss_scale_value (float): Init loss scale.

    Examples:
        >>> net_with_loss = Net()
        >>> optimizer = nn.Momentum(net_with_loss.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> manager = nn.FixedLossScaleUpdateCell(loss_scale_value=2**12)
        >>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_update_cell=manager)
        >>> train_network.set_train()
        >>>
        >>> inputs = Tensor(np.ones([16, 16]).astype(np.float32))
        >>> label = Tensor(np.zeros([16, 16]).astype(np.float32))
        >>> scaling_sens = Tensor(np.full((1), np.finfo(np.float32).max), dtype=mindspore.float32)
        >>> output = train_network(inputs, label, scaling_sens)
    """

    def __init__(self, loss_scale_value):
        super(FixedLossScaleUpdateCell, self).__init__()
        self.loss_scale_value = loss_scale_value

    def get_loss_scale(self):
        return self.loss_scale_value

    def construct(self, _, overflow):
        return overflow


class TrainOneStepWithLossScaleCell(Cell):
    r"""
    Network training with loss scaling.

    This is a training step with loss scaling. It takes a network, an optimizer and possibly a scale update
    Cell as args. The loss scale value can be updated in both host side or device side. The
    TrainOneStepWithLossScaleCell will be compiled to be graph which takes `data`, `label`, `sens` as input
    data. The `sens` is acting as loss scaling value. If you want to update it on host side, the value should
    be provided. If `sens` is not given, the loss scale update logic should be provied by `scale_update_cell`.
    If `scale_update_cell` is not None and `sens` is provided, the `scale_update_cell` will be ignored.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        scale_update_cell(Cell): The loss scaling update logic cell. Default: None.

    Inputs:
        - **inputs** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **scaling_sens** (Tensor) - Tensor of shape :math:`()`.

    Outputs:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scaling value.

        - **loss** (Tensor) -  Tensor with shape :math:`()`.
        - **overflow** (Tensor) -  Tensor with shape :math:`()`, type is bool.
        - **loss_scale** (Tensor) -  Tensor with shape :math:`()`.

    Examples:
        >>> net_with_loss = Net()
        >>> optimizer = nn.Momentum(net_with_loss.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> manager = nn.DynamicLossScaleUpdateCell(init_loss_scale=2**12, scale_factor=2, scale_window=1000)
        >>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_update_cell=manager)
        >>> train_network.set_train()
        >>>
        >>> inputs = Tensor(np.ones([16, 16]).astype(np.float32))
        >>> label = Tensor(np.zeros([16, 16]).astype(np.float32))
        >>> scaling_sens = Tensor(np.full((1), np.finfo(np.float32).max), dtype=mindspore.float32)
        >>> output = train_network(inputs, label, scaling_sens)
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(TrainOneStepWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation('grad', get_by_list=True, sens_param=True)
        self.hyper_map = C.HyperMap()
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
        else:
            self.gpu_target = False
            self.alloc_status = NPUAllocFloatStatus()
            self.get_status = NPUGetFloatStatus()
            self.clear_status = NPUClearFloatStatus()
        self.reduce_sum = ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = LessEqual()
        self.depend_parameter_use = ControlDepend(depend_mode=1)
        self.allreduce = P.AllReduce()
        self.parallel_mode = _get_parallel_mode()
        self.grad_reducer = None
        self.reducer_flag = self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]
        if self.reducer_flag:
            mean = _get_mirror_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = self.parallel_mode != ParallelMode.STAND_ALONE

        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")
        self.add_flags(has_effect=True)

    def construct(self, data, label, sens=None):
        weights = self.weights
        loss = self.network(data, label)
        init = False
        if not self.gpu_target:
            # init overflow buffer
            init = self.alloc_status()
            # clear overflow buffer
            self.clear_status(init)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        grads = self.grad(self.network, weights)(data, label, F.cast(scaling_sens, F.dtype(loss)))
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        # get the overflow buffer
        if not self.gpu_target:
            self.get_status(init)
            # sum overflow buffer elements, 0:not overflow , >0:overflow
            flag_sum = self.reduce_sum(init, (0,))
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        # if there is no overflow, do optimize
        if overflow:
            opt = False
        else:
            opt = self.optimizer(grads)
        ret = (loss, cond, scaling_sens)
        return F.depend(ret, opt)
