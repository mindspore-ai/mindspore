# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from mindspore.context import ParallelMode
from .cell_wrapper import TrainOneStepCell
from ..cell import Cell
from ...common import Tensor, RowTensor
from ...common.parameter import Parameter
from ...ops import functional as F
from ...ops import composite as C
from ...ops import operations as P
from ...common import dtype as mstype

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))

@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)

@_grad_overflow.register("RowTensor")
def _tensor_grad_overflow_row_tensor(grad):
    return grad_overflow(grad.values)


class DynamicLossScaleUpdateCell(Cell):
    r"""
    Dynamic Loss scale update cell.

    For loss scaling training, the initial loss scaling value will be set to be `loss_scale_value`.
    In each training step, the loss scaling value  will be updated by loss scaling value/`scale_factor`
    when there is an overflow. And it will be increased by loss scaling value * `scale_factor` if there is no
    overflow for a continuous `scale_window` steps. This cell is used for Graph mode training in which all
    logic will be executed on device side(Another training mode is normal(non-sink) mode in which some logic will be
    executed on host).

    Args:
        loss_scale_value (float): Initializes loss scale.
        scale_factor (int): Coefficient of increase and decrease.
        scale_window (int): Maximum continuous training steps that do not have overflow.

    Inputs:
        - **inputs** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a scalar Tensor with shape :math:`()`.

    Raises:
        TypeError: If dtype of `inputs` or `label` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, Parameter, nn
        >>> from mindspore.ops import operations as P
        >>> from mindspore.nn.wrap.cell_wrapper import WithLossCell
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self, in_features, out_features):
        ...         super(Net, self).__init__()
        ...         self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
        ...                                 name='weight')
        ...         self.matmul = P.MatMul()
        ...
        ...     def construct(self, x):
        ...         output = self.matmul(x, self.weight)
        ...         return output
        ...
        >>> in_features, out_features = 16, 10
        >>> net = Net(in_features, out_features)
        >>> loss = nn.MSELoss()
        >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> net_with_loss = WithLossCell(net, loss)
        >>> manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
        >>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)
        >>> input = Tensor(np.ones([out_features, in_features]), mindspore.float32)
        >>> labels = Tensor(np.ones([out_features,]), mindspore.float32)
        >>> output = train_network(input, labels)
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
        last_iter = F.assign(self.last_overflow_iter, last_overflow_iter)
        update_scale_cond = self.logic_and(should_inc, self.logic_not(overflow_cond))
        scale_mul_res = loss_scale_on_overflow * self.scale_factor
        scaled_loss_scale = self.select(update_scale_cond, scale_mul_res, loss_scale_on_overflow)
        F.assign(loss_scale, scaled_loss_scale)
        inc_cur_iter = self.cur_iter + 1
        inc_cur_iter = F.depend(inc_cur_iter, last_iter)
        F.assign(self.cur_iter, inc_cur_iter)
        return overflow


class FixedLossScaleUpdateCell(Cell):
    """
    Static scale update cell, the loss scaling value will not be updated.

    For usage, refer to `DynamicLossScaleUpdateCell`.

    Args:
        loss_scale_value (float): Initializes loss scale.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, Parameter, nn
        >>> from mindspore.ops import operations as P
        >>> from mindspore.nn.wrap.cell_wrapper import WithLossCell
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self, in_features, out_features):
        ...         super(Net, self).__init__()
        ...         self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
        ...                                 name='weight')
        ...         self.matmul = P.MatMul()
        ...
        ...     def construct(self, x):
        ...         output = self.matmul(x, self.weight)
        ...         return output
        ...
        >>> in_features, out_features = 16, 10
        >>> net = Net(in_features, out_features)
        >>> loss = nn.MSELoss()
        >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> net_with_loss = WithLossCell(net, loss)
        >>> manager = nn.FixedLossScaleUpdateCell(loss_scale_value=2**12)
        >>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)
        >>> input = Tensor(np.ones([out_features, in_features]), mindspore.float32)
        >>> labels = Tensor(np.ones([out_features,]), mindspore.float32)
        >>> output = train_network(input, labels)
    """

    def __init__(self, loss_scale_value):
        super(FixedLossScaleUpdateCell, self).__init__()
        self.loss_scale_value = loss_scale_value

    def get_loss_scale(self):
        return self.loss_scale_value

    def construct(self, _, overflow):
        return overflow


class TrainOneStepWithLossScaleCell(TrainOneStepCell):
    r"""
    Network training with loss scaling.

    This is a training step with loss scaling. It takes a network, an optimizer and possibly a scale update
    Cell as args. The loss scale value can be updated in both host side or device side. The
    TrainOneStepWithLossScaleCell will be compiled to be graph which takes `*inputs` as input data.
    The Tensor type of `scale_sense` is acting as loss scaling value. If you want to update it on host side,
    the value must be provided. If  the Tensor type of `scale_sense` is not given, the loss scale update logic
    must be provied by Cell type of `scale_sense`.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Cell): Optimizer for updating the weights.
        scale_sense (Union[Tensor, Cell]): If this value is Cell type, the loss scaling update logic cell.If this value
                                          is Tensor type, Tensor with shape :math:`()` or :math:`(1,)`.

    Inputs:
        - **(*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scaling value.

        - **loss** (Tensor) -  Tensor with shape :math:`()`.
        - **overflow** (Tensor) -  Tensor with shape :math:`()`, type is bool.
        - **loss scaling value** (Tensor) -  Tensor with shape :math:`()`

    Raises:
        TypeError: If `scale_sense` is neither Cell nor Tensor.
        ValueError: If shape of `scale_sense` is neither (1,) nor ().

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, Parameter, nn
        >>> from mindspore.ops import operations as P
        >>> from mindspore.nn.wrap.cell_wrapper import WithLossCell
        >>> from mindspore.common import dtype as mstype
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self, in_features, out_features):
        ...         super(Net, self).__init__()
        ...         self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
        ...                                 name='weight')
        ...         self.matmul = P.MatMul()
        ...
        ...     def construct(self, x):
        ...         output = self.matmul(x, self.weight)
        ...         return output
        ...
        >>> size, in_features, out_features = 16, 16, 10
        >>> #1) when the type of scale_sense is Cell:
        >>> net = Net(in_features, out_features)
        >>> loss = nn.MSELoss()
        >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> net_with_loss = WithLossCell(net, loss)
        >>> manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
        >>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)
        >>> input = Tensor(np.ones([out_features, in_features]), mindspore.float32)
        >>> labels = Tensor(np.ones([out_features,]), mindspore.float32)
        >>> output = train_network(input, labels)
        >>>
        >>> #2) when the type of scale_sense is Tensor:
        >>> net = Net(in_features, out_features)
        >>> loss = nn.MSELoss()
        >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> net_with_loss = WithLossCell(net, loss)
        >>> inputs = Tensor(np.ones([size, in_features]).astype(np.float32))
        >>> label = Tensor(np.zeros([size, out_features]).astype(np.float32))
        >>> scaling_sens = Tensor(np.full((1), np.finfo(np.float32).max), dtype=mstype.float32)
        >>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=scaling_sens)
        >>> output = train_network(inputs, label)
    """
    def __init__(self, network, optimizer, scale_sense):
        super(TrainOneStepWithLossScaleCell, self).__init__(network, optimizer, sens=None)
        self.hyper_map = C.HyperMap()
        self.base = Tensor(1, mstype.float32)
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.less_equal = P.LessEqual()
        self.allreduce = P.AllReduce()
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.gpu_target = (context.get_context("device_target") == "GPU")
        self.loss_scaling_manager = None
        if isinstance(scale_sense, Cell):
            self.loss_scaling_manager = scale_sense
            self.scale_sense = Parameter(Tensor(scale_sense.get_loss_scale(), dtype=mstype.float32),
                                         name="scale_sense")
        elif isinstance(scale_sense, Tensor):
            if scale_sense.shape == (1,) or scale_sense.shape == ():
                self.scale_sense = Parameter(scale_sense, name='scale_sense')
            else:
                raise ValueError("The shape of scale_sense must be (1,) or (), but got {}".format(scale_sense.shape))
        else:
            raise TypeError("The scale_sense must be Cell or Tensor, but got {}".format(type(scale_sense)))

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            loss = F.depend(loss, self.optimizer(grads))
        return loss, cond, scaling_sens

    def set_sense_scale(self, sens):
        """If the user has set the sens in the training process and wants to reassign the value, he can call
        this function again to make modification, and sens needs to be of type Tensor."""
        if self.scale_sense and isinstance(sens, Tensor):
            self.scale_sense.set_data(sens)
        else:
            raise TypeError("The input type must be Tensor, but got {}".format(type(sens)))

    def start_overflow_check(self, pre_cond, compute_input):
        """
        Start floating-point overflow detection. Create and clear the overflow detection state.

        Specify the argument 'pre_cond' and 'compute_input' to make sure overflow status is cleared at the right time.
        Taking this situation as an example, we need to execute state clearing after loss calculation and then detect
        overflow in the process of gradient calculation. In this case, pre_cond should be the output of the loss
        function, and compute_input should be the input of gradients-computing function.

        Args:
            pre_cond(object): A precondition for starting overflow detection. It determines the executing order of
                overflow state clearing and prior processions. It makes sure that the function 'start_overflow' clears
                status after finishing the process of precondition.
            compute_input(object): The input of subsequent process. Overflow detection should be performed on a certain
                computation. Set `compute_input` as the input of the computation, to ensure overflow status is cleared
                before executing the computation.

        Returns:
            Tuple[object, object], the first value is False for GPU backend, while it is a instance of
            NPUAllocFloatStatus for other backend. The status is used to detect overflow during overflow detection.
            The second value is the same as the input of `compute_input`, but contains some information about the
            execution order.
        """
        status = False
        if not self.gpu_target:
            # init overflow buffer
            status = P.NPUAllocFloatStatus()()
            status = F.depend(status, pre_cond)
            # clear overflow buffer
            clear_status = P.NPUClearFloatStatus()(status)
            compute_input = F.depend(compute_input, clear_status)
        return status, compute_input

    def get_overflow_status(self, status, compute_output):
        """
        Get floating-point overflow status.

        Get overflow results after executing the target process for overflow detection.

        Args:
            status(object): A status instance used to detect the overflow.
            compute_output: Overflow detection should be performed on a certain computation. Set `compute_output` as
                the output of the computation, to ensure overflow status is acquired before executing the computation.

        Returns:
            bool, whether the overflow occurs or not.
        """
        if not self.gpu_target:
            status = F.depend(status, compute_output)
            get_status = P.NPUGetFloatStatus()(status)
            status = F.depend(status, get_status)
            # sum overflow buffer elements, 0:not overflow , >0:overflow
            flag_sum = self.reduce_sum(status, (0,))
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), compute_output)
            flag_sum = P.AddN()(flag_sum)
            # convert flag_sum to scalar
            flag_sum = P.Reshape()(flag_sum, (()))

        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            overflow = self.less_equal(self.base, flag_reduce)
        else:
            overflow = self.less_equal(self.base, flag_sum)
        return overflow

    def process_loss_scale(self, overflow):
        """
        Calculate loss scale according to the overflow.

        Args:
            overflow(bool): Whether the overflow occurs or not.

        Returns:
            bool, overflow value.
        """
        if self.loss_scaling_manager is not None:
            return self.loss_scaling_manager(self.scale_sense, overflow)
        return overflow
