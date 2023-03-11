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
from __future__ import absolute_import

import mindspore.context as context
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_enable_parallel_optimizer
from mindspore.nn.wrap.cell_wrapper import TrainOneStepCell
from mindspore.nn.cell import Cell
from mindspore.common import Tensor
from mindspore.common.sparse_tensor import RowTensorInner
from mindspore.common.parameter import Parameter
from mindspore.ops.operations.math_ops import NPUGetFloatStatusV2, NPUClearFloatStatusV2
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.common.api import jit

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensorInner(grad.indices,
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
    In each training step, the loss scaling value will be decreased by `loss_scale`/`scale_factor`
    when there is an overflow. And it will be increased by `loss_scale` * `scale_factor` if there is no
    overflow for a continuous `scale_window` steps.

    `get_update_cell` method of :class:`mindspore.amp.DynamicLossScaleManager` will return this class. It will be called
    by :class:`mindspore.nn.TrainOneStepWithLossScaleCell` during training to update loss scale.

    Args:
        loss_scale_value (float): Initializes loss scale.
        scale_factor (int): Coefficient of increase and decrease.
        scale_window (int): Maximum continuous training steps that do not have overflow to increase the loss scale.

    Inputs:
        - **loss_scale** (Tensor) - The loss scale value during training with shape :math:`()`.
        - **overflow** (bool) - Whether the overflow occurs or not.

    Outputs:
        bool, the input `overflow`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, Parameter, nn
        >>> import mindspore.ops as ops
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self, in_features, out_features):
        ...         super(Net, self).__init__()
        ...         self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
        ...                                 name='weight')
        ...         self.matmul = ops.MatMul()
        ...
        ...     def construct(self, x):
        ...         output = self.matmul(x, self.weight)
        ...         return output
        ...
        >>> in_features, out_features = 16, 10
        >>> net = Net(in_features, out_features)
        >>> loss = nn.MSELoss()
        >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> net_with_loss = nn.WithLossCell(net, loss)
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
        """
        Get Loss Scale value.

        Returns:
            float, the loss scale value.
        """
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
    Update cell with fixed loss scaling value.

    `get_update_cell` method of :class:`mindspore.amp.FixedLossScaleManager` will return this class. It will be called
    by :class:`mindspore.nn.TrainOneStepWithLossScaleCell` during trainning.

    Args:
        loss_scale_value (float): Initializes loss scale.

    Inputs:
        - **loss_scale** (Tensor) - The loss scale value during training with shape :math:`()`, it is ignored in this
          class.
        - **overflow** (bool) - Whether the overflow occurs or not.

    Outputs:
        bool, the input `overflow`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, Parameter, nn, ops
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self, in_features, out_features):
        ...         super(Net, self).__init__()
        ...         self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
        ...                                 name='weight')
        ...         self.matmul = ops.MatMul()
        ...
        ...     def construct(self, x):
        ...         output = self.matmul(x, self.weight)
        ...         return output
        ...
        >>> in_features, out_features = 16, 10
        >>> net = Net(in_features, out_features)
        >>> loss = nn.MSELoss()
        >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> net_with_loss = nn.WithLossCell(net, loss)
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
        """
        Get Loss Scale value.

        Returns:
            float, the loss scale value.
        """
        return self.loss_scale_value

    def construct(self, _, overflow):
        return overflow


class TrainOneStepWithLossScaleCell(TrainOneStepCell):
    r"""
    Network training with loss scaling.

    This is a training step with loss scaling. It takes a network, an optimizer and a scale update Cell(or a Tensor) as
    args. The loss scale value can be updated in both host side or device side. If you want to update it on
    host side, using a value of Tensor type as `scale_sense`, otherwise, using a Cell instance for updating loss
    scale as `scale_sense`.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Cell): Optimizer for updating the network parameters.
        scale_sense (Union[Tensor, Cell]): If this value is a Cell, it will be called by `TrainOneStepWithLossScaleCell`
            to update loss scale. If this value is a Tensor, the loss scale can be modified by `set_sense_scale`,
            the shape should be :math:`()` or :math:`(1,)`.

    Inputs:
        - **\*inputs** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scale value.

        - **loss** (Tensor) -  A scalar, the loss value.
        - **overflow** (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        - **loss scale** (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.

    Raises:
        TypeError: If `scale_sense` is neither Cell nor Tensor.
        ValueError: If shape of `scale_sense` is neither (1,) nor ().

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, Parameter, nn, ops
        >>> from mindspore import dtype as mstype
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self, in_features, out_features):
        ...         super(Net, self).__init__()
        ...         self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
        ...                                 name='weight')
        ...         self.matmul = ops.MatMul()
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
        >>> net_with_loss = nn.WithLossCell(net, loss)
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
        >>> net_with_loss = nn.WithLossCell(net, loss)
        >>> inputs = Tensor(np.ones([size, in_features]).astype(np.float32))
        >>> label = Tensor(np.zeros([size, out_features]).astype(np.float32))
        >>> scaling_sens = Tensor([1024], dtype=mstype.float32)
        >>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=scaling_sens)
        >>> output = train_network(inputs, label)
        >>>
        >>> # update scaling sens and train the network
        >>> scaling_sens = Tensor([1], dtype=mstype.float32)
        >>> train_network.set_sense_scale(scaling_sens)
        >>> output = train_network(inputs, label)
    """
    def __init__(self, network, optimizer, scale_sense):
        super(TrainOneStepWithLossScaleCell, self).__init__(network, optimizer, sens=None)
        self.hyper_map = C.HyperMap()
        self.base = Tensor(1, mstype.float32)
        self.base0 = Tensor(0, mstype.int32)
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.reduce_all = P.ReduceAll(keep_dims=False)
        self.less_equal = P.LessEqual()
        self.equal = P.Equal()
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
                raise ValueError("For 'TrainOneStepWithLossScaleCell', "
                                 "the shape of 'scale_sense' must be (1,) or (), but got {}."
                                 .format(scale_sense.shape))
        else:
            raise TypeError("For 'TrainOneStepWithLossScaleCell', "
                            "the 'scale_sense' must be Cell or Tensor, but got 'scale_sense' type: {}."
                            .format(type(scale_sense)))
        self.enable_tuple_broaden = True

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
        """
        If the user has set the `scale_sense` of Tensor type, he can call this function to reassign the value.

        Args:
            sens(Tensor): The new sense whose shape and type are the same with original `scale_sense`.
        """
        if self.scale_sense and isinstance(sens, Tensor):
            self.scale_sense.set_data(sens)
        else:
            raise TypeError("For 'TrainOneStepWithLossScaleCell', "
                            "the type of 'sens' must be Tensor, but got {}".format(type(sens)))

    def start_overflow_check(self, pre_cond, compute_input):
        """
        Start floating-point overflow detection. Create and clear the overflow detection state.

        Specify the argument 'pre_cond' and 'compute_input' to make sure overflow status is cleared at the right time.
        Taking this situation as an example, we need to execute state clearing after loss calculation and then detect
        overflow in the process of gradient calculation. In this case, pre_cond should be the output of the loss
        function, and compute_input should be the input of gradients-computing function. User-defined training network
        based on this class can also call this interface to process the overflow.

        Args:
            pre_cond(Tensor): A precondition for starting overflow detection. It determines the executing order
              of overflow state clearing and prior processions. It makes sure that the function 'start_overflow'
              clears status after finishing the process of precondition.
            compute_input(object): The input of subsequent process. Overflow detection should be performed on a
              certain computation. Set `compute_input` as the input of the computation, to ensure overflow status is
              cleared before executing the computation.

        Returns:
            Tuple[object, object], the first value is False for GPU backend, while it is an instance of
            NPUAllocFloatStatus for other backend. The status is used to detect overflow during `get_overflow_status`.
            The second value is the same as the input of `compute_input`, but contains some information about the
            execution order.
        """
        status = Tensor([0] * 8, mstype.int32)
        if not self.gpu_target:
            status = F.depend(status, pre_cond)
            # clear overflow buffer
            clear_status = NPUClearFloatStatusV2()(status)
            compute_input = F.depend(compute_input, clear_status)
        return status, compute_input

    @jit
    def get_overflow_status(self, status, compute_output):
        """
        Get floating-point overflow status.

        Get overflow results after executing the target process for overflow detection. User-defined training network
        based on this class can also call this interface to process the overflow.

        Args:
            status (object): A status instance used to detect the overflow.
            compute_output: Overflow detection should be performed on a certain computation. Set `compute_output`
              as the output of the computation, to ensure overflow `status` is acquired before executing the
              computation.

        Returns:
            bool, whether the overflow occurs or not.
        """
        if not self.gpu_target:
            status = F.depend(status, compute_output)
            get_status = NPUGetFloatStatusV2()(status)

            if self.is_distributed:
                # sum overflow flag over devices
                flag_reduce = self.allreduce(get_status)
                # get_status not equal to [0]*8 means overflow
                flag = self.equal(self.base0, flag_reduce)
                status = F.depend(status, flag)
                clear_status = NPUClearFloatStatusV2()(status)
                flag = F.depend(flag, clear_status)
                overall_finite = self.reduce_all(flag)
            else:
                status = F.depend(status, get_status)
                clear_status = NPUClearFloatStatusV2()(status)
                get_status = F.depend(get_status, clear_status)
                flag = self.equal(self.base0, get_status)
                overall_finite = self.reduce_all(flag)
            overflow = not overall_finite
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

        User-defined training network based on this class can also call this interface to process the overflow.

        Args:
            overflow(bool): Whether the overflow occurs or not.

        Returns:
            bool, the input overflow value.
        """
        if self.loss_scaling_manager is not None:
            return self.loss_scaling_manager(self.scale_sense, overflow)
        return overflow


grad_scale = C.MultitypeFuncGraph("grad_scale")
shard_grad_scale = C.MultitypeFuncGraph("shard_grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_grad_scale_pipeline(scale, grad, accu_grad):
    accu_grad = F.depend(accu_grad, grad)
    new_grad = accu_grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    zeros = F.tensor_mul(accu_grad, 0.0)
    new_grad = F.depend(new_grad, F.assign(accu_grad, zeros))
    return new_grad


@shard_grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_shard_grad_scale_pipeline(scale, grad, accu_grad):
    new_grad = grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    new_grad = F.depend(new_grad, F.assign(accu_grad, F.zeros_like(accu_grad)))
    return new_grad


class _TrainPipelineWithLossScaleCell(TrainOneStepCell):
    """
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_sense (Cell): Cell to do the loss scale.
    """
    def __init__(self, network, optimizer, scale_sense):
        super(_TrainPipelineWithLossScaleCell, self).__init__(network, optimizer, sens=None)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.grad_reducer = self.identity
        self.degree = 1
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        if self.parallel_mode not in [ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL]:
            raise ValueError(f"ParallelMode must be one of "
                             f"[ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL], but found "
                             f"{self.parallel_mode}.")
        self.allreduce = P.AllReduce()
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.reshape = P.Reshape()
        self.loss_scaling_manager = None
        if isinstance(scale_sense, Cell):
            self.loss_scaling_manager = scale_sense
            self.scale_sense = Parameter(Tensor(scale_sense.get_loss_scale(), dtype=mstype.float32),
                                         name="scale_sense")
        elif isinstance(scale_sense, Tensor):
            if scale_sense.shape == (1,) or scale_sense.shape == ():
                self.scale_sense = Parameter(scale_sense, name='scale_sense')
            else:
                raise ValueError("The shape of 'scale_sense' must be (1,) or (), but got {}"
                                 .format(scale_sense.shape))
        else:
            raise TypeError("The 'scale_sense' must be Cell or Tensor, but got {}".format(type(scale_sense)))
        self.opt_shard = _get_enable_parallel_optimizer()

    def construct(self, *inputs):
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense
        init = self.alloc_status()
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, self.weights)(*inputs, scaling_sens_filled)
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        loss = F.depend(loss, self.clear_before_grad(init))
        if self.opt_shard:
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(F.partial(shard_grad_scale, scaling_sens * self.degree), grads, self.accu_grads)
        else:
            accu_grads = self.grad_reducer(self.accu_grads)
            grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads, accu_grads)
        # sum overflow flag over devices
        flag_reduce = self.allreduce(flag_sum)
        cond = self.less_equal(self.base, flag_reduce)
        overflow = cond
        if self.loss_scaling_manager is not None:
            overflow = self.loss_scaling_manager(self.scale_sense, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, overflow, scaling_sens)

    def identity(self, x):
        return x
