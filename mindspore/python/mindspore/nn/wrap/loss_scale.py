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

import os
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_enable_parallel_optimizer
from mindspore import nn
from mindspore.nn.wrap.cell_wrapper import TrainOneStepCell
from mindspore.nn.cell import Cell
from mindspore.common import Tensor
from mindspore.common.sparse_tensor import RowTensorInner
from mindspore.common.parameter import Parameter
from mindspore.ops.operations.math_ops import NPUGetFloatStatusV2, NPUClearFloatStatusV2
from mindspore import ops
from mindspore.ops.operations.nn_ops import AllFinite
from mindspore.common import dtype as mstype
from mindspore._c_expression import MSContext
from mindspore.run_check._check_version import AscendEnvChecker
from mindspore import log as logger

_grad_scale = ops.MultitypeFuncGraph("grad_scale")
reciprocal = ops.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(reciprocal(scale), ops.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensorInner(grad.indices,
                          grad.values * ops.cast(reciprocal(scale), ops.dtype(grad.values)),
                          grad.dense_shape)


_grad_overflow = ops.MultitypeFuncGraph("_grad_overflow")
grad_overflow = ops.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


@_grad_overflow.register("RowTensor")
def _tensor_grad_overflow_row_tensor(grad):
    return grad_overflow(grad.values)


_ascend_grad_overflow = ops.MultitypeFuncGraph("_ascend_grad_overflow")
ascend_grad_overflow = ops.IsFinite()


@_ascend_grad_overflow.register("Tensor")
def _tensor_ascend_grad_overflow(grad):
    status = ascend_grad_overflow(grad)
    base = Tensor(1.0, dtype=mstype.float32)
    output = base - status.all()
    output = ops.Reshape()(output, ((-1,)))
    return output


@_ascend_grad_overflow.register("RowTensor")
def _tensor_ascend_grad_overflow_row_tensor(grad):
    status = ascend_grad_overflow(grad.values)
    base = Tensor(1.0, dtype=mstype.float32)
    output = base - status.all()
    output = ops.Reshape()(output, ((1,)))
    return output


class DynamicLossScaleUpdateCell(Cell):
    r"""
    Dynamic Loss scale update cell.

    For loss scaling training, the initial loss scaling value will be set to be `loss_scale_value`.
    In each training step, the loss scaling value will be decreased by :math:`loss\_scale/scale\_factor`
    when there is an overflow. And it will be increased by :math:`loss\_scale * scale\_factor` if there is no
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
        >>> import mindspore
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
        super().__init__()

        self.scale_window = Tensor(scale_window, dtype=mstype.int32)
        self.scale_factor = Tensor(scale_factor, dtype=mstype.float32)
        self.loss_scale_value = loss_scale_value

        self.cur_iter = Parameter(Tensor(1, dtype=mstype.int32), name="current_iterator_step")
        self.last_overflow_iter = Parameter(Tensor(0, dtype=mstype.int32), name="last_overflow_iterator_step")
        self.select = ops.Select()
        self.max = ops.Maximum()
        self.minimum_loss_scale = Tensor(1.0, dtype=mstype.float32)
        self.reciprocal = ops.Reciprocal()
        self.less_equal = ops.LessEqual()
        self.logic_and = ops.LogicalAnd()
        self.logic_not = ops.LogicalNot()
        self.logic_or = ops.LogicalOr()
        self.const_true = Tensor(True, dtype=mstype.bool_)

    def get_loss_scale(self):
        """
        Get Loss Scale value.

        Returns:
            float, the loss scale value.

        Examples:
            >>> from mindspore import nn
            >>> manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=212, scale_factor=2, scale_window=1000)
            >>> output = manager.get_loss_scale()
            >>> print(output)
            212
        """
        return self.loss_scale_value

    def construct(self, loss_scale, overflow):
        overflow_cond = overflow
        loss_scale_on_overflow = self.select(overflow_cond, self.max(loss_scale * self.reciprocal(self.scale_factor),
                                                                     self.minimum_loss_scale), loss_scale)
        should_inc = self.less_equal(self.scale_window, self.cur_iter - self.last_overflow_iter)
        last_iter_cond = self.logic_or(overflow_cond, should_inc)
        last_overflow_iter = self.select(last_iter_cond, self.cur_iter, self.last_overflow_iter)
        last_iter = ops.assign(self.last_overflow_iter, last_overflow_iter)
        update_scale_cond = self.logic_and(should_inc, self.logic_not(overflow_cond))
        scale_mul_res = loss_scale_on_overflow * self.scale_factor
        scaled_loss_scale = self.select(update_scale_cond, scale_mul_res, loss_scale_on_overflow)
        ops.assign(loss_scale, scaled_loss_scale)
        inc_cur_iter = self.cur_iter + 1
        inc_cur_iter = ops.depend(inc_cur_iter, last_iter)
        ops.assign(self.cur_iter, inc_cur_iter)
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
        >>> import mindspore
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
        super().__init__()
        self.loss_scale_value = loss_scale_value

    def get_loss_scale(self):
        """
        Get Loss Scale value.

        Returns:
            float, the loss scale value.

        Examples:
            >>> from mindspore import nn
            >>> manager = nn.FixedLossScaleUpdateCell(loss_scale_value=212)
            >>> output = manager.get_loss_scale()
            >>> print(output)
            212
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
        ValueError: If shape of `scale_sense` is neither :math:`(1,)` nor :math:`()`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
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
        >>> size, in_features, out_features = 16, 16, 10
        >>> #1) when the type of scale_sense is Cell:
        >>> net = Net(in_features, out_features)
        >>> loss_fn = nn.MSELoss()
        >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> net_with_loss = nn.WithLossCell(net, loss_fn)
        >>> input = Tensor(np.ones([out_features, in_features]), mindspore.float32)
        >>> labels = Tensor(np.ones([out_features,]), mindspore.float32)
        >>> loss = net_with_loss(input, labels)
        >>> manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
        >>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)
        >>> status = Tensor([0] * 8, mindspore.int32)
        >>> scaling_sens = train_network.scale_sense
        >>> scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        >>> grads = train_network.grad(train_network.network, train_network.weights)(input, labels, scaling_sens_filled)
        >>> grads = train_network.grad_reducer(grads)
        >>> cond = train_network.get_overflow_status(status, grads)
        >>> overflow = train_network.process_loss_scale(cond)
        >>>
        >>> #2) when the type of scale_sense is Tensor:
        >>> net = Net(in_features, out_features)
        >>> loss = nn.MSELoss()
        >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> net_with_loss = nn.WithLossCell(net, loss)
        >>> inputs = Tensor(np.ones([size, in_features]).astype(np.float32))
        >>> label = Tensor(np.zeros([size, out_features]).astype(np.float32))
        >>> scaling_sens = Tensor([1024], dtype=mindspore.float32)
        >>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=scaling_sens)
        >>> scaling_sens = Tensor([1], dtype=mstype.float32)
        >>> train_network.set_sense_scale(scaling_sens)
        >>> output = train_network(inputs, label)
        >>>
        >>> # update scaling sens and train the network
        >>> scaling_sens = Tensor([1], dtype=mindspore.float32)
        >>> train_network.set_sense_scale(scaling_sens)
        >>> output = train_network(inputs, label)
    """

    def __init__(self, network, optimizer, scale_sense):
        super().__init__(network, optimizer, sens=None)
        self.hyper_map = ops.HyperMap()
        self.base = Tensor(1, mstype.float32)
        self.base0 = Tensor(0, mstype.int32)
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.reduce_all = ops.ReduceAll(keep_dims=False)
        self.less_equal = ops.LessEqual()
        self.equal = ops.Equal()
        self.logic_not = ops.LogicalNot()
        self.allreduce = ops.AllReduce()
        self.is_distributed = self.parallel_mode != ParallelMode.STAND_ALONE
        self.gpu_target = context.get_context("device_target") == "GPU"
        self.ascend_910a_target = MSContext.get_instance().ascend_soc_supports_feature(
            "amp_loss_scale_train_classic_target")
        self.ascend_910b_target = MSContext.get_instance().ascend_soc_supports_feature(
            "train_hq_dtype_line")
        self.loss_scaling_manager = None
        self._ascend_check_overflow_mode = os.environ.get('MS_ASCEND_CHECK_OVERFLOW_MODE')

        self.enable_allfinite = True
        runtime_conf = os.environ.get('MS_DEV_RUNTIME_CONF')
        global_jit_config = context.get_jit_config()
        if runtime_conf is not None and ("all_finite:True" in runtime_conf or "all_finite:true" in runtime_conf):
            logger.debug("Enable AllFinite through the environment variable MS_DEV_RUNTIME_CONF.")
            self.enable_allfinite = True
        elif runtime_conf is not None and ("all_finite:False" in runtime_conf or "all_finite:false" in runtime_conf):
            logger.debug("Disable AllFinite through the environment variable MS_DEV_RUNTIME_CONF.")
            self.enable_allfinite = False
        elif global_jit_config:
            logger.debug("Current global jit config is: {}".format(global_jit_config["jit_level"]))
            self.enable_allfinite = global_jit_config["jit_level"] == "O0" or global_jit_config["jit_level"] == "O1"
        if "RANK_TABLE_FILE" in os.environ:
            self.enable_allfinite = False
        if self.ascend_910b_target:
            checker = AscendEnvChecker(None)
            if not checker.check_custom_version():
                logger.debug("Disable AllFinite due to version check failure.")
                self.enable_allfinite = False

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
        self._get_attr_from_cell(network)

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense
        status = Tensor([0] * 8, mstype.int32)

        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            loss = ops.depend(loss, self.optimizer(grads))
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
            Tuple[object, object], the first output is used to control the execution sequence. To ensure that the
            `start_overflow_check` is executed before get_overflow_status after compilation optimization is performed.
            This value should be used as the first input of get_overflow_status. The second output is the same as
            the input of compute_input, used to control the execution sequence, and make ensure that the overflow flag
            is cleaned up when the function returns.
        """
        status = Tensor([0] * 8, mstype.int32)
        if self.ascend_910a_target or (self.ascend_910b_target and \
                                       self._ascend_check_overflow_mode == "SATURATION_MODE"):
            status = ops.depend(status, pre_cond)
            # clear overflow buffer
            clear_status = NPUClearFloatStatusV2()(status)
            compute_input = ops.depend(compute_input, clear_status)
        return status, compute_input

    def _check_overflow_status_on_infnan_mode(self, grad_overflow_check_func, compute_output):
        """check overflow status on infnan mode."""
        flag_sum = self.hyper_map(ops.partial(grad_overflow_check_func), compute_output)
        flag_sum = ops.AddN()(flag_sum)
        # convert flag_sum to scalar
        flag_sum = ops.Reshape()(flag_sum, (()))
        return flag_sum

    def _get_distributed_overflow_status_on_infnan_mode(self, grad_overflow_check_func, compute_output):
        """converge the distributed overflow status on infnan mode."""
        flag_sum = self._check_overflow_status_on_infnan_mode(grad_overflow_check_func, compute_output)

        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            overflow = self.less_equal(self.base, flag_reduce)
        else:
            overflow = self.less_equal(self.base, flag_sum)
        return overflow

    def _get_distributed_overflow_status_on_infnan_enable_allfinite(self, compute_output):
        """check overflow status on infnan kernel mode."""
        overflow = AllFinite()(compute_output)

        if self.is_distributed:
            overflow = ops.Cast()(overflow, mstype.float32)
            overflow = ops.Cast()(self.allreduce(overflow), mstype.bool_)
        return overflow

    def _get_gpu_overflow_status(self, compute_output):
        """get overflow status of gpu."""
        overflow = self._get_distributed_overflow_status_on_infnan_mode(_grad_overflow, compute_output)
        return overflow

    def _get_ascend_overflow_status_on_infnan_mode(self, compute_output):
        """get overflow status of ascend on infnan mode."""
        overflow = False
        if self.enable_allfinite:
            overflow = self._get_distributed_overflow_status_on_infnan_enable_allfinite(compute_output)
        else:
            overflow = self._get_distributed_overflow_status_on_infnan_mode(_ascend_grad_overflow, compute_output)
        return overflow

    def _get_ascend_overflow_status_on_saturation_mode(self, status, compute_output):
        """get overflow status of ascend on saturation mode"""
        status = ops.depend(status, compute_output)
        get_status = NPUGetFloatStatusV2()(status)

        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(get_status)
            # get_status not equal to [0]*8 means overflow
            flag = self.equal(self.base0, flag_reduce)
            status = ops.depend(status, flag)
            # distributed needs to skip allreduce to avoid its overflow affecting the next step
            clear_status = NPUClearFloatStatusV2()(status)
            flag = ops.depend(flag, clear_status)
            overall_finite = self.reduce_all(flag)
        else:
            status = ops.depend(status, get_status)
            clear_status = NPUClearFloatStatusV2()(status)
            get_status = ops.depend(get_status, clear_status)
            flag = self.equal(self.base0, get_status)
            overall_finite = self.reduce_all(flag)
        overflow = self.logic_not(overall_finite)
        return overflow

    def get_overflow_status(self, status, compute_output):
        """
        Get floating-point overflow status.

        Get overflow results after executing the target process for overflow detection. User-defined training network
        based on this class can also call this interface to process the overflow.

        Args:
            status (object): To control the execution sequence with start_overflow_check, it should be set as the first
              output of start_overflow_check.
            compute_output: Overflow detection should be performed in a certain computation process. Set
              `compute_output` as the output of the computation process.

        Returns:
            bool, whether the overflow occurs or not.
        """
        if self.gpu_target:
            overflow = self._get_gpu_overflow_status(compute_output)
        elif self.ascend_910b_target:
            if self._ascend_check_overflow_mode == "SATURATION_MODE":
                overflow = self._get_ascend_overflow_status_on_saturation_mode(status, compute_output)
            else:
                overflow = self._get_ascend_overflow_status_on_infnan_mode(compute_output)
        else:  # ascend_910a_target
            overflow = self._get_ascend_overflow_status_on_saturation_mode(status, compute_output)
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


grad_scale = ops.MultitypeFuncGraph("grad_scale")
shard_grad_scale = ops.MultitypeFuncGraph("shard_grad_scale")
reciprocal = ops.Reciprocal()


@grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_grad_scale_pipeline(scale, grad, accu_grad):
    accu_grad = ops.depend(accu_grad, grad)
    new_grad = accu_grad * reciprocal(scale)
    accu_grad = ops.depend(accu_grad, new_grad)
    zeros = ops.tensor_mul(accu_grad, 0.0)
    new_grad = ops.depend(new_grad, ops.assign(accu_grad, zeros))
    return new_grad


@shard_grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_shard_grad_scale_pipeline(scale, grad, accu_grad):
    new_grad = grad * reciprocal(scale)
    accu_grad = ops.depend(accu_grad, new_grad)
    new_grad = ops.depend(new_grad, ops.assign(accu_grad, ops.zeros_like(accu_grad)))
    return new_grad


class _TrainGradAccuWithLossScaleCell(TrainOneStepCell):
    """
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_sense (Cell): Cell to do the loss scale.
    """

    def __init__(self, network, optimizer, scale_sense):
        super().__init__(network, optimizer, sens=None)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.grad_reducer = nn.Identity()
        self.degree = 1
        self.cast = ops.Cast()
        self.alloc_status = ops.NPUAllocFloatStatus()
        self.get_status = ops.NPUGetFloatStatus()
        self.clear_before_grad = ops.NPUClearFloatStatus()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        if self.parallel_mode not in [ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL]:
            raise ValueError(f"ParallelMode must be one of "
                             f"[ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL], but found "
                             f"{self.parallel_mode}.")
        self.allreduce = ops.AllReduce()
        self.base = Tensor(1, mstype.float32)
        self.less_equal = ops.LessEqual()
        self.hyper_map = ops.HyperMap()
        self.reshape = ops.Reshape()
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
        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        scaling_sens_filled = ops.depend(scaling_sens_filled, self.clear_before_grad(init))
        grads = self.grad(self.network, self.weights)(*inputs, scaling_sens_filled)
        init = ops.depend(init, grads)
        get_status = self.get_status(init)
        init = ops.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        if self.opt_shard:
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(ops.partial(shard_grad_scale, scaling_sens * self.degree), grads, self.accu_grads)
        else:
            accu_grads = self.grad_reducer(self.accu_grads)
            grads = self.hyper_map(ops.partial(grad_scale, scaling_sens * self.degree), grads, accu_grads)
        # sum overflow flag over devices
        flag_reduce = self.allreduce(flag_sum)
        cond = self.less_equal(self.base, flag_reduce)
        overflow = cond
        if self.loss_scaling_manager is not None:
            overflow = self.loss_scaling_manager(self.scale_sense, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, overflow, scaling_sens)
