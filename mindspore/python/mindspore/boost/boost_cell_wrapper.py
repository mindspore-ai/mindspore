# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Boost Mode Cell Wrapper."""
from __future__ import absolute_import

import numpy as np
from mindspore.nn.wrap import TrainOneStepCell
import mindspore.context as context
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_global_rank, _get_device_num, _get_gradients_mean
from mindspore.communication.management import get_group_size, create_group
from mindspore.nn.cell import Cell
from mindspore.nn import SequentialCell
from mindspore.common import Tensor
from mindspore.common.sparse_tensor import RowTensorInner
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.boost.boost import AutoBoost
from mindspore.boost.grad_freeze import FreezeOpt, freeze_cell
from mindspore.boost.adasum import AdaSum
from mindspore.boost.dim_reduce import DimReduce
from mindspore.boost.grad_accumulation import gradient_accumulation_op, gradient_clear_op
from mindspore.boost.base import _load_local_pca_mat

__all__ = ["BoostTrainOneStepCell", "BoostTrainOneStepWithLossScaleCell"]

_get_delta_weight = C.MultitypeFuncGraph("_get_delta_weight")


@_get_delta_weight.register("Tensor", "Tensor")
def _get_delta_weight_process(new_parameter, old_parameter):
    delta_w = old_parameter - new_parameter
    return delta_w


_save_weight = C.MultitypeFuncGraph("_save_weight")


@_save_weight.register("Tensor", "Tensor")
def _save_weight_process(new_parameter, old_parameter):
    P.Assign()(new_parameter, old_parameter)
    return new_parameter


_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """grad scale function for tensor"""
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    """grad scale function for row tensor"""
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


class _OutputToFloat16(Cell):
    "Wrap cell for amp. Cast network output back to float16"

    def __init__(self, op):
        super(_OutputToFloat16, self).__init__(auto_prefix=False)
        self._op = op

    def construct(self, *inputs):
        return F.cast(self._op(*inputs), mstype.float16)


class BoostTrainOneStepCell(TrainOneStepCell):
    r"""
    Boost Network training package class.

    Wraps the network with an optimizer. The resulting Cell is trained with input '\*inputs'.
    The backward graph will be created in the construct function to update the parameter, and different
    parallel modes are available for training.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Union[Cell]): Optimizer for updating the weights.
        sens (numbers.Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Inputs:
        - **\*inputs** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a tensor means the loss value, the shape of which is usually :math:`()`.

        - loss(Tensor): A scalar Tensor.
        - overflow(Tensor): A scalar Tensor which type is bool.
        - loss scaling value(Tensor): A scalar Tensor.

    Raises:
        TypeError: If `sens` is not a number.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import boost
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> #1) Using the WithLossCell existing provide
        >>> loss_net = nn.WithLossCell(net, loss_fn)
        >>> train_net = boost.BoostTrainOneStepCell(loss_net, optim)
        >>>
        >>> #2) Using user-defined WithLossCell
        >>> class MyWithLossCell(Cell):
        ...    def __init__(self, backbone, loss_fn):
        ...        super(MyWithLossCell, self).__init__(auto_prefix=False)
        ...        self._backbone = backbone
        ...        self._loss_fn = loss_fn
        ...
        ...    def construct(self, x, y, label):
        ...        out = self._backbone(x, y)
        ...        return self._loss_fn(out, label)
        ...
        ...    @property
        ...    def backbone_network(self):
        ...        return self._backbone
        ...
        >>> loss_net = MyWithLossCell(net, loss_fn)
        >>> train_net = boost.BoostTrainOneStepCell(loss_net, optim)
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(BoostTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.hyper_map = C.HyperMap()
        self.freeze = isinstance(optimizer, FreezeOpt)
        if not self.freeze:
            self.weights = self.optimizer.parameters
        self.train_strategy = getattr(self.optimizer, 'train_strategy', None)

        self.auto_boost = AutoBoost()
        self.use_grad_accumulation = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.STAND_ALONE)
        self.use_grad_accumulation = self.use_grad_accumulation & \
                                     self.auto_boost.boost_config.get("grad_accumulation", False)
        self.max_accumulation_step = 1
        if self.use_grad_accumulation:

            self.max_accumulation_step = self.auto_boost.grad_accumulation_step
            if self.max_accumulation_step <= 1:
                self.max_accumulation_step = 1
                self.use_grad_accumulation = False
        self.accumulation_step = Parameter(Tensor(0, dtype=mstype.int32), name="accumulation_step")
        if self.use_grad_accumulation:
            self.grad_accumulation = self.weights.clone(prefix="grad_accumulation", init='zeros')

        self.enable_dim_reduce = self.check_dim_reduce_enable()
        if self.enable_dim_reduce:
            self.__init_dim_reduce()

        self.freeze_nets = None
        self.step = Parameter(Tensor(0, dtype=mstype.int32))
        if self.freeze:
            if self.reducer_flag:
                self.mean = _get_gradients_mean()
                self.degree = _get_device_num()
            else:
                self.mean = None
                self.degree = None
            self.freeze_nets = freeze_cell(self.reducer_flag, self.network, self.optimizer, self.sens,
                                           self.grad, self.use_grad_accumulation, self.mean, self.degree,
                                           self.max_accumulation_step)

        self.enable_adasum = self.check_adasum_enable()
        self.sync_tensor = Parameter(Tensor(0, dtype=mstype.int32))
        if self.enable_adasum:
            self.__init_adasum()

    def construct(self, *inputs):
        if self.freeze:
            loss = self.gradient_freeze_process(*inputs)
        else:
            loss = self.network(*inputs)
            sens = F.fill(loss.dtype, loss.shape, self.sens)
            grads = self.grad(self.network, self.weights)(*inputs, sens)
            grads = self.grad_reducer(grads)
            if self.use_grad_accumulation:
                loss = self.gradient_accumulation_process(loss, grads, sens, *inputs)
            else:
                if self.enable_dim_reduce:
                    loss = F.depend(loss, self.dim_reduce(loss, grads, sens, self.weights, self.weights_clone, *inputs))
                elif self.enable_adasum:
                    loss = F.depend(loss, self.adasum_process(loss, grads))
                else:
                    loss = F.depend(loss, self.optimizer(grads))
        return loss

    def gradient_freeze_process(self, *inputs):
        r"""
        Gradient freeze algorithm process.

        Args:
            inputs (tuple(Tensor)): Tuple of input tensors with shape :math:`(N, \ldots)`.

        Returns:
            - **loss** (Tensor) -  Network loss, tensor with shape :math:`()`.
        """
        if self.train_strategy is None:
            step = self.step
            max_index = len(self.freeze_nets)
        else:
            step = self.train_strategy[self.step]
            max_index = len(self.train_strategy)
        loss = self.freeze_nets[step](*inputs)
        if self.step + 1 >= max_index:
            self.step = 0
        else:
            self.step += 1
        return loss

    def gradient_accumulation_process(self, loss, grads, sens, *inputs):
        r"""
        Gradient accumulation algorithm process.

        Args:
            loss (Tensor): Tensor with shape :math:`()`.
            grads (tuple(Tensor)): Tuple of gradient tensors.
            sens (Tensor): Tensor with shape :math:`()`.
            inputs (tuple(Tensor)): Tuple of input tensors with shape :math:`(N, \ldots)`.

        Returns:
            - **loss** (Tensor) - Network loss, tensor with shape :math:`()`.
        """
        loss = F.depend(loss, self.hyper_map(F.partial(gradient_accumulation_op, self.max_accumulation_step),
                                             self.grad_accumulation, grads))
        self.accumulation_step += 1

        if self.accumulation_step >= self.max_accumulation_step:
            if self.enable_dim_reduce:
                loss = F.depend(loss, self.dim_reduce(loss, self.grad_accumulation, sens, self.weights,
                                                      self.weights_clone, *inputs))
            elif self.enable_adasum:
                loss = F.depend(loss, self.adasum_process(loss, self.grad_accumulation))
            else:
                loss = F.depend(loss, self.optimizer(self.grad_accumulation))
            self.accumulation_step = 0

        if self.accumulation_step == 0:
            loss = F.depend(loss, self.hyper_map(F.partial(gradient_clear_op), self.grad_accumulation))

        return loss

    def adasum_process(self, loss, grads):
        r"""
        Adasum algorithm process.

        Args:
            loss (Tensor): Tensor with shape :math:`()`.
            grads (tuple(Tensor)): Tuple of gradient tensors.

        Returns:
            - **loss** (Tensor) - Network loss, tensor with shape :math:`()`.
        """
        loss = F.depend(loss, self.optimizer(grads))
        rank_weights = self.weights[self.start[self.server_rank]: self.end[self.server_rank]]
        grad_clone = F.depend(self.grad_clone, loss)
        delta_w = self.hyper_map(F.partial(_get_delta_weight), rank_weights, grad_clone)
        adasum_res = self.adasum(delta_w, rank_weights, grad_clone)
        sync_tensor = F.depend(self.sync_tensor, adasum_res)
        sync_flag = self.adasum.sync_barrier(sync_tensor)
        for i in range(self.device_number):
            weight_tuple = self.weights[self.start[i]: self.end[i]]
            node_rank = F.depend(weight_tuple, sync_flag)
            update_weights = self.adasum.broadcast_list[i](node_rank)
            if i == self.server_rank:
                self.hyper_map(F.partial(_save_weight), self.grad_clone, update_weights)
            else:
                self.hyper_map(F.partial(_save_weight), weight_tuple, update_weights)
        return loss

    def check_adasum_enable(self):
        r"""
        Check adasum enable.

        Returns:
            - **enable_adasum** (bool) - Check whether the Adasum algorithm is enabled.
        """
        if not getattr(self.optimizer, "adasum", None) or not self.reducer_flag:
            return False
        _rank_size = get_group_size()
        _device_number = 8
        group_number = _rank_size // _device_number
        is_enable = bool(group_number > 1 and group_number & (group_number - 1) == 0)
        return is_enable

    def check_dim_reduce_enable(self):
        r"""
        Check dim_reduce enable.

        Returns:
            - **enable_dim_reduce** (bool) - Check whether the dimensionality reduction second-order training
              algorithm is enabled.
        """
        if not getattr(self.optimizer, "dim_reduce", None):
            return False
        return True

    def __init_dim_reduce(self):
        """dim reduce algorithm init method."""
        local_pca_mat_path = self.auto_boost.local_pca_mat_path
        rho = self.auto_boost.rho
        gamma = self.auto_boost.gamma
        alpha = self.auto_boost.alpha
        sigma = self.auto_boost.sigma
        _rank = _get_global_rank()
        _rank_size = 1 if self.parallel_mode == ParallelMode.STAND_ALONE else get_group_size()
        n_components = self.auto_boost.n_components
        timeout = self.auto_boost.timeout
        pca_mat = _load_local_pca_mat(local_pca_mat_path, timeout)
        self.weights_clone = ParameterTuple(self.weights).clone(prefix="weights_clone", init="same")
        self.dim_reduce = DimReduce(self.network, self.optimizer, self.weights, pca_mat, n_components, rho, gamma,
                                    alpha, sigma, _rank, _rank_size)

    def __init_adasum(self):
        """adasum algorithm init method."""
        _rank = _get_global_rank()
        _rank_size = get_group_size()
        _device_number = self.auto_boost.device_number
        self.device_number = _device_number
        group_number = _rank_size // _device_number

        self.server_rank = _rank % _device_number
        parameter_rank_number = len(self.weights) // _device_number
        self.start = [x * parameter_rank_number for x in range(_device_number)]
        self.end = [(x + 1) * parameter_rank_number for x in range(_device_number)]
        self.end[-1] = len(self.weights)

        current_weights = self.weights[self.start[self.server_rank]: self.end[self.server_rank]]
        self.grad_clone = ParameterTuple(current_weights).clone(prefix="delta_weight")
        self.adasum = AdaSum(_rank, _device_number, group_number, self.grad_clone)

        self.degree = int(self.degree // group_number)
        group_list = [list(range(x * self.degree, (x + 1) * self.degree)) for x in range(group_number)]
        current_index = _rank // _device_number
        server_group_name = "allreduce_" + str(current_index)
        create_group(server_group_name, group_list[current_index])
        self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree, group=server_group_name)


class BoostTrainOneStepWithLossScaleCell(BoostTrainOneStepCell):
    r"""
    Boost Network training with loss scaling.

    This is a training step with loss scaling. It takes a network, an optimizer and possibly a scale update
    Cell as args. The loss scale value can be updated in both host side or device side. The
    BoostTrainOneStepWithLossScaleCell will be compiled to be graph which takes `*inputs` as input data.
    The Tensor type of `scale_sense` is acting as loss scaling value. If you want to update it on host side,
    the value must be provided. If the Tensor type of `scale_sense` is not given, the loss scale update logic
    must be provide by Cell type of `scale_sense`.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Cell): Optimizer for updating the weights.
        scale_sense (Union[Tensor, Cell]): If this value is Cell type, the loss scaling update logic cell.If this value
                                          is Tensor type, Tensor with shape :math:`()` or :math:`(1,)`.

    Inputs:
        - **\*inputs** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

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
        >>> import mindspore.ops as ops
        >>> from mindspore.nn import WithLossCell
        >>> from mindspore import dtype as mstype
        >>> from mindspore import boost
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
        >>> net_with_loss = WithLossCell(net, loss)
        >>> manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
        >>> train_network = boost.BoostTrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)
        >>> input = Tensor(np.ones([out_features, in_features]), mstype.float32)
        >>> labels = Tensor(np.ones([out_features,]), mstype.float32)
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
        >>> train_network = boost.BoostTrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=scaling_sens)
        >>> output = train_network(inputs, label)
    """

    def __init__(self, network, optimizer, scale_sense):
        super(BoostTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, sens=None)
        self.base = Tensor(1, mstype.float32)
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.less_equal = P.LessEqual()
        self.allreduce = P.AllReduce()
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.gpu_target = (context.get_context("device_target") == "GPU")
        self.loss_scaling_manager = None

        if self.auto_boost.boost_config.get("loss_scale_group", False):
            self.enable_enhanced_amp = True
            if not isinstance(scale_sense, Cell) or not hasattr(scale_sense, "set_loss_scale_status"):
                raise TypeError("The scale_sense must be enhanced amp Cell, bug got {}".format(type(scale_sense)))
            self.loss_scaling_manager = scale_sense
            self.loss_scale_groups = scale_sense.loss_scale_groups
            self._init_enhanced_amp()
            self._do_keep_mix_fp32(self.network)
        else:
            self.enable_enhanced_amp = False
            if isinstance(scale_sense, Cell):
                self.loss_scaling_manager = scale_sense
                self.scale_sense = Parameter(Tensor(scale_sense.get_loss_scale(), dtype=mstype.float32),
                                             name="scale_sense")
            elif isinstance(scale_sense, Tensor):
                if scale_sense.shape == (1,) or scale_sense.shape == ():
                    self.scale_sense = Parameter(scale_sense, name='scale_sense')
                else:
                    raise ValueError("The shape of scale_sense must be (1,) or (), \
                                     but got {}".format(scale_sense.shape))
            else:
                raise TypeError("The scale_sense must be Cell or Tensor, but got {}".format(type(scale_sense)))

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)

        if self.enable_enhanced_amp:
            scaling_sens = F.fill(loss.dtype, loss.shape, 1)
            grads = self.grad(self.network, weights)(*inputs, scaling_sens)
            grads = self.grad_reducer(grads)
            cond, scaling_sens = self._enhanced_amp_process_overflow_status(grads)
        else:
            scaling_sens = self.scale_sense
            status, scaling_sens = self._start_overflow_check(loss, scaling_sens)
            scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))

            grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
            grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
            grads = self.grad_reducer(grads)

            # get the overflow buffer
            cond = self._get_overflow_status(status, grads)
            overflow = self._process_loss_scale(cond)
            # if there is no overflow, do optimize
            if not overflow:
                loss = self.__multi_update(loss, grads, scaling_sens_filled, *inputs)
        return loss, cond, scaling_sens

    def __multi_update(self, loss, grads, scaling_sens_filled, *inputs):
        """enable multi-algorithm's process"""
        if self.use_grad_accumulation:
            loss = self.gradient_accumulation_process(loss, grads, scaling_sens_filled, *inputs)
        else:
            if self.enable_dim_reduce:
                loss = F.depend(loss, self.dim_reduce(loss, grads, scaling_sens_filled, self.weights,
                                                      self.weights_clone, *inputs))
            elif self.enable_adasum:
                loss = F.depend(loss, self.adasum_process(loss, grads))
            else:
                loss = F.depend(loss, self.optimizer(grads))
        return loss

    def _get_dynamic_overflow_status(self, param):
        """
        Judge whether the current network overflows.

        Inputs:
            - **param** (Tensor) - Whether the overflow occurs or not.

        Outputs:
            bool, overflow value.
            float, update ratio.
        """
        flag_sum = self.reduce_sum(param, (0,))
        if self.reducer_flag:
            flag_reduce = self.allreduce(flag_sum)
            overflow = self.less_equal(self.base, flag_reduce)
        else:
            overflow = self.less_equal(self.base, flag_sum)
        if overflow:
            update_ratio = self.reduce_ratio
        else:
            update_ratio = self.growth_ratio
        return overflow, update_ratio

    def _enhanced_amp_process_overflow_status(self, grads):
        """
        Enhanced hybrid precision update loss scale and update weights.

        Inputs:
            - **grads** (Tuple(Tensor)) - Tuple of gradients.

        Outputs:
            bool, overflow value.
            float, loss scale value.
        """
        overflow_global_flag = Tensor(0, mstype.int32)
        layer = 0
        loss_scale_temp = ()
        for param in self.overflow_status_list:
            overflow, update_ratio = self._get_dynamic_overflow_status(param)
            if overflow:
                overflow_global_flag += 1
            new_loss_scale_value = self.loss_scaling_manager.update_loss_scale_status(layer, update_ratio)
            loss_scale_temp += (new_loss_scale_value,) * self.optimizer_loss_scale[layer]
            layer += 1
        if P.Less()(overflow_global_flag, self.base):
            grads = self.hyper_map(F.partial(_grad_scale), loss_scale_temp, grads)
            overflow_global_flag = F.depend(overflow_global_flag, self.optimizer(grads))
        return overflow_global_flag, loss_scale_temp[0]

    def _set_sense_scale(self, sens):
        """
        If the user has set the sens in the training process and wants to reassign the value, he can call
        this function again to make modification, and sens needs to be of type Tensor.

        Inputs:
            - **sens** (Tensor) - The new sense whose shape and type are the same with original `scale_sense`.
        """
        if self.scale_sense and isinstance(sens, Tensor):
            self.scale_sense.set_data(sens)
        else:
            raise TypeError("The input type must be Tensor, but got {}".format(type(sens)))

    def _start_overflow_check(self, pre_cond, compute_input):
        """
        Start floating-point overflow detection. Create and clear the overflow detection state.

        Specify the argument 'pre_cond' and 'compute_input' to make sure overflow status is cleared at the right time.
        Taking this situation as an example, we need to execute state clearing after loss calculation and then detect
        overflow in the process of gradient calculation. In this case, pre_cond should be the output of the loss
        function, and compute_input should be the input of gradients-computing function.

        Inputs:
            - **pre_cond** (Tensor) - A precondition for starting overflow detection. It determines the executing order
              of overflow state clearing and prior processions. It makes sure that the function 'start_overflow'
              clears status after finishing the process of precondition.
            - **compute_input** (object) - The input of subsequent process. Overflow detection should be performed on a
              certain computation. Set `compute_input` as the input of the computation, to ensure overflow status is
              cleared before executing the computation.

        Outputs:
            Tuple[object, object], the first value is False for GPU backend, while it is an instance of
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

    def _get_overflow_status(self, status, compute_output):
        """
        Get floating-point overflow status.

        Get overflow results after executing the target process for overflow detection.

        Inputs:
            - **status** (object) - A status instance used to detect the overflow.
            - **compute_output** - Overflow detection should be performed on a certain computation. Set `compute_output`
              as the output of the computation, to ensure overflow status is acquired before executing the
              computation.

        Outputs:
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

    def _process_loss_scale(self, overflow):
        """
        Calculate loss scale according to the overflow.

        Inputs:
            - **overflow** (bool) - Whether the overflow occurs or not.

        Outputs:
            bool, overflow value.
        """
        if self.loss_scaling_manager is not None:
            return self.loss_scaling_manager(self.scale_sense, overflow)
        return overflow

    def _init_enhanced_amp(self):
        """
        Init enhanced hybrid precision.
        """
        self.params_len = len(self.optimizer.params)
        self.parent = list(range(self.params_len))
        self.layer_rank = [0 for _ in range(self.params_len)]
        index = 0
        loss_scale_number = len(self.loss_scale_groups)
        for loss_scale_group in self.loss_scale_groups:
            for i, _ in enumerate(loss_scale_group):
                if i == 0:
                    index += 1
                    continue
                self._union(index - 1, index)
                index += 1
        parent_set = list(set(self.parent))
        self.optimizer_loss_scale = [self.parent.count(x) for x in parent_set]
        self.reduce_ratio = Tensor(1.0 / (2 ** 0.5), mstype.float32)
        self.growth_ratio = Tensor(2 ** (1.0 / 1000.0), mstype.float32)
        self.overflow_status_list = ParameterTuple(Parameter(Tensor(np.zeros(shape=[8]), mstype.float32),
                                                             name='mix_layer_status_{}'.format(x), requires_grad=False)
                                                   for x in range(loss_scale_number))
        self.loss_scaling_manager.set_loss_scale_status(loss_scale_number, self.loss_scaling_manager.get_loss_scale())

    def _get_root(self, i):
        """
        Get parent id.

        Args:
            i (int): the current parameters's id.

        Returns:
            Number, the parent id.
        """
        if self.parent[i] != self.parent[self.parent[i]]:
            self.parent[i] = self.get_root(self.parent[i])
        return self.parent[i]

    def _union(self, i, j):
        """
        Aggregate parameters of the same category.

        Args:
            i (int): the last parameters's id.
            j (int): the current parameters's id.
        """
        i_root = self._get_root(i)
        j_root = self._get_root(j)

        if self.layer_rank[i_root] == self.layer_rank[j_root]:
            self.parent[j_root] = i_root
            self.layer_rank[i_root] += 1
        elif self.layer_rank[i_root] > self.layer_rank[j_root]:
            self.parent[j_root] = i_root
        else:
            self.parent[i_root] = j_root

    def _do_keep_mix_fp32(self, network):
        """
        Keep enhanced amp cell of type float32.

        Args:
            network (Cell): The training network.
        """
        cells = network.name_cells()
        change = False
        for name in cells:
            subcell = cells[name]
            if subcell == network:
                continue
            if "GroupLossScaleManager" in subcell.cls_name:
                network._cells[name] = _OutputToFloat16(subcell.to_float(mstype.float32))  # pylint: disable=W0212
                change = True
            else:
                self._do_keep_mix_fp32(subcell)
        if isinstance(network, SequentialCell) and change:
            network.cell_list = list(network.cells())
