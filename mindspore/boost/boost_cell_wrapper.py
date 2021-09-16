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
"""Boost Mode Cell Wrapper."""
from mindspore.nn.wrap import TrainOneStepCell
import mindspore.context as context
from mindspore.context import ParallelMode, get_auto_parallel_context
from mindspore.parallel._utils import _get_global_rank, _get_device_num, _get_gradients_mean
from mindspore.communication.management import get_group_size, create_group
from mindspore.nn.cell import Cell
from mindspore.common import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from .grad_freeze import FreezeOpt, freeze_cell
from .adasum import AdaSum
from .grad_accumulation import gradient_accumulation_op, gradient_clear_op


__all__ = ["BoostTrainOneStepCell", "BoostTrainOneStepWithLossScaleCell"]


_get_delta_weight = C.MultitypeFuncGraph("_get_delta_weight")


@_get_delta_weight.register("Tensor", "Tensor")
def _get_delta_weight_process(new_parameter, old_parameter):
    delta_w = old_parameter - new_parameter
    return delta_w


_save_weight = C.MultitypeFuncGraph("_save_weight")


@_save_weight.register("Tensor", "Tensor")
def _save_weight_process(new_parameter, old_parameter):
    return P.Assign()(new_parameter, old_parameter)


class BoostTrainOneStepCell(TrainOneStepCell):
    r"""
    Boost Network training package class.

    Wraps the network with an optimizer. The resulting Cell is trained with input '\*inputs'.
    The backward graph will be created in the construct function to update the parameter. Different
    parallel modes are available for training.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Union[Cell]): Optimizer for updating the weights.
        sens (numbers.Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Inputs:
        - **(\*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a tensor means the loss value, the shape of which is usually :math:`()`.

    Raises:
        TypeError: If `sens` is not a number.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
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

        self.use_grad_accumulation = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.STAND_ALONE)
        if self.use_grad_accumulation:
            self.max_accumulation_step = get_auto_parallel_context("grad_accumulation_step")
            if self.max_accumulation_step <= 1:
                self.max_accumulation_step = 1
                self.use_grad_accumulation = False
        self.accumulation_step = Parameter(Tensor(0, dtype=mstype.int32), name="accumulation_step")
        if self.use_grad_accumulation:
            self.grad_accumulation = self.weights.clone(prefix="grad_accumulation", init='zeros')

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

        self.enable_adasum = self.check_adasum_enable(optimizer, self.reducer_flag)
        self.sync_tensor = Parameter(Tensor(0, dtype=mstype.int32))
        if self.enable_adasum:
            _rank = _get_global_rank()
            _rank_size = get_group_size()
            _device_number = 8
            self.device_number = _device_number
            group_number = _rank_size // _device_number

            self.server_rank = _rank % _device_number
            parameter_rank_number = len(self.weights) // _device_number
            self.start = [x * parameter_rank_number for x in range(_device_number)]
            self.end = [(x + 1) * parameter_rank_number for x in range(_device_number)]
            self.end[-1] = len(self.weights)

            current_weights = self.weights[self.start[self.server_rank] : self.end[self.server_rank]]
            self.grad_clone = ParameterTuple(current_weights).clone(prefix="delta_weight")
            self.adasum = AdaSum(_rank, _device_number, group_number, self.grad_clone)

            self.degree = int(self.degree / group_number)
            group_list = [list(range(x * self.degree, (x + 1) * self.degree)) for x in range(group_number)]
            current_index = _rank // _device_number
            server_group_name = "allreduce_" + str(current_index)
            create_group(server_group_name, group_list[current_index])
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree, group=server_group_name)


    def construct(self, *inputs):
        if self.freeze:
            loss = self.gradient_freeze_process(*inputs)
        else:
            loss = self.network(*inputs)
            sens = F.fill(loss.dtype, loss.shape, self.sens)
            grads = self.grad(self.network, self.weights)(*inputs, sens)
            grads = self.grad_reducer(grads)
            if self.use_grad_accumulation:
                loss = self.gradient_accumulation_process(loss, grads)
            else:
                if self.enable_adasum:
                    loss = F.depend(loss, self.adasum_process(loss, grads))
                else:
                    loss = F.depend(loss, self.optimizer(grads))
        return loss

    def gradient_freeze_process(self, *inputs):
        """gradient freeze algorithm process."""
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

    def gradient_accumulation_process(self, loss, grads):
        """gradient accumulation algorithm process."""
        loss = F.depend(loss, self.hyper_map(F.partial(gradient_accumulation_op, self.max_accumulation_step),
                                             self.grad_accumulation, grads))
        self.accumulation_step += 1

        if self.accumulation_step >= self.max_accumulation_step:
            if self.enable_adasum:
                loss = F.depend(loss, self.adasum_process(loss, self.grad_accumulation))
            else:
                loss = F.depend(loss, self.optimizer(self.grad_accumulation))
            self.accumulation_step = 0

        if self.accumulation_step == 0:
            loss = F.depend(loss, self.hyper_map(F.partial(gradient_clear_op), self.grad_accumulation))

        return loss

    def adasum_process(self, loss, grads):
        """adasum algorithm process."""
        loss = F.depend(loss, self.optimizer(grads))
        rank_weights = self.weights[self.start[self.server_rank] : self.end[self.server_rank]]
        grad_clone = F.depend(self.grad_clone, loss)
        delta_w = self.hyper_map(F.partial(_get_delta_weight), rank_weights, grad_clone)
        adasum_res = self.adasum(delta_w, rank_weights, grad_clone)
        sync_tensor = F.depend(self.sync_tensor, adasum_res)
        sync_flag = self.adasum.sync_barrier(sync_tensor)
        for i in range(self.device_number):
            weight_tuple = self.weights[self.start[i] : self.end[i]]
            node_rank = F.depend(weight_tuple, sync_flag)
            update_weights = self.adasum.broadcast_list[i](node_rank)
            if i == self.server_rank:
                self.hyper_map(F.partial(_save_weight), self.grad_clone, update_weights)
            else:
                self.hyper_map(F.partial(_save_weight), weight_tuple, update_weights)
        return loss

    def check_adasum_enable(self, optimizer, reducer_flag):
        """check adasum enable."""
        if not getattr(optimizer, "adasum", None) or not reducer_flag:
            return False
        _rank_size = get_group_size()
        _device_number = 8
        group_number = _rank_size // _device_number
        is_enable = bool(group_number > 1 and group_number & (group_number - 1) == 0)
        return is_enable


class BoostTrainOneStepWithLossScaleCell(BoostTrainOneStepCell):
    r"""
    Boost Network training with loss scaling.

    This is a training step with loss scaling. It takes a network, an optimizer and possibly a scale update
    Cell as args. The loss scale value can be updated in both host side or device side. The
    BoostTrainOneStepWithLossScaleCell will be compiled to be graph which takes `*inputs` as input data.
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
        >>> import mindspore.ops as ops
        >>> from mindspore.nn import WithLossCell
        >>> from mindspore.common import dtype as mstype
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

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.use_grad_accumulation:
                loss = self.gradient_accumulation_process(loss, grads)
            else:
                if self.enable_adasum:
                    loss = F.depend(loss, self.adasum_process(loss, self.grad_accumulation))
                else:
                    loss = F.depend(loss, self.optimizer(grads))
        return loss, cond, scaling_sens
