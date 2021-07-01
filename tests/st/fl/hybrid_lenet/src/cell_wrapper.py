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

from mindspore.context import ParallelMode
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.nn.cell import Cell
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

class TrainOneStepCellWithServerCommunicator(Cell):
    r"""
    Network training package class.

    Wraps the network with an optimizer and communicator operators.
    The resulting Cell is trained with input '\*inputs'.
    The backward graph will be created in the construct function to update the parameter. Different
    parallel modes are available for training.

    This cell is used for hybrid training mode for now.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Inputs:
        - **(\*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a scalar Tensor with shape :math:`()`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> #1) Using the WithLossCell existing provide
        >>> loss_net = nn.WithLossCell(net, loss_fn)
        >>> train_net = nn.TrainOneStepCell(loss_net, optim)
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
        >>> train_net = nn.TrainOneStepCell(loss_net, optim)
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCellWithServerCommunicator, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)

        self.hyper_map = C.HyperMap()

        self.pull_weight_by_key = P.PullWeight()
        self.push_weight_by_key = P.PushWeight()

        self.pull_weights,\
        self.pull_weight_names,\
        self.pull_weight_indices = self._pull_weight_inputs(self.network.parameters_and_names())

        self.push_weights,\
        self.push_weight_names,\
        self.push_weight_indices = self._push_weight_inputs(self.network.parameters_and_names())

    def _pull_from_server(self, weights, names, indices):
        result = self.hyper_map(F.partial(self.pull_weight_by_key), weights, names, indices)
        return result

    def _push_to_server(self, weights, names, indices):
        result = self.hyper_map(F.partial(self.push_weight_by_key), weights, names, indices)
        return result

    @staticmethod
    def _pull_weight_inputs(weights):
        """pull weight by key inputs."""
        filtered_weights = []
        weight_names = []
        weight_indices = []
        index = 0
        for weight in weights:
            if weight[1].pull_weight_from_server:
                filtered_weights.append(weight[1])
                weight_names.append(weight[1].name)
                weight_indices.append(index)
            index += 1

        return ParameterTuple(filtered_weights), tuple(weight_names), tuple(weight_indices)

    @staticmethod
    def _push_weight_inputs(weights):
        """push weight by key inputs."""
        filtered_weights = []
        weight_names = []
        weight_indices = []
        index = 0
        for weight in weights:
            if weight[1].push_weight_to_server:
                filtered_weights.append(weight[1])
                weight_names.append(weight[1].name)
                weight_indices.append(index)
            index += 1

        return ParameterTuple(filtered_weights), tuple(weight_names), tuple(weight_indices)

    def construct(self, *inputs):
        weights = self.weights
        res = self._pull_from_server(self.pull_weights,
                                     self.pull_weight_names, self.pull_weight_indices)
        inputs = F.depend(inputs, res)
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        grads = self.grad_reducer(grads)

        loss = F.depend(loss, self.optimizer(grads))
        push_weights = F.depend(self.push_weights, loss)
        loss = F.depend(loss, self._push_to_server(push_weights,
                                                   self.push_weight_names, self.push_weight_indices))
        return loss
