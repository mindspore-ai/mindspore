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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class TrainOneStepCellWithServerCommunicator(nn.Cell):
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

    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCellWithServerCommunicator, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

        self.hyper_map = ops.HyperMap()

        self.pull_weight_by_key = ops.PullWeight()
        self.push_weight_by_key = ops.PushWeight()

        self.pull_weights, \
        self.pull_weight_names, \
        self.pull_weight_indices = self._pull_weight_inputs(self.network.parameters_and_names())

        self.push_weights, \
        self.push_weight_names, \
        self.push_weight_indices = self._push_weight_inputs(self.network.parameters_and_names())

    def _pull_from_server(self, weights, names, indices):
        result = self.hyper_map(ops.partial(self.pull_weight_by_key), weights, names, indices)
        return result

    def _push_to_server(self, weights, names, indices):
        result = self.hyper_map(ops.partial(self.push_weight_by_key), weights, names, indices)
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

        return ms.ParameterTuple(filtered_weights), tuple(weight_names), tuple(weight_indices)

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

        return ms.ParameterTuple(filtered_weights), tuple(weight_names), tuple(weight_indices)

    def construct(self, *inputs):
        weights = self.weights
        res = self._pull_from_server(self.pull_weights,
                                     self.pull_weight_names, self.pull_weight_indices)
        inputs = ops.depend(inputs, res)
        loss = self.network(*inputs)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)

        loss = ops.depend(loss, self.optimizer(grads))
        push_weights = ops.depend(self.push_weights, loss)
        loss = ops.depend(loss, self._push_to_server(push_weights,
                                                     self.push_weight_names, self.push_weight_indices))
        return loss
