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
""" test_lr_schedule """
import numpy as np

from mindspore import Parameter, ParameterTuple, Tensor
from mindspore.nn import Cell
from mindspore.nn.optim import Optimizer
from mindspore.ops.operations import BiasAdd, MatMul
import mindspore.ops.composite as C


grad_by_list = C.GradOperation(get_by_list=True)


class Net(Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10])), name="weight")
        self.bias = Parameter(Tensor(np.ones([10])), name="bias")
        self.matmul = MatMul()
        self.biasAdd = BiasAdd()

    def construct(self, x):
        x = self.biasAdd(self.matmul(x, self.weight), self.bias)
        return x


class _TrainOneStepCell(Cell):
    """ _TrainOneStepCell definition """

    def __init__(self, network, optimizer):
        """
        Append an optimizer to the training network after that the construct
        function can be called to create the backward graph.
        Arguments:
            network: The training network.
                Note that loss function should have been added.
            optimizer: optimizer for updating the weights
        """
        super(_TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.get_parameters())

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an optimizer'.format(
                type(optimizer).__name__))

        self.has_lr_schedule = False
        self.optimizer = optimizer

    def construct(self, data, label, *args):
        weights = self.weights
        grads = grad_by_list(self.network, weights)(data, label)
        if self.lr_schedule:
            self.schedule.update_lr(*args)
        return self.optimizer(grads)
