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
""" test SGD """
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, Parameter
import mindspore.ops as ops
import mindspore.nn as nn


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([10]).astype((np.float32))), name="bias")
        self.matmul = ops.MatMul()
        self.biasadd = ops.BiasAdd()

    def construct(self, x):
        x = self.biasadd(self.matmul(x, self.weight), self.bias)
        return x


class WeightDecaySchdule(nn.Cell):
    def __init__(self):
        super(WeightDecaySchdule, self).__init__()
        self.weight_decay_list = Tensor([0.001, 0.001, 0.1], ms.float32)

    def construct(self, global_step):
        return self.weight_decay_list[global_step]


def test_sgd_dynamic_weightdecay():
    """
    Feature: Test SGD optimizer.
    Description: Test if error is raised when weight decay is dynamic.
    Expectation: ValueError is raised.
    """
    net = Net()
    params = net.trainable_params()
    group_params = [{'params': [params[0]], 'weight_decay': WeightDecaySchdule()}, {'params': [params[1]]}]

    weight_decay_error = "For 'SGD', dynamic weight decay is currently not supported, the argument 'weight_decay' " \
                         "or 'weight_decay' set in grouped 'params' must be float or int type."
    with pytest.raises(TypeError, match=weight_decay_error):
        nn.SGD(params, learning_rate=0.1, weight_decay=WeightDecaySchdule())

    with pytest.raises(TypeError, match=weight_decay_error):
        nn.SGD(group_params, learning_rate=0.1)
