# Copyright 2024 Huawei Technologies Co., Ltd
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
""" Test tensor hook"""

import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, ops, context, Tensor, Parameter
from tests.mark_utils import arg_mark


def hook_double(grad):
    return grad * 2

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight1")

    def construct(self, x):
        self.weight1.register_hook(hook_double)
        out = x * self.weight1
        return out

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_hook_inside_graph():
    """
    Feature: Parameter.register_hook(hook_fn) inside graph
    Description: Test register parameter hook inside graph
    Expectation: Raise ValueError with error message: "Register hook for Parameter inside graph is not supported.".
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x = Tensor(np.array([4.0, 5.0, 6.0]), ms.float32)
    net = Net()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    with pytest.raises(ValueError) as e:
        grad_net(input_x)
    assert "Register hook for Parameter inside graph is not supported." in str(e.value)
