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
import numpy as np
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu_grad = G.ReluGrad()

    def construct(self, y_backprop, x):
        return self.relu_grad(y_backprop, x)


def get_output(y_backprop, x, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net()
    output = net(y_backprop, x)
    return output


def run_relu_grad(shape1, shape2, dtype):
    x = Tensor(np.random.normal(0, 10, shape1).astype(dtype))
    y_backprop = Tensor(np.random.normal(0, 10, shape2).astype(dtype))
    expect = get_output(y_backprop, x, False)
    output = get_output(y_backprop, x, True)

    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()

    assert np.allclose(expect_np, output_np, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_relu_grad_gpu():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_relu_grad((4, 3), (4, 3), np.int32)
    run_relu_grad((12, 1), (12, 1), np.float16)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_relu_grad_ascend():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_relu_grad((4, 3), (4, 3), np.int32)
    run_relu_grad((12, 1), (12, 1), np.float16)
