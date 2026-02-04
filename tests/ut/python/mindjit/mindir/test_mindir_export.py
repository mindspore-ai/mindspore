# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore.nn import Cell, GraphCell
from mindspore import ops, nn, jit
from mindspore import Tensor, export, load, Parameter, dtype, context
from mindspore.common.initializer import initializer
import mindspore as ms


def test_export_control_flow():
    """
    Feature: Test MindIR Export model
    Description: test mindir export when parameter is not use
    Expectation: No exception.
    """

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = Parameter(Tensor([-2], dtype.float32), name="weight")
            self.b = Parameter(Tensor([-5], dtype.float32), name="bias")

        def construct(self, x, y):
            if len(x.shape) == 1:
                return y
            while y >= x:
                if self.b <= x:
                    return y
                elif self.w < x:
                    return x
                x += y

            return x + y

    context.set_context(mode=context.GRAPH_MODE)
    x = np.array([3], np.float32)
    y = np.array([0], np.float32)
    net = Net()
    export(net, Tensor(x), Tensor(y), file_name="ctrl", file_format='MINDIR')
    graph = load('ctrl.mindir')
    g_net = GraphCell(graph)
    export_out = g_net(Tensor(x), Tensor(y))
    correct_out = net(Tensor(x), Tensor(y))
    assert np.allclose(export_out.asnumpy(), correct_out.asnumpy())


def test_export_graph_cell_in_jit():
    """
    Feature: Test MindIR Export model
    Description: test mindir export when there is a GraphCell in jit function.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
    x = Tensor(np.ones([1, 120, 1024, 640]), ms.float32)
    export(net, x, file_name="test_conv2d", file_format='MINDIR')
    graph = load('test_conv2d.mindir')
    g_net = GraphCell(graph)

    @jit
    def g_net_jit_func(inputs):
        return g_net(inputs)

    g_net_jit_func(x)


def test_mindir_export_none():
    """
    Feature: Test MindIR Export model
    Description: test mindir export type none
    Expectation: No exception.
    """
    class TestCell(Cell):
        def __init__(self):
            super(TestCell, self).__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            return self.relu(x), None, None

    input_tensor = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    net = TestCell()
    export(net, input_tensor, file_name="none_net", file_format='MINDIR')
    graph = load("none_net.mindir")
    assert graph is not None


def test_mindir_export_parameter_as_tensor():
    """
    Feature: Test MindIR Export model
    Description: test mindir export parameter as construct input
    Expectation: No exception.
    """
    input_np_x = np.random.randn(3).astype(np.float32)
    input_np_x_param = Parameter(input_np_x)

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = nn.ReLU()
            self.x = Parameter(Tensor(input_np_x))

        def construct(self, x):
            x = x + x
            x = x * self.x
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    out_net = net(input_np_x_param)
    export(net, input_np_x_param, file_name="test", file_format="MINDIR")
    graph = load("test.mindir")
    net_mindir = nn.GraphCell(graph)
    result_mindir = net_mindir(input_np_x_param)
    assert np.allclose(result_mindir.asnumpy(), out_net.asnumpy(), 0.0001, 0.001, equal_nan=True)

def test_mindir_export_bfloat16():
    """
    Feature: Test MindIR Export model
    Description: test mindir export bfloat16 types
    Expectation: No exception.
    """
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.val = Parameter(initializer("normal", [2, 2], dtype.bfloat16), 'val')
            self.add = ops.Add()

        def construct(self, x):
            x = self.add(x, self.val)
            return x

    context.set_context(mode=context.GRAPH_MODE)
    input_x = Tensor(np.ones((2, 2)), dtype=dtype.bfloat16)
    net = Net()
    export(net, input_x, file_name="test", file_format="MINDIR")
    graph = load("test.mindir")
    net_mindir = nn.GraphCell(graph)
    net_mindir(input_x)
