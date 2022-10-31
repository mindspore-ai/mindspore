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
from mindspore import ops
from mindspore import Tensor, export, load, Parameter, dtype, context


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
