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
from mindspore import ms_function
from mindspore import Tensor, export, load, Parameter, dtype, context
from mindspore.nn import GraphCell


def test_controller():
    """
    Feature: Test MindIR Export msfunction without using decorator.
    Description: test msfunction export as mindir.
    Expectation: No exception, assert True.
    """
    def controller(x, y):
        w = Parameter(Tensor([-2], dtype.float32), name="weight")
        b = Parameter(Tensor([-5], dtype.float32), name="bias")
        if len(x.shape) == 1:
            return y
        while y >= x:
            if b <= x:
                return y
            if w < x:
                return x
            x += y
        return x + y

    context.set_context(mode=context.GRAPH_MODE)
    input1 = Tensor(np.array([3], np.float32))
    input2 = Tensor(np.array([0], np.float32))
    controller_graph = ms_function(fn=controller)
    expected_out = controller_graph(input1, input2)
    export(controller_graph, input1, input2, file_name="control.mindir", file_format="MINDIR")
    c_graph = load("control.mindir")
    c_net = GraphCell(c_graph)
    actual_out = c_net(input1, input2)
    assert np.allclose(actual_out.asnumpy(), expected_out.asnumpy())
