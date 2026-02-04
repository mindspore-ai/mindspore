# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test nn.module in PIJIT"""

import torch
import mindspore as ms
from mindspore import jit, ops
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_executed_by_graph_mode, assert_equal


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nn_module():
    """
    Feature: PIJIT
    Description: Test torch.nn.Module
    Expectation: success
    """

    class Net(torch.nn.Module):
        def forward(self, x):
            return x + 1

    net = Net()

    @ms.jit(capture_mode="bytecode", fullgraph=True)
    def func(x):
        return net(x)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    assert func(1) == 2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nn_module_call_super_method():
    """
    Feature: pijit + msadapter
    Description: Test torch.nn.Module call super method.
    Expectation: No graph break.
    """

    class Linear(torch.nn.Module):
        def forward(self, w, x, b):
            return w @ x + b

    class LinearReLU(Linear):
        def forward(self, w, x, b):
            # In order to call `super()`, Python automatically adds a free variable named `__class__`.
            o = super().forward(w, x, b)
            return ops.relu(o)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = LinearReLU()

        def forward(self, w, x, b):
            return self.linear(w, x, b)

    model = Model()
    x = ops.randn(2, 4)
    y = ops.randn(4, 2)
    b = ops.randn(2)

    o1 = model(x, y, b)

    model.forward = jit(model.forward, capture_mode="bytecode", fullgraph=True)
    o2 = model(x, y, b)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(model.forward)
