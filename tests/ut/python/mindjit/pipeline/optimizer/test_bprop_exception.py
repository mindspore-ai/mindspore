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

import pytest
import mindspore as ms
from mindspore import ops
from mindspore import context


def test_self_define_cell_output_not_tuple():
    """
    Feature: Custom cell bprop.
    Description: If bprop output is not tuple, a exception will be expected raised to tell user it should be a tuple.
    Expectation: Expected exception raised.
    """
    context.set_context(mode=context.GRAPH_MODE)

    class SelfDefineCell(ms.nn.Cell):
        def construct(self, x):
            return x + 1, x + 2

        def bprop(self, x, out, dout):
            return out[1]

    class ForwardNet(ms.nn.Cell):
        def __init__(self):
            super(ForwardNet, self).__init__()
            self.self_defined_cell = SelfDefineCell()

        def construct(self, x):
            # keep out1 not used in fprop.
            out0, _ = self.self_defined_cell(x)
            return out0

    class TestNet(ms.nn.Cell):
        def __init__(self):
            super(TestNet, self).__init__()
            self.forward_net = ForwardNet()
            self.grad_op = ops.GradOperation(get_all=True)

        def construct(self, x):
            grad_out = self.grad_op(self.forward_net)(x)
            return grad_out

    with pytest.raises(TypeError) as info:
        net = TestNet()
        x_input = ms.Tensor([1])
        out = net(x_input)
        print("out:", out)
    assert "For bprop function, output should be a tuple" in str(info.value)
