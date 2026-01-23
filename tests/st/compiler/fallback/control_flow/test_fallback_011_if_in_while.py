# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
""" test graph fallback control flow if in while scenario"""
import numpy as np
from mindspore import Tensor, jit, context, nn, ops
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_if_in_while_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if_in_while():
        x = Tensor(1)
        y = Tensor(0)
        while x < Tensor(5):
            if x % 2 == Tensor(0):
                y += Tensor(1)
            x += Tensor(1)
        return x + y

    res = control_flow_if_in_while()
    assert res == 7


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_if_in_while_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if_in_while():
        x = Tensor(1)
        while x < Tensor(5):
            if x % 3 == Tensor(0):
                break
            x += Tensor(1)
        return x

    res = control_flow_if_in_while()
    assert res == 3

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_if_in_while_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if_in_while():
        x = Tensor(1)
        y = Tensor(0)
        while x < Tensor(5):
            if x % 3 == Tensor(0):
                x += Tensor(1)
                y += Tensor(1)
                continue
            x += Tensor(1)
        return x + y

    res = control_flow_if_in_while()
    assert res == 6

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_if_in_while_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    # pylint: disable=no-else-continue
    @jit(backend="ms_backend")
    def control_flow_if_in_while():
        x = Tensor(1)
        y = Tensor(0)
        while x < Tensor(10) and x + y < Tensor(20):
            if x % 3 == Tensor(0):
                x += Tensor(1)
                y += Tensor(1)
                continue
            elif y % 2 == Tensor(0):
                x += Tensor(1)
            elif (x+y) % 5 == Tensor(0):
                break
            else:
                x += Tensor(1)
        return x + y

    res = control_flow_if_in_while()
    assert res == 5

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_if_in_while_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if_in_while():
        x = np.array([1, 2])
        y = np.array([3, 2])
        index = Tensor(1)
        while index < Tensor(3):
            index += 1
            if (y > x).all():
                y += x
        return Tensor(y)
    res = control_flow_if_in_while()
    assert (res.asnumpy() == [3, 2]).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.00)
def test_parser_fallback_modify_args_to_control_branch():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    class ModifyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p = 4

        def construct(self, x):
            count = 0
            while self.p > 0:
                if x < self.p:
                    self.p = self.p - 1
                else:
                    x = -x
                    self.p += 1
                count += 1
                self.p = self.p - 1
            self.p = count
            return self.p

    class GradNet(nn.Cell):
        def __init__(self, net, grad_position=0):
            super().__init__()
            self.grad = ops.grad
            self.grad_net = self.grad(net, grad_position=grad_position)

        def construct(self,  *x):
            return self.grad_net(*x)

    out1 = ModifyNet()(Tensor([2]))
    out2 = ModifyNet()(Tensor([2]))
    assert out1 == 3
    assert out2 == 3

    ms_grad = GradNet(ModifyNet())(Tensor([2]))
    assert ms_grad == 0
