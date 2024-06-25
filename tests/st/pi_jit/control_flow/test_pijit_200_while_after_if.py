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
""" test PIJit control flow."""
import pytest
import numpy as np
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_while_after_if_tensor():
    """
    Feature: PIJit
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit(mode="PIJit")
    def control_flow_while_after_if():
        x = Tensor([1])
        y = Tensor([2])
        if x > y:
            y = y * x
        z = Tensor([7]) + y
        while y > x and x < z:
            y -= x
        z = z + y
        return y + z

    context.set_context(mode=context.PYNATIVE_MODE)
    res = control_flow_while_after_if()
    assert res == 11


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_while_after_if_tensor_2():
    """
    Feature: PIJit
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit(mode="PIJit")
    def control_flow_while_after_if():
        x = Tensor([1])
        y = Tensor([2])
        if x > y:
            y = y * x
        else:
            x = x * y
        z = Tensor(7) - y
        while x < z:
            y += x
            z = z - y
        return x, y, z

    context.set_context(mode=context.PYNATIVE_MODE)
    res_x, res_y, res_z = control_flow_while_after_if()
    assert res_x == 2
    assert res_y == 4
    assert res_z == 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_while_after_if_numpy():
    """
    Feature: PIJit
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit(mode="PIJit")
    def control_flow_while_after_if():
        x = np.array([3, 2])
        y = Tensor(np.array([3, 2]))
        if x[0] > x[1]:
            x += 3
        else:
            x -= 4
        while (y >= 0).all():
            y -= Tensor(x[0])
        return y

    context.set_context(mode=context.PYNATIVE_MODE)
    res = control_flow_while_after_if()
    assert (res.asnumpy() == [-3, -4]).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_while_after_if_numpy_2():
    """
    Feature: PIJit
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit(mode="PIJit")
    def control_flow_while_after_if():
        x = np.array([3, 2])
        y = [1, 2, 3, 4]
        if x[0] > x[1]:
            x += 3
        else:
            x -= 4
        while len(y) <= 5:
            y.append(x[1])
        return Tensor(y)

    context.set_context(mode=context.PYNATIVE_MODE)
    res = control_flow_while_after_if()
    assert (res.asnumpy() == [1, 2, 3, 4, 5, 5]).all()
