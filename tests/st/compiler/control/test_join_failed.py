# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
"""test join failed in control flow"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import mutable
from mindspore import jit, nn, Tensor, context
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_if_branch_have_two_return():
    """
    Feature: Test join failed in if with two return.
    Description: Abstract type AbstractTensor cannot join with AbstractScalar.
    Expectation: No exception.
    """
    # pylint: disable=no-else-return
    @jit
    def foo(x, y):
        if x < y:
            return Tensor([1, 2, 3])
        else:
            return 0

    x = Tensor(2, ms.float32)
    y = Tensor(6, ms.float32)
    with pytest.raises(TypeError) as ex:
        foo(x, y)
    assert "Cannot join the return values of different branches" in str(ex.value)
    assert "return Tensor([1, 2, 3])" in str(ex.value)
    assert "return 0" in str(ex.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_if_branch_has_one_return():
    """
    Feature: Test join failed in if with one return.
    Description: Abstract type AbstractTensor cannot join with AbstractScalar.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        if x < y:
            a = Tensor([1, 2, 3])
        else:
            print(x)
            return 0
        return a

    x = Tensor(2, ms.float32)
    y = Tensor(6, ms.float32)
    with pytest.raises(TypeError) as ex:
        foo(x, y)
    assert "Cannot join the return values of different branches" in str(ex.value)
    assert "return 0" in str(ex.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_if_branch_has_no_return():
    """
    Feature: Test join failed in if with one return.
    Description: Abstract type AbstractTensor cannot join with AbstractScalar.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        if x < y:
            a = Tensor([1, 2, 3])
        else:
            a = 0
        return a

    x = Tensor(2, ms.float32)
    y = Tensor(6, ms.float32)
    with pytest.raises(TypeError) as ex:
        foo(x, y)
    assert "Cannot join the return values of different branches" in str(ex.value)
    assert "a = Tensor([1, 2, 3])" in str(ex.value)
    assert "a = 0" in str(ex.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_while_body_has_return():
    """
    Feature: Test join failed in while.
    Description: Abstract type AbstractTensor cannot join with AbstractScalar.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        while x < 10:
            return Tensor([1, 2, 3])
        return 0

    x = Tensor([1], ms.float32)
    with pytest.raises(TypeError) as ex:
        foo(x)
    assert "Cannot join the return values of different branches" in str(ex.value)
    assert "return Tensor([1, 2, 3])" in str(ex.value)
    assert "return 0" in str(ex.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_switch_layer_join_failed():
    """
    Feature: Test join failed in switch layer.
    Description: Abstract type AbstractTuple cannot join with AbstractTensor.
    Expectation: No exception.
    """
    class JoinFailedCell1(nn.Cell):
        def construct(self, x):
            return x, Tensor(10)

    class JoinFailedCell2(nn.Cell):
        def construct(self, x):
            return x**2

    class SwitchLayerNet(nn.Cell):
        def __init__(self):
            super(SwitchLayerNet, self).__init__()
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax()
            self.join_failed1 = JoinFailedCell1()
            self.join_failed2 = JoinFailedCell2()
            self.layers = (self.relu, self.softmax, self.join_failed1, self.join_failed2)

        def construct(self, x, index):
            x = self.layers[index](x)
            return x

    context.set_context(mode=context.GRAPH_MODE)
    net = SwitchLayerNet()
    data = Tensor(np.ones((1, 1, 224, 224)), ms.float32)
    idx = Tensor(1, ms.int32)
    with pytest.raises(TypeError) as ex:
        net(data, idx)
    assert "Cannot join the return values of different branches" in str(ex.value)
    assert "return self.relu(input)" in str(ex.value)
    assert "return self.softmax(input)" in str(ex.value)
    assert "return x, Tensor(10)" in str(ex.value)
    assert "return x**2" in str(ex.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_with_fix_length_input_1():
    """
    Feature: Control flow with fix length input.
    Description: Abstract can't be converted to dynamic length
    Expectation: Catch exception, no core dump.
    """
    class MyNet(nn.Cell):
        def construct(self, x, y, z):
            if x == 1:
                return x, y, z
            return x, y

    net = MyNet()
    x = mutable(input_data=1, dynamic_len=False)
    y = mutable(input_data=Tensor(np.random.randn(18, 22).astype(np.float16)), dynamic_len=False)
    z = mutable(input_data=10, dynamic_len=False)
    try:
        net(x, y, z)
    except TypeError:
        assert "Cannot join the return values of different branches" in str(TypeError.value)
        assert "if x == 1:" in str(TypeError.value)
        assert "return x, y, z" in str(TypeError.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_control_flow_with_fix_length_input_2():
    """
    Feature: Control flow with fix length input.
    Description: Abstract type AbstractTuple cannot join with AbstractTensor.
    Expectation: Catch exception, no core dump.
    """
    class MyNet(nn.Cell):
        def construct(self, x, y):
            if x == 1:
                return x, y
            return y

    net = MyNet()
    x = mutable(input_data=1, dynamic_len=False)
    y = mutable(input_data=Tensor(np.random.randn(18, 22).astype(np.float16)), dynamic_len=False)

    try:
        net(x, y)
    except TypeError:
        assert "Cannot join the return values of different branches" in str(TypeError.value)
        assert "if x == 1:" in str(TypeError.value)
        assert "return x, y" in str(TypeError.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_type_join_error_msg():
    """
    Feature: Control flow with different type in two branches.
    Description: Abstract type AbstractScalar(Type: Int64) cannot join with AbstractScalar(Type: String).
    Expectation: Catch exception, no core dump.
    """
    @jit
    def func(input_x):
        """
        if_if_if分支都包含return
        """
        if input_x < 10:
            if input_x > 0:
                if input_x > 1:
                    return '2'
                return 1
            return [1]
        return 1, 2

    input_x = Tensor(9)
    with pytest.raises(TypeError) as exec_info:
        func(input_x)
    error_info = str(exec_info.value)
    assert "Cannot join the return values of different branches" in error_info
    assert "if input_x > 1:" in error_info


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_type_join_difftype():
    """
    Feature: Control flow with different type in two branches.
    Description: Type Float32 cannot join with Type: Int32.
    Expectation: Catch exception, no core dump.
    """
    @jit
    def func(x, y, input1, input2):
        if x > y:
            out = input1 + input1
        else:
            out = input2 + input2
        return out

    with pytest.raises(TypeError) as exec_info:
        func(Tensor(1), Tensor(0), Tensor(1, ms.float32), ms.Tensor(1, ms.int32))
    error_info = str(exec_info.value)
    assert "Cannot join the return values of different branches" in error_info


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_type_join_mstype():
    """
    Feature: Control flow with different type in two branches.
    Description: AbstractType(ms.int32) join with AbstractType(ms.int64).
    Expectation: No exception.
    """
    @ms.jit(backend='ms_backend')
    def func(x):
        return ms.int32 if x else ms.int64

    out = func(ms.mutable(False))
    assert out == ms.int64
