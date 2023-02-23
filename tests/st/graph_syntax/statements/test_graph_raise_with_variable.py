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
""" test graph raise """
import pytest
import mindspore.nn as nn
from mindspore import Tensor, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_with_variable_1():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            raise ValueError(f"The input can not be {x}.")

    with pytest.raises(ValueError) as raise_info_9:
        net = RaiseNet()
        x = Tensor(11)
        res = net(x)
        print("res:", res)
    assert "The input can not be 11." in str(raise_info_9.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_with_variable_2():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise(string % var).
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            raise ValueError(f"The input can not be %s." % x)

    with pytest.raises(ValueError) as raise_info_10:
        net = RaiseNet()
        res = net(Tensor(11))
        print("res:", res)
    assert "The input can not be 11." in str(raise_info_10.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_with_variable_3():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            raise ValueError(f"The input can not be ", x, ".")

    with pytest.raises(ValueError) as raise_info_11:
        net = RaiseNet()
        res = net(Tensor(11))
        print("res:", res)
    assert "('The input can not be ', Tensor(shape=[], dtype=Int64, value= 11), '.')" in str(
        raise_info_11.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_with_variable_list():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = [Tensor(1), Tensor(2), Tensor(3), Tensor(4)]
            raise ValueError(x)

    with pytest.raises(ValueError) as raise_info_list:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "[Tensor(shape=[1], dtype=Int64, value= [1])," in str(
        raise_info_list.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_with_variable_tuple_1():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = (Tensor(1), Tensor(2), Tensor(3), Tensor(4))
            raise ValueError(x)

    with pytest.raises(ValueError) as raise_info_tuple:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "(Tensor(shape=[1], dtype=Int64, value= [1])," in str(
        raise_info_tuple.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_with_variable_tuple_2():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = (Tensor(1), Tensor(2), Tensor(3), Tensor(4))
            raise ValueError("test_string_tuple", x)

    with pytest.raises(ValueError) as raise_info_string_tuple:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "('test_string_tuple', (Tensor(shape=[1], dtype=Int64, value= [1])" in str(
        raise_info_string_tuple.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_with_variable_joinedstr_tensor():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            raise RuntimeError(f"The input should not be {x}.")

    with pytest.raises(RuntimeError) as raise_info_joinedstr_tensor:
        net = RaiseNet()
        x = Tensor(1)
        res = net(x)
        print("res:", res)
    assert "The input should not be 1" in str(
        raise_info_joinedstr_tensor.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_with_variable_dic():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = Tensor(1)
            y = Tensor(2)
            z = {"x": x, "y": y}
            raise ValueError(z)

    with pytest.raises(RuntimeError) as raise_info_list:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "Dictionary type is currently not supporting" in str(
        raise_info_list.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_with_variable_control_flow1():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x, y):
            if x == y:
                raise RuntimeError(f"The input should not be {x}.")

    with pytest.raises(RuntimeError) as raise_info_joinedstr_tensor:
        net = RaiseNet()
        x = Tensor(1)
        y = Tensor(1)
        res = net(x, y)
        print("res:", res)
    assert "The input should not be 1" in str(
        raise_info_joinedstr_tensor.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_with_variable_control_flow2():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x, y):
            if x == y:
                raise RuntimeError(f"The input should not be {x}.")
            return x

    with pytest.raises(RuntimeError) as raise_info_joinedstr_tensor:
        net = RaiseNet()
        x = Tensor(1)
        y = Tensor(1)
        res = net(x, y)
        print("res:", res)
    assert "The input should not be 1" in str(
        raise_info_joinedstr_tensor.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_with_variable_control_flow3():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x, y, z):
            if x == y:
                raise RuntimeError(f"The input should not be {x}.")
            return z

    with pytest.raises(RuntimeError) as raise_info_joinedstr_tensor:
        net = RaiseNet()
        x = Tensor(1)
        y = Tensor(1)
        z = (x, y)
        res = net(x, y, z)
        print("res:", res)
    assert "The input should not be 1" in str(
        raise_info_joinedstr_tensor.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_with_variable_control_flow4():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x, y, z):
            if x == y:
                raise RuntimeError(f"The input should not be {x}.")
            return z

    with pytest.raises(RuntimeError) as raise_info_joinedstr_tensor:
        net = RaiseNet()
        x = Tensor(1)
        y = Tensor(1)
        z = [x, y]
        res = net(x, y, z)
        print("res:", res)
    assert "The input should not be 1" in str(
        raise_info_joinedstr_tensor.value)
