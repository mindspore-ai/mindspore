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
def test_raise_1():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x == 1:
                raise ValueError()
            return x

    with pytest.raises(ValueError, match=""):
        net = RaiseNet()
        res = net(1)
        print("res:", res)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_2():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x == 1:
                raise ValueError(1)
            return x

    with pytest.raises(ValueError, match="1"):
        net = RaiseNet()
        res = net(1)
        print("res:", res)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_3():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x == 1:
                raise ValueError(f"The input should not be 1.")
            return x

    with pytest.raises(ValueError, match="The input should not be 1."):
        net = RaiseNet()
        res = net(1)
        print("res:", res)


@pytest.mark.skip(reason="This case will be fixed by another pr in the future.")
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_6():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class NetWithRaise(nn.Cell):
        def subfunc(self):
            raise ValueError(f"exception in subfunc.")

        def construct(self, x):
            y = Tensor(0)
            if x > 0:
                y = Tensor(1)
            elif x == 1:
                y = Tensor(2)
            else:
                self.subfunc()
            return y

    with pytest.raises(ValueError, match="exception in subfunc."):
        net = NetWithRaise()
        x = -1
        res = net(x)
        print("res:", res)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_7():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = [1, 3, 5, 7, 9]
            raise ValueError("Not expected value, x is {}".format(x))

    with pytest.raises(ValueError) as raise_info_7:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "Not expected value, x is [1, 3, 5, 7, 9]" in str(
        raise_info_7.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_8():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def __init__(self):
            super(RaiseNet, self).__init__()
            self.x = [1, 3, 5, 7]

        def construct(self):
            if self.x == [1, 3, 5, 7, 9]:
                return 5
            if self.x == [1, 3, 5]:
                return 3
            raise ValueError("Not expected value, x is {}".format(self.x))

    with pytest.raises(ValueError) as raise_info_8:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "Not expected value, x is [1, 3, 5, 7]" in str(raise_info_8.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_9():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = 11
            raise ValueError(f"The input can not be {x}.")

    with pytest.raises(ValueError) as raise_info_9:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "The input can not be 11." in str(raise_info_9.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_10():
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
        res = net(11)
        print("res:", res)
    assert "The input can not be 11." in str(raise_info_10.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_11():
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
        res = net(11)
        print("res:", res)
    assert "('The input can not be ', 11, '.')" in str(raise_info_11.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_12():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise(string % var).
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = 1
            if x == 1:
                raise ValueError(
                    "The var name is %s, it can not be %d." % ("x", x))
            return x

    with pytest.raises(ValueError) as raise_info_12:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "The var name is x, it can not be 1." in str(raise_info_12.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_13():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise(string % var).
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = Tensor(1)
            if x == 1:
                raise ValueError("The input should not be Tensor(1).")
            return x

    with pytest.raises(ValueError) as raise_info_13:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "The input should not be Tensor(1)." in str(raise_info_13.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_15():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = 5
            y = [1, 2, 3, 4]
            if x > len(y):
                raise IndexError("The list index out of range.")
            return y[x]

    with pytest.raises(IndexError) as raise_info_15:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "The list index out of range." in str(raise_info_15.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_16():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = [1, 2, 3, 4]
            if isinstance(x, list):
                raise TypeError("The input should not be list.")
            return x

    with pytest.raises(TypeError) as raise_info_16:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "The input should not be list." in str(raise_info_16.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_17():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            name = "name_a"
            if name == "name_a":
                raise NameError("The name should not be name_a.")
            return self.param_a

    with pytest.raises(NameError) as raise_info_17:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "The name should not be name_a." in str(raise_info_17.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_18():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def __init__(self):
            super(RaiseNet, self).__init__()
            self.input = Tensor(1)

        def construct(self):
            if self.input == 1:
                raise AssertionError("The input should not be 1.")
            return self.param_a

    with pytest.raises(AssertionError) as raise_info_18:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "The input should not be 1." in str(raise_info_18.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_19():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x == 1:
                raise NotSupportError(f"The input should not be 1.")
            return x

    with pytest.raises(RuntimeError) as raise_info_19:
        net = RaiseNet()
        res = net(1)
        print("res:", res)
    assert "Unsupported exception type: NotSupportError" in str(
        raise_info_19.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_20():
    """
    Feature: graph raise.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self, x):
            if x == 1:
                raise RuntimeWarning(f"The input should not be 1.")
            return x

    with pytest.raises(RuntimeWarning, match="The input should not be 1."):
        net = RaiseNet()
        res = net(1)
        print("res:", res)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_list():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = [1, 2, 3, 4]
            raise ValueError(x)

    with pytest.raises(ValueError) as raise_info_list:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "[1, 2, 3, 4]" in str(raise_info_list.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_tuple():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = (1, 2, 3, 4)
            raise ValueError(x)

    with pytest.raises(ValueError) as raise_info_tuple:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "(1, 2, 3, 4)" in str(raise_info_tuple.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_string_tuple():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = (1, 2, 3, 4)
            raise ValueError("test_string_tuple", x)

    with pytest.raises(ValueError) as raise_info_string_tuple:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "'test_string_tuple', (1, 2, 3, 4)" in str(
        raise_info_string_tuple.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_string_list():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = [1, 2, 3, 4]
            raise ValueError("test_string_list", x)

    with pytest.raises(ValueError) as raise_info_string_list:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "'test_string_list', [1, 2, 3, 4]" in str(
        raise_info_string_list.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_float():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = 1.1
            raise ValueError(x)

    with pytest.raises(ValueError) as raise_info_float:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "1.1" in str(raise_info_float.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_nested_list():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = [1, 2]
            y = [x, x]
            raise ValueError(x, y)

    with pytest.raises(ValueError) as raise_info_nested_list:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "([1, 2], [[1, 2], [1, 2]])" in str(
        raise_info_nested_list.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_nested_tuple():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = (1, 2)
            y = (x, x)
            raise ValueError(x, y)

    with pytest.raises(ValueError) as raise_info_nested_tuple:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "((1, 2), ((1, 2), (1, 2)))" in str(
        raise_info_nested_tuple.value)


@pytest.mark.skip(reason='Not support dict yet')
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_dict():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            x = {'a': 1, 'b': 2}
            raise ValueError(x)

    with pytest.raises(ValueError) as raise_info_dict:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "{'a': 1, 'b': 2}" in str(raise_info_dict.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_raise_joinedstr_tensor():
    """
    Feature: graph raise by JIT Fallback.
    Description: Test raise.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def construct(self):
            raise RuntimeError(f"The input should not be {Tensor([1])}.")

    with pytest.raises(RuntimeError) as raise_info_joinedstr_tensor:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "The input should not be [1]" in str(
        raise_info_joinedstr_tensor.value)


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
    assert "('The input can not be ', Tensor(shape=[], dtype=Int64, value= 11), '.')" or \
        "('The input can not be ', Tensor(shape=[1], dtype=Int64, value= [11]), '.')" in str(
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
    assert "[Tensor(shape=[], dtype=Int64, value= 1)," or \
        "(Tensor(shape=[1], dtype=Int64, value= [1])," in str(
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
    assert "(Tensor(shape=[], dtype=Int64, value= 1)," or \
        "(Tensor(shape=[1], dtype=Int64, value= [1])," in str(
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
    assert "('test_string_tuple', (Tensor(shape=[], dtype=Int64, value= 1)" or \
        "('test_string_tuple', (Tensor(shape=[1], dtype=Int64, value= [1])" in str(
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


@pytest.mark.skip(reason='Not support dict yet')
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

    with pytest.raises(ValueError) as raise_info_list:
        net = RaiseNet()
        res = net()
        print("res:", res)
    assert "{'x': Tensor(shape=[], dtype=Int64, value= 1)" or \
        "{'x': Tensor(shape=[1], dtype=Int64, value= [1])" in str(
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
        def construct(self, x, y):  # pylint: disable=R1711
            if x == y:
                raise RuntimeError(f"The input should not be {x}.")
            return None

    with pytest.raises(RuntimeError) as raise_info_joinedstr_tensor:
        net = RaiseNet()
        x = Tensor(1)
        y = Tensor(1)
        res = net(x, y)
        print("res:", res)
    assert "The input should not be 1" in str(
        raise_info_joinedstr_tensor.value)
