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
""" test use undefined variables for error reporting in control flow scenarios"""
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


def test_use_local_variable_in_if_true_branch():
    """
    Feature: use undefined variables in if.
    Description: local variable 'y' referenced before assignment.
    Expectation: Raises UnboundLocalError.
    """
    class Net(nn.Cell):
        def construct(self, x):
            if x < 0:
                y = 0
            return y

    net = Net()
    with pytest.raises(UnboundLocalError) as err:
        net(Tensor([1], mstype.float32))
    assert "The local variable 'y' is not defined in false branch, " \
           "but defined in true branch." in str(err.value)
    assert "return y" in str(err.value)


def test_use_local_variable_in_if_false_branch():
    """
    Feature: use undefined variables in if.
    Description: local variable 'y' referenced before assignment.
    Expectation: Raises UnboundLocalError.
    """
    class Net(nn.Cell):
        def construct(self, x):
            if x < 0:
                print(x)
            else:
                y = 0
            return y

    net = Net()
    with pytest.raises(UnboundLocalError) as err:
        net(Tensor([1], mstype.float32))
    assert "The local variable 'y' is not defined in true branch, " \
           "but defined in false branch." in str(err.value)
    assert "return y" in str(err.value)


def test_use_local_variable_in_while_body_branch():
    """
    Feature: use undefined variables in while.
    Description: local variable 'y' referenced before assignment.
    Expectation: Raises UnboundLocalError.
    """
    class Net(nn.Cell):
        def construct(self, x):
            while x < 0:
                y = 0
            return y

    net = Net()
    with pytest.raises(UnboundLocalError) as err:
        net(Tensor([1], mstype.float32))
    assert "The local variable 'y' defined in the 'while' loop body " \
           "cannot be used outside of the loop body." in str(err.value)
    assert "return y" in str(err.value)


def test_use_local_variable_in_def_if():
    """
    Feature: use undefined variables in if with defined function.
    Description: local variable 'y' referenced before assignment.
    Expectation: Raises UnboundLocalError.
    """
    class Net(nn.Cell):
        def construct(self, x):
            def func(x):
                if x < 0:
                    y = 0
                return y

            return func(x)

    net = Net()
    with pytest.raises(UnboundLocalError) as err:
        net(Tensor([1], mstype.float32))
    assert "The local variable 'y' is not defined in false branch, " \
           "but defined in true branch." in str(err.value)
    assert "return y" in str(err.value)


def test_use_local_variable_in_def_while():
    """
    Feature: use undefined variables in while with defined function.
    Description: local variable 'y' referenced before assignment.
    Expectation: Raises UnboundLocalError.
    """
    class Net(nn.Cell):
        def construct(self, x):
            def func(x):
                while x < 0:
                    y = 0
                return y

            return func(x)

    net = Net()
    with pytest.raises(UnboundLocalError) as err:
        net(Tensor([1], mstype.float32))
    assert "The local variable 'y' defined in the 'while' loop body " \
           "cannot be used outside of the loop body." in str(err.value)
    assert "return y" in str(err.value)


def test_use_local_variable_in_while_with_nested_for():
    """
    Feature: use undefined variables in while with defined function.
    Description: local variable 'y' referenced before assignment.
    Expectation: Raises UnboundLocalError.
    """
    class Net(nn.Cell):
        def construct(self, x):
            for _ in range(3):
                for _ in range(4):
                    for _ in range(5):
                        while x < 0:
                            y = 0
            return y

    net = Net()
    with pytest.raises(UnboundLocalError) as err:
        net(Tensor([1], mstype.float32))
    assert "The local variable 'y' defined in the 'for' loop body " \
           "cannot be used outside of the loop body." in str(err.value)
    assert "return y" in str(err.value)


def test_use_local_variable_in_if_with_nested_for():
    """
    Feature: use undefined variables in while with defined function.
    Description: local variable 'y' referenced before assignment.
    Expectation: Raises UnboundLocalError.
    """
    class Net(nn.Cell):
        def construct(self, x):
            for _ in range(3):
                for _ in range(4):
                    for _ in range(5):
                        if x < 0:
                            y = 0
            return y

    net = Net()
    with pytest.raises(UnboundLocalError) as err:
        net(Tensor([1], mstype.float32))
    assert "The local variable 'y' defined in the 'for' loop body " \
           "cannot be used outside of the loop body." in str(err.value)
    assert "return y" in str(err.value)


def test_use_local_variable_in_for_if():
    """
    Feature: use undefined variables in while with defined function.
    Description: local variable 'y' referenced before assignment.
    Expectation: Raises UnboundLocalError.
    """
    class Net(nn.Cell):
        def construct(self, x):
            y = 0
            for _ in range(3):
                if True: # pylint: disable=using-constant-test
                    a = 1
                y += a
            return y

    net = Net()
    with pytest.raises(UnboundLocalError) as err:
        net(Tensor([1], mstype.float32))
    assert "The local variable 'a' is not defined in false branch, " \
           "but defined in true branch." in str(err.value)
    assert "a = 1" in str(err.value)


def test_use_local_variable_by_assigned_parameter():
    """
    Feature: use undefined variables in if.
    Description: local variable 'y' referenced before assignment.
    Expectation: Raises UnboundLocalError.
    """
    class Net(nn.Cell):
        def construct(self, x):
            if x < 0:
                y = x
            return y

    net = Net()
    with pytest.raises(UnboundLocalError) as err:
        net(Tensor([1], mstype.float32))
    assert "The local variable 'y' is not defined in false branch, " \
           "but defined in true branch." in str(err.value)
    assert "def construct(self, x):" in str(err.value)


def test_use_local_variable_by_assigned_parameter_for_if():
    """
    Feature: use undefined variables in for if.
    Description: local variable 'y' referenced before assignment.
    Expectation: Raises UnboundLocalError.
    """
    class Net(nn.Cell):
        def construct(self, x):
            for _ in range(3):
                if x < 0:
                    y = x
            return y

    net = Net()
    with pytest.raises(UnboundLocalError) as err:
        net(Tensor([1], mstype.float32))
    assert "The local variable 'y' defined in the 'for' loop body " \
           "cannot be used outside of the loop body." in str(err.value)
    assert "y = x" in str(err.value)


def test_function_args_same_name():
    """
    Feature: Parse function.
    Description: Function argument has the same same as the function.
    Expectation: No exception.
    """
    @ms.jit
    def f(f, x):
        return f + x

    assert f(1, 2) == 3
