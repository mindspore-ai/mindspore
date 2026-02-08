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
""" test kwargs. """
import pytest
from mindspore import Tensor, context, nn, jit

context.set_context(mode=context.GRAPH_MODE)


def test_parser_args_var_kwargs_name_loss():
    """
    Feature: Support the kwargs.
    Description: Support the kwargs in  graph mode.
    Expectation: No error.
    """
    class Net(nn.Cell):
        def construct(self, *, b, c):
            x = b + c
            return x

    net = Net()
    with pytest.raises(TypeError) as raise_info:
        net(Tensor([5, 5, 6]), c=2)
    assert "too many positional arguments" in str(raise_info.value)


@jit
def func1(x=0, y=0, z=0):
    return x * 100 + y * 10 + z


def test_jit_with_kwargs_input_1():
    """
    Feature: Support the kwargs.
    Description: Number of passed parameters is more than interface parameters.
    Expectation: Expect the correct error info.
    """
    with pytest.raises(TypeError) as raise_info:
        func1(1, 2, 3, 4)  # pylint:disable=E1121
    assert "needs 0 positional argument and 3 default argument, total 3, but got 4." in str(raise_info.value)


def test_jit_with_kwargs_input_2():
    """
    Feature: Support the kwargs.
    Description: multiple values for default argument
    Expectation: Expect the correct error info.
    """
    with pytest.raises(TypeError) as raise_info:
        func1(1, 2, y=4)  # pylint:disable=E1124
    assert "multiple values for argument" in str(raise_info.value)


def test_jit_with_kwargs_input_3():
    """
    Feature: Support the kwargs.
    Description: Unexpected keyword.
    Expectation: Expect the correct error info.
    """
    with pytest.raises(TypeError) as raise_info:
        func1(1, 2, l=5)
    assert "got an unexpected keyword" in str(raise_info.value)


def test_jit_with_kwargs_input_4():
    """
    Feature: Support the kwargs.
    Description: unexpected keyword argument.
    Expectation: Expect the correct error info.
    """
    with pytest.raises(TypeError) as raise_info:
        func1(x=1, q=3, y=2)
    assert "got an unexpected keyword argument" in str(raise_info.value)


@jit
def func2(x, y=0, z=0):
    return x * 100 + y * 10 + z

@jit
def func3(*, x=0, y=0, z=0):
    return x * 100 + y * 10 + z

@jit
def func4(x, *, y=0, z=0):
    return x * 100 + y * 10 + z


def test_jit_with_kwargs_input_5():
    """
    Feature: Support the kwargs.
    Description: Mismatched parameter count.
    Expectation: Expect the correct error info.
    """
    with pytest.raises(TypeError) as raise_info:
        func2(1, 2, 3, 4)  # pylint:disable=E1121
    assert "needs 1 positional argument and 2 default argument, total 3, but got 4." in str(raise_info.value)


def test_jit_with_kwargs_input_6():
    """
    Feature: Support the kwargs.
    Description: Mismatched parameter count.
    Expectation: Expect the correct error info.
    """
    with pytest.raises(TypeError) as raise_info:
        func3(1, 2, 3)  # pylint:disable=E1121
    assert "needs 0 positional argument and 0 default argument, total 0, but got 3." in str(raise_info.value)


def test_jit_with_kwargs_input_7():
    """
    Feature: Support the kwargs.
    Description: Mismatched parameter count.
    Expectation: Expect the correct error info.
    """
    with pytest.raises(TypeError) as raise_info:
        func4(1, 2, 3, 4)  # pylint:disable=E1121
    assert "needs 1 positional argument and 0 default argument, total 1, but got 4." in str(raise_info.value)


def test_jit_with_kwargs_input_8():
    """
    Feature: Support the kwargs.
    Description: Missing a positional argument.
    Expectation: Expect the correct error info.
    """
    with pytest.raises(TypeError) as raise_info:
        func2(q=1, z=3, y=2)  # pylint:disable=E1120
    assert "missing a required argument" in str(raise_info.value)


def test_jit_with_kwargs_input_9():
    """
    Feature: Support the kwargs.
    Description: Unexpected keyword.
    Expectation: Expect the correct error info.
    """
    with pytest.raises(TypeError) as raise_info:
        func2(x=1, q=3)
    assert "got an unexpected keyword" in str(raise_info.value)
