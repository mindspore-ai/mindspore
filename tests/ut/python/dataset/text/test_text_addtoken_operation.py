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
# ==============================================================================
"""text transform - addtoken"""

import numpy as np
import pytest
from mindspore.dataset import text
from mindspore import log as logger


def test_addtoken_operation_01():
    """
    Feature: AddToken op
    Description: Test AddToken op with different begin parameter and input dimensions
    Expectation: Successfully add token at beginning or end of sequences
    """
    # test begin true 1d
    input_one_dimension = ['a', 'b', 'c', 'd', 'e']
    expected = ['TOKEN', 'a', 'b', 'c', 'd', 'e']
    out = text.AddToken(token='TOKEN', begin=True)
    result = out(input_one_dimension)
    assert np.array_equal(result, np.array(expected))

    # test begin false 1d
    input_one_dimension = ['a', 'b', 'c', 'd', 'e']
    expected = ['a', 'b', 'c', 'd', 'e', 'TOKEN']
    out = text.AddToken(token='TOKEN', begin=False)
    result = out(input_one_dimension)
    assert np.array_equal(result, np.array(expected))

    # test begin default true
    input_one_dimension = ['a', 'b', 'c', '5', '9']
    expected = ['TOKEN', 'a', 'b', 'c', '5', '9']
    out = text.AddToken(token='TOKEN')
    result = out(input_one_dimension)
    assert np.array_equal(result, np.array(expected))

    # test begin true 2d
    input_two_dimension = [['a', 'b', 'c', 'd'], ['A', 'B', 'C', 'D'], ['1', '2', '3', '4']]
    expected = [['TOKEN', 'a', 'b', 'c', 'd'], ['TOKEN', 'A', 'B', 'C', 'D'], ['TOKEN', '1', '2', '3', '4']]
    out = text.AddToken(token='TOKEN', begin=True)
    result = out(input_two_dimension)
    assert np.array_equal(result, np.array(expected))

    # test begin false 2d
    input_two_dimension = [['a', 'b', 'c', 'd'], ['A', 'B', 'C', 'D'], ['1', '2', '3', '4']]
    expected = [['a', 'b', 'c', 'd', 'TOKEN'], ['A', 'B', 'C', 'D', 'TOKEN'], ['1', '2', '3', '4', 'TOKEN']]
    out = text.AddToken(token='TOKEN', begin=False)
    result = out(input_two_dimension)
    assert np.array_equal(result, np.array(expected))

    # test begin default 2d
    input_two_dimension = [['a', 'b', 'c', 'd'], ['A', 'B', 'C', 'D'], ['1', '2', '3', '4']]
    expected = [['TOKEN', 'a', 'b', 'c', 'd'], ['TOKEN', 'A', 'B', 'C', 'D'], ['TOKEN', '1', '2', '3', '4']]
    out = text.AddToken(token='TOKEN')
    result = out(input_two_dimension)
    assert np.array_equal(result, np.array(expected))

    # test input int -- when there are both int and str, numpy will convert int to str.
    input_value = [66, '100', 'aaa', 300]
    expected = ['test', '66', '100', 'aaa', '300']
    out = text.AddToken(token="test", begin=True)
    result = out(input_value)
    assert np.array_equal(result, np.array(expected))


def test_add_token_at_begin():
    """
    Feature: AddToken op
    Description: Test AddToken with begin = True
    Expectation: Output is equal to the expected output
    """
    input_one_dimension = ['a', 'b', 'c', 'd', 'e']
    expected = ['TOKEN', 'a', 'b', 'c', 'd', 'e']
    out = text.AddToken(token='TOKEN', begin=True)
    result = out(input_one_dimension)
    assert np.array_equal(result, np.array(expected))


def test_add_token_at_end():
    """
    Feature: AddToken op
    Description: Test AddToken with begin = False
    Expectation: Output is equal to the expected output
    """
    input_one_dimension = ['a', 'b', 'c', 'd', 'e']
    expected = ['a', 'b', 'c', 'd', 'e', 'TOKEN']
    out = text.AddToken(token='TOKEN', begin=False)
    result = out(input_one_dimension)
    assert np.array_equal(result, np.array(expected))


def test_add_token_fail():
    """
    Feature: AddToken op
    Description: fail to test AddToken
    Expectation: TypeError is raised as expected
    """
    try:
        _ = text.AddToken(token=1.0, begin=True)
    except TypeError as error:
        assert "Argument token with value 1.0 is not of type [<class 'str'>], but got <class 'float'>." in str(error)
    try:
        _ = text.AddToken(token='TOKEN', begin=12.3)
    except TypeError as error:
        assert "Argument begin with value 12.3 is not of type [<class 'bool'>], but got <class 'float'>." in str(error)


def test_addtoken_exception_01():
    """
    Feature: AddToken op
    Description: Test AddToken op with invalid parameter types and input dimensions
    Expectation: Raise expected exceptions for invalid inputs
    """
    # test token int
    with pytest.raises(TypeError) as e:
        _ = text.AddToken(token=1, begin=True)
    assert "Argument token with value 1 is not of type [<class 'str'>], but got <class 'int'>." in str(e.value)

    # test begin int
    with pytest.raises(TypeError) as e:
        _ = text.AddToken(token='test', begin=1)
    assert "Argument begin with value 1 is not of type [<class 'bool'>], but got <class 'int'>." in str(e.value)

    # test token int
    with pytest.raises(TypeError) as e:
        _ = text.AddToken(token='test', begin='True')
    assert "Argument begin with value True is not of type [<class 'bool'>], but got <class 'str'>" in str(e.value)

    # test input int
    input_value = 100
    with pytest.raises(RuntimeError) as e:
        out = text.AddToken(token="test", begin=True)
        _ = out(input_value)
    assert "input tensor rank should be 1 or 2" in str(e.value)

    # test input int
    input_value = "100"
    with pytest.raises(RuntimeError) as e:
        out = text.AddToken(token="test", begin=True)
        _ = out(input_value)
    assert "input tensor rank should be 1 or 2" in str(e.value)

    # test input int
    input_value = [100, 200, 300]
    with pytest.raises(RuntimeError) as e:
        out = text.AddToken(token="test", begin=True)
        _ = out(input_value)
    logger.info(e.value)
    assert "input tensor type should be string" in str(e.value)

    # test begin default 3d
    input_value = [[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']], [['h', 'j'], ['k', 'm']]]
    with pytest.raises(RuntimeError) as e:
        out = text.AddToken(token='TOKEN')
        _ = out(input_value)
    assert "input tensor rank should be 1 or 2" in str(e.value)
