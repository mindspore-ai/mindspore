# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
""" test syntax for logic expression """

import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor

context.set_context(mode=context.GRAPH_MODE)


class LogicAnd(nn.Cell):
    def __init__(self):
        super(LogicAnd, self).__init__()
        self.m = 1

    def construct(self, x, y):
        and_v = x and y
        return and_v


class LogicAndSpec(nn.Cell):
    def __init__(self, x, y):
        super(LogicAndSpec, self).__init__()
        self.x = x
        self.y = y

    def construct(self, x, y):
        and_v = self.x and self.y
        return and_v


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_int_and_int():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    net = LogicAnd()
    ret = net(1, 2)
    assert ret == 2


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_float_and_float():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    net = LogicAnd()
    ret = net(1.89, 1.99)
    assert ret == 1.99


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_float_and_int():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    net = LogicAnd()
    ret = net(1.89, 1)
    assert ret == 1


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_tensor_1_int_and_tensor_1_int():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    net = LogicAnd()
    x = Tensor(np.ones([1], np.int32))
    y = Tensor(np.zeros([1], np.int32))
    ret = net(x, y)
    assert (ret.asnumpy() == [0]).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_tensor_1_float_and_tensor_1_int():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    with pytest.raises(TypeError, match="Cannot join the return values of different branches"):
        net = LogicAnd()
        x = Tensor(np.ones([1], np.float))
        y = Tensor(np.zeros([1], np.int32))
        ret = net(x, y)
        print(ret)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_tensor_1_int_and_int():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    with pytest.raises(TypeError, match="Cannot join the return values of different branches"):
        net = LogicAnd()
        x = Tensor(np.ones([1], np.int32))
        y = 2
        ret = net(x, y)
        print(ret)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_tensor_2x2_int_and_tensor_2x2_int():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    with pytest.raises(ValueError) as err:
        net = LogicAnd()
        x = Tensor(np.ones([2, 2], np.int32))
        y = Tensor(np.zeros([2, 2], np.int32))
        ret = net(x, y)
        print(ret)
    assert "Only tensor which shape is () or (1,) can be converted to bool, but got tensor shape is (2, 2)" in str(err)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_int_and_str():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    net = LogicAnd()
    ret = net(1, "cba")
    assert ret == "cba"


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_int_and_str_2():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    net = LogicAndSpec(1, "cba")
    ret = net(1, 2)
    assert ret == "cba"


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_str_and_str():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    net = LogicAndSpec("abc", "cba")
    ret = net(1, 2)
    assert ret == "cba"


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_list_int_and_list_int():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    net = LogicAnd()
    ret = net([1, 2, 3], [3, 2, 1])
    assert ret == [3, 2, 1]


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_list_int_and_int():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    net = LogicAnd()
    ret = net([1, 2, 3], 1)
    assert ret == 1


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_list_int_and_str():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    net = LogicAndSpec([1, 2, 3], "aaa")
    ret = net(1, 2)
    assert ret == "aaa"


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_list_int_and_list_str():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    net = LogicAndSpec([1, 2, 3], ["1", "2", "3"])
    ret = net(1, 2)
    assert ret == ["1", "2", "3"]


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ms_syntax_operator_logic_list_str_and_tensor_int():
    """
    Feature: simple expression
    Description: test logic and operator.
    Expectation: No exception
    """
    left = ["1", "2", "3"]
    right = Tensor(np.ones([2, 2], np.int32))
    net = LogicAndSpec(left, right)
    ret = net(1, 2)
    assert (ret.asnumpy() == [[1, 1], [1, 1]]).all()
