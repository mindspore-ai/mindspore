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
""" test graph fallback control flow."""
import pytest
import numpy as np
from mindspore import Tensor, ms_function, context
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        for _ in range(3):
            y += x
        return y
    res = control_flow_for()
    assert res == 21


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        for _ in range(Tensor(3).astype("int32")):
            y += x
        return y

    with pytest.raises(RuntimeError, match="The type of inputs in range operator only support int64 number."):
        res = control_flow_for()
        assert res == 21


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for_zip():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for():
        tuple_x = (Tensor(1).astype("int32"), Tensor(3).astype("int32"), Tensor(5).astype("int32"))
        sum_x = Tensor(0).astype("int32")
        for x in zip(tuple_x):
            sum_x += x
        return sum_x

    res = control_flow_for()
    assert res == 9


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for():
        x = np.array([1, 3, 5])
        y = np.array([0, 2, 4])
        for _ in range(3):
            x = x + y
        return Tensor(x)
    res = control_flow_for()
    assert (res.asnumpy() == [1, 9, 17]).all()


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_single_for_builtin_function_sum():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for():
        x = np.array([1, 3, 5, 7, 9])
        y = np.array([0, 2, 4, 6, 8])
        result = x
        for _ in range(3):
            x = x + y
            result = sum(x, y)
        return Tensor(result)
    res = control_flow_for()
    assert (res.asnumpy() == [85, 87, 89, 91, 93]).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for_builtin_function_int():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for():
        x = np.array(1.1)
        for _ in range(3):
            x = x + int(x)
        return Tensor(x, mstype.float32)
    res = control_flow_for()
    assert res == 8.1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_for_builtin_function_list():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for():
        x = np.array([1.1, 2.2])
        for _ in range(3):
            x = x + list(x)
        return Tensor(x)
    res = control_flow_for()
    assert (res.asnumpy() == [8.8, 17.6]).all()
