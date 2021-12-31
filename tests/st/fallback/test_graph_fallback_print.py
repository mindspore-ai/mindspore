# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test graph fallback """
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, ms_function, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_print_1():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @ms_function
    def np_print():
        x = np.array([1, 2, 3, 4, 5])
        print("x: ", x)
        return Tensor(x)
    assert np.all(np_print().asnumpy() == np.array([1, 2, 3, 4, 5]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_print_2():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    class PrintNet(nn.Cell):
        def construct(self):
            x = np.array([1, 2, 3, 4, 5])
            print("x: ", x)
            return Tensor(x)

    net = PrintNet()
    res = net()
    print("res: ", res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tensor_print_1():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @ms_function
    def np_print():
        x = np.array([1, 2, 3, 4, 5])
        print("Tensor(x): ", Tensor(x))
        return Tensor(x)
    assert np.all(np_print().asnumpy() == np.array([1, 2, 3, 4, 5]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tensor_print_2():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    class PrintNet(nn.Cell):
        def construct(self):
            x = np.array([1, 2, 3, 4, 5])
            print("Tensor(x): ", Tensor(x))
            return Tensor(x)

    net = PrintNet()
    res = net()
    print("res: ", res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_print_cnode_1():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @ms_function
    def print_func(x, y):
        res_sum = x + y
        print("res_sum: ", res_sum)
        return res_sum

    x = Tensor(np.array([1, 2, 3, 4, 5]))
    y = Tensor(np.array([1, 2, 3, 4, 5]))
    res = print_func(x, y)
    print("res: ", res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_print_cnode_2():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @ms_function
    def print_func():
        x = Tensor(np.array([1, 2, 3, 4, 5]))
        y = Tensor(np.array([1, 2, 3, 4, 5]))
        res_sum = x + y
        print("res_sum: ", res_sum)
        return res_sum

    res = print_func()
    print("res: ", res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_print_cnode_3():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @ms_function
    def print_func():
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        res_sum = x + y
        print("res_sum: ", res_sum)
        return Tensor(res_sum)

    res = print_func()
    print("res: ", res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_print_validate_tuple():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @ms_function
    def print_func():
        x = Tensor(np.array([1, 2, 3, 4, 5]))
        y = Tensor(np.array([1, 2, 3, 4, 5]))
        tensor_sum = x + y
        print("tensor_sum: ", tensor_sum)
        np_x = np.array([1, 2, 3, 4, 5])
        np_y = np.array([1, 2, 3, 4, 5])
        np_sum = np_x + np_y
        print("np_sum: ", np_sum)
        return tensor_sum, np_sum

    with pytest.raises(RuntimeError) as err:
        res1, res2 = print_func()
        print("res1: ", res1)
        print("res2: ", res2)
    assert "Should not use Python object in runtime" in str(err.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_print_validate():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @ms_function
    def print_func():
        np_x = np.array([1, 2, 3, 4, 5])
        np_y = np.array([1, 2, 3, 4, 5])
        np_sum = np_x + np_y
        print("np_sum: ", np_sum)
        return np_sum

    with pytest.raises(RuntimeError) as err:
        res = print_func()
        print("res: ", res)
    assert "Should not use Python object in runtime" in str(err.value)
