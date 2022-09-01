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
""" test bitwise operator """
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bitwise_and_1():
    """
    Feature: bitwise and operator
    Description: test bitwise and operator between Tensor and Tensor
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            res = x & y
            return res

    x = Tensor(np.array([1, 2, -4]))
    y = Tensor(np.array([3, 4, -5]))
    net = Net()
    result = net(x, y)
    assert np.allclose(result.asnumpy(), np.array([1, 0, -8]))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bitwise_and_2():
    """
    Feature: bitwise and operator
    Description: test bitwise and operator between Tensor and Number
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            res = x & y
            return res

    x = Tensor(np.array([1, 2, -4]))
    y = 1
    net = Net()
    result = net(x, y)
    assert np.allclose(result.asnumpy(), np.array([1, 0, 0]))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bitwise_and_3():
    """
    Feature: bitwise and operator
    Description: test bitwise and operator between Number and Tensor
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            res = x & y
            return res

    x = 1
    y = Tensor(np.array([1, 2, -4]))
    net = Net()
    result = net(x, y)
    assert np.allclose(result.asnumpy(), np.array([1, 0, 0]))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bitwise_or_1():
    """
    Feature: bitwise or operator
    Description: test bitwise or operator between Tensor and Tensor
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            res = x | y
            return res

    x = Tensor(np.array([1, 2, -4]))
    y = Tensor(np.array([3, 4, -5]))
    net = Net()
    result = net(x, y)
    assert np.allclose(result.asnumpy(), np.array([3, 6, -1]))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bitwise_or_2():
    """
    Feature: bitwise or operator
    Description: test bitwise or operator between Tensor and Number
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            res = x | y
            return res

    x = Tensor(np.array([1, 2, -4]))
    y = 1
    net = Net()
    result = net(x, y)
    assert np.allclose(result.asnumpy(), np.array([1, 3, -3]))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bitwise_or_3():
    """
    Feature: bitwise or operator
    Description: test bitwise or operator between Number and Tensor
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            res = x | y
            return res

    x = 1
    y = Tensor(np.array([1, 2, -4]))
    net = Net()
    result = net(x, y)
    assert np.allclose(result.asnumpy(), np.array([1, 3, -3]))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bitwise_xor_1():
    """
    Feature: bitwise xor operator
    Description: test bitwise xor operator between Tensor and Tensor
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            res = x ^ y
            return res

    x = Tensor(np.array([1, 2, -4]))
    y = Tensor(np.array([3, 4, -5]))
    net = Net()
    result = net(x, y)
    assert np.allclose(result.asnumpy(), np.array([2, 6, 7]))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bitwise_xor_2():
    """
    Feature: bitwise xor operator
    Description: test bitwise xor operator between Tensor and Number
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            res = x ^ y
            return res

    x = Tensor(np.array([1, 2, -4]))
    y = 1
    net = Net()
    result = net(x, y)
    assert np.allclose(result.asnumpy(), np.array([0, 3, -3]))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bitwise_xor_3():
    """
    Feature: bitwise xor operator
    Description: test bitwise xor operator between Number and Tensor
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            res = x ^ y
            return res

    x = 1
    y = Tensor(np.array([1, 2, -4]))
    net = Net()
    result = net(x, y)
    assert np.allclose(result.asnumpy(), np.array([0, 3, -3]))
