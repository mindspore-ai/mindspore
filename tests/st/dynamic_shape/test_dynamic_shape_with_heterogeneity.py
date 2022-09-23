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
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)


class Unique(nn.Cell):
    def __init__(self):
        super(Unique, self).__init__()
        self.unique_cpu = P.Unique().set_device("CPU")

    def construct(self, x):
        x, y = self.unique_cpu(x)
        return (x, y)


class UniqueSquare(nn.Cell):
    def __init__(self):
        super(UniqueSquare, self).__init__()
        self.unique_cpu = P.Unique().set_device("CPU")
        self.square = P.Square()

    def construct(self, x):
        x, _ = self.unique_cpu(x)
        return self.square(x)


class UniqueSquareRelu(nn.Cell):
    def __init__(self):
        super(UniqueSquareRelu, self).__init__()
        self.unique_cpu = P.Unique().set_device("CPU")
        self.square_cpu = P.Square().set_device("CPU")
        self.relu = P.ReLU()

    def construct(self, x):
        x, _ = self.unique_cpu(x)
        x = self.square_cpu(x)
        return self.relu(x)


class UniqueReshapeAdd(nn.Cell):
    def __init__(self):
        super(UniqueReshapeAdd, self).__init__()
        self.unique_cpu = P.Unique().set_device("CPU")
        self.unique = P.Unique()
        self.reshape_cpu = P.Reshape().set_device("CPU")
        self.reshape = P.Reshape()
        self.add = P.Add()

    def construct(self, x, y):
        x, _ = self.unique_cpu(x)
        x = self.reshape(x, (3, 1))
        y, _ = self.unique(y)
        y = self.reshape_cpu(y, (3, 1))
        return self.add(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_unique():
    """
    Feature: Dynamic shape with heterogeneity.
    Description: Test unique kernel in dynamic shape with heterogeneity scenarios.
    Expectation: The value and shape of output are the expected values.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 3]), mstype.float32)
    net = Unique()
    output = net(x)
    expect1 = np.array([1, 2, 3])
    expect2 = np.array([0, 0, 1, 1, 2, 2])
    assert (output[0].asnumpy() == expect1).all()
    assert (output[1].asnumpy() == expect2).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_unique_square():
    """
    Feature: Dynamic shape with heterogeneity.
    Description: Test unique and square kernels in dynamic shape with heterogeneity scenarios.
    Expectation: The value and shape of output are the expected values.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 3]), mstype.float32)
    net = UniqueSquare()
    output = net(x)
    expect = np.array([1, 4, 9])
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_unique_square_relu():
    """
    Feature: Dynamic shape with heterogeneity.
    Description: Test unique, square and relu kernels in dynamic shape with heterogeneity scenarios.
    Expectation: The value and shape of output are the expected values.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 3]), mstype.float32)
    net = UniqueSquareRelu()
    output = net(x)
    expect = np.array([1, 4, 9])
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_unique_reshape_add():
    """
    Feature: Dynamic shape with heterogeneity.
    Description: Test unique, reshape and add kernels in dynamic shape with heterogeneity scenarios.
    Expectation: The value and shape of output are the expected values.
    """
    x = Tensor(np.array([1, 1, 2, 2, 3, 3]), mstype.int32)
    y = Tensor(np.array([4, 4, 5, 5, 6, 6]), mstype.int32)
    net = UniqueReshapeAdd()
    output = net(x, y)
    expect = np.array([[5], [7], [9]])
    assert (output.asnumpy() == expect).all()
