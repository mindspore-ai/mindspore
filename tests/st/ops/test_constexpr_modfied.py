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
import mindspore as ms
from mindspore import Tensor, ops, nn
from mindspore import context
context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_repeat_interleave():
    """
    Feature: repeat_interleave func
    Description: Verify the result of repeat_interleave
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.repeat_interleave

        def construct(self, x):
            return self.func(x, repeats=2, dim=0)

    x = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), ms.int32)
    expect = Tensor(
        np.array([[0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5]]), ms.int32)
    net = Net()
    output = net(x)
    print(output)
    assert np.allclose(output.asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tensor_dot():
    """
    Feature: tensor_dot func
    Description: Verify the result of tensor_dot
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.tensor_dot

        def construct(self, input_x1, input_x2):
            return self.func(input_x1, input_x2, ((0, 1), (1, 2)))

    input_x1 = Tensor(np.ones(shape=[1, 2, 3]), ms.float32)
    input_x2 = Tensor(np.ones(shape=[3, 1, 2]), ms.float32)
    expect = Tensor(
        np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]), ms.float32)
    net = Net()
    output = net(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dot():
    """
    Feature: dot func
    Description: Verify the result of dot
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.dot

        def construct(self, input_x1, input_x2):
            return self.func(input_x1, input_x2)

    input_x1 = Tensor(np.ones(shape=[1, 2, 3]), ms.float32)
    input_x2 = Tensor(np.ones(shape=[1, 3, 2]), ms.float32)
    expect = Tensor(np.array([[[[3, 3]], [[3, 3]]]]), ms.float32)
    net = Net()
    output = net(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_batch_dot():
    """
    Feature: batch_dot func
    Description: Verify the result of batch_dot
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.batch_dot

        def construct(self, input_x1, input_x2):
            return self.func(input_x1, input_x2, (-1, -2))

    input_x1 = Tensor(np.ones(shape=[2, 2, 3]), ms.float32)
    input_x2 = Tensor(np.ones(shape=[2, 3, 2]), ms.float32)
    expect = Tensor(
        np.array([[[3, 3], [3, 3]], [[3, 3], [3, 3]]]), ms.float32)
    net = Net()
    output = net(input_x1, input_x2)
    assert np.allclose(output.asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_cummin():
    """
    Feature: cummin func
    Description: Verify the result of cummin
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.cummin

        def construct(self, a):
            return self.func(a, 0)

    a = Tensor([-0.2284, -0.6628, 0.0975, 0.2680, -1.3298, -0.4220], ms.float32)
    expect = Tensor(np.array(
        [-0.2284, -0.6628, -0.6628, -0.6628, -1.3298, -1.3298]), ms.float32)
    net = Net()
    output = net(a)
    assert np.allclose(output[0].asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul():
    """
    Feature: matmul func
    Description: Verify the result of matmul
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.matmul

        def construct(self, a, b):
            return self.func(a, b)

    x1 = Tensor(np.arange(1*3).reshape(1, 3), ms.float32)
    x2 = Tensor(np.arange(3*2).reshape(3, 2), ms.float32)
    expect = Tensor(np.array([10, 13]), ms.float32)
    net = Net()
    output = net(x1, x2)
    assert np.allclose(output.asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_flip():
    """
    Feature: flip func
    Description: Verify the result of flip
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.func = ops.flip

        def construct(self, a):
            return self.func(a, (0, 2))

    x = Tensor(np.arange(8).reshape((2, 2, 2)))
    expect = Tensor(
        np.array([[[5, 4], [7, 6]], [[1, 0], [3, 2]]]), ms.int32)
    net = Net()
    output = net(x)
    assert np.allclose(output.asnumpy(), expect.asnumpy())
