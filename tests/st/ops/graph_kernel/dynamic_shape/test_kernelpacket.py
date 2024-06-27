# Copyright 2024 Huawei Technologies Co., Ltd
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

import os
import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor
import pytest


def helper(net_type, inputs_dyn, inputs):
    ms.set_context(mode=ms.GRAPH_MODE, enable_graph_kernel=True)
    os.environ["MS_DEV_ENABLE_KERNEL_PACKET"] = "off"
    net1 = net_type()
    net1.set_inputs(*inputs_dyn)
    expect = net1(*inputs)
    os.environ["MS_DEV_ENABLE_KERNEL_PACKET"] = "on"
    net2 = net_type()
    net2.set_inputs(*inputs_dyn)
    output = net2(*inputs)
    del os.environ["MS_DEV_ENABLE_KERNEL_PACKET"]
    if isinstance(expect, ms.Tensor):
        assert np.allclose(expect.asnumpy(), output.asnumpy(), 1e-4, 1e-4)
    else:
        assert all(np.allclose(e.asnumpy(), o.asnumpy(), 1e-4, 1e-4) for e, o in zip(expect, output))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_reshape():
    """
    Feature: KernelPacket
    Description: test kernelpacket with reshape
    Expectation: success
    """
    class ReshapeNet(nn.Cell):
        def __init__(self):
            super(ReshapeNet, self).__init__()
            self.shape = ops.Shape()
            self.reshape = ops.Reshape()

        def construct(self, x, y):
            shape = self.shape(x)
            a = shape[0]
            y2 = self.reshape(y, (a, a))
            z = y2 + x
            return z
    x_dyn = Tensor(shape=[None], dtype=ms.float32)
    y_dyn = Tensor(shape=[None, None], dtype=ms.float32)

    x = Tensor(np.random.random([2]), dtype=ms.float32)
    y = Tensor(np.random.random([2, 2]), dtype=ms.float32)
    helper(ReshapeNet, (x_dyn, y_dyn), (x, y))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_reducesum():
    """
    Feature: KernelPacket
    Description: test kernelpacket with ReduceSum
    Expectation: success
    """
    class ReduceSumNet(nn.Cell):
        def __init__(self):
            super(ReduceSumNet, self).__init__()
            self.add = ops.Add()
            self.shape = ops.Shape()
            self.reducesum = ops.ReduceSum(True, True)

        def construct(self, x):
            shape = self.shape(x)
            b = shape[1]
            y = self.reducesum(x, b)
            return y
    x_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    x = Tensor(np.array([[2], [1]]), dtype=ms.float32)
    helper(ReduceSumNet, (x_dyn,), (x,))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [ms.float16, ms.float32])
def test_shape_cast(data_type):
    """
    Feature: KernelPacket
    Description: test kernelpacket with host-device ops
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            t = ops.shape(x)[-1] ** 0.5
            return t * y
    dyn = Tensor(shape=[None, None], dtype=data_type)
    x = Tensor(np.random.random([32, 32]), dtype=data_type)
    y = Tensor(np.random.random([16, 16]), dtype=data_type)
    helper(Net, (dyn, dyn), (x, y))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_stridedslice():
    """
    Feature: KernelPacket
    Description: test kernelpacket with stridedslice
    Expectation: success
    """
    class SdNet(nn.Cell):
        def __init__(self):
            super(SdNet, self).__init__()
            self.stack = ops.Stack()
            self.tensorshape = ops.TensorShape()
            self.stridedslice = ops.StridedSlice(2, 2, 0, 0, 1)

        def construct(self, x):
            shape = self.tensorshape(x)
            shape2 = shape[1]
            a = Tensor(1, ms.int64)
            shape3 = self.stack([a, shape2])
            y = self.stridedslice(x, (0, 0), shape3, (1, 1))
            return y
    x_dyn = Tensor(shape=[32, None], dtype=ms.float32)
    input_x = Tensor(np.random.random([32, 16]), dtype=ms.float32)
    helper(SdNet, (x_dyn,), (input_x,))


class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.grad = ops.composite.GradOperation(get_all=True)

    def construct(self, *inputs):
        return self.grad(self.net)(*inputs)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_concat_grad():
    """
    Feature: KernelPacket
    Description: test kernelpacket with slice
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.abs = ops.Abs()

        def construct(self, x, y):
            t = ops.concat((self.abs(x), self.abs(y)), 0)
            return self.abs(t)

    dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    x = ms.Tensor(np.random.rand(1, 3).astype(np.float32))
    y = ms.Tensor(np.random.rand(5, 3).astype(np.float32))
    helper(lambda: GradNet(Net()), (dyn, dyn), (x, y))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_stridedslice_grad():
    """
    Feature: KernelPacket
    Description: test kernelpacket with stridedslicegrad
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.stridedslice = ops.StridedSlice(0, 0, 0, 0, 0)

        def construct(self, x, y):
            shape = ops.shape(y)
            end = (10, shape[0])
            return self.stridedslice(x, (0, 0), end, (1, 1))

    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    x = ms.Tensor(np.random.rand(32, 16).astype(np.float32))
    y = ms.Tensor(np.random.rand(10).astype(np.float32))
    helper(lambda: GradNet(Net()), (x_dyn, y_dyn), (x, y))
