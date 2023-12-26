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

import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor
from mindspore.ops.operations._inner_ops import DynamicBroadcastTo
import pytest


def helper(net_type, input_dyns, inputs, device_target):
    ms.set_context(mode=ms.GRAPH_MODE, device_target=device_target,
                   enable_graph_kernel=False)
    net1 = net_type()
    net1.set_inputs(*input_dyns)
    expect = net1(*inputs)
    ms.set_context(enable_graph_kernel=True)
    net2 = net_type()
    net2.set_inputs(*input_dyns)
    output = net2(*inputs)
    assert np.allclose(expect.asnumpy(), output.asnumpy())


class SdNet(nn.Cell):
    def __init__(self):
        super().__init__()
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stridedslice_gpu():
    """
    Feature: KernelPacket
    Description: test kernelpacket with stridedslice in gpu
    Expectation: success
    """
    x_dyn = Tensor(shape=[32, None], dtype=ms.float32)
    input_x = Tensor(np.random.random([32, 16]), dtype=ms.float32)
    helper(SdNet, (x_dyn,), (input_x,), "GPU")


class ReshapeNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()

    def construct(self, x, y):
        shape = self.shape(x)
        a = shape[0]
        y2 = self.reshape(y, (a, a))
        z = y2 + x
        return z


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reshape_gpu():
    """
    Feature: KernelPacket
    Description: test kernelpacket with reshape in gpu
    Expectation: success
    """
    x_dyn = Tensor(shape=[None], dtype=ms.float32)
    y_dyn = Tensor(shape=[None, None], dtype=ms.float32)

    x = Tensor(np.random.random([2]), dtype=ms.float32)
    y = Tensor(np.random.random([2, 2]), dtype=ms.float32)
    helper(ReshapeNet, (x_dyn, y_dyn), (x, y), "GPU")


class DynamicBroadcastToNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.shape = ops.Shape()
        self.dbt = DynamicBroadcastTo()

    def construct(self, x):
        shape = list(self.shape(x))
        a = shape[0]
        z = self.dbt(x, (a, a))
        return z


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_broadcast_to_gpu():
    """
    Feature: KernelPacket
    Description: test kernelpacket with DynamicBroadcastTo in gpu
    Expectation: success
    """
    x_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    x = Tensor(np.random.random([4, 4]), dtype=ms.float32)
    helper(DynamicBroadcastToNet, (x_dyn,), (x,), "GPU")


class ReduceSumNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.Add()
        self.shape = ops.Shape()
        self.reducesum = ops.ReduceSum()

    def construct(self, x):
        shape = list(self.shape(x))
        b = shape[1]
        y = self.reducesum(x, b)
        return y


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reducesum_gpu():
    """
    Feature: KernelPacket
    Description: test kernelpacket with ReduceSum in gpu
    Expectation: success
    """
    x_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    x = Tensor(np.array([[2], [1]]), dtype=ms.float32)
    helper(ReduceSumNet, (x_dyn,), (x,), "GPU")
