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

"""Kernelpacket cases."""
import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor, JitConfig
import pytest


def helper(net, inputs_dyn, inputs, expect):
    ms.set_context(mode=ms.GRAPH_MODE)
    net.set_jit_config(JitConfig(jit_level="O1"))
    net.set_inputs(*inputs_dyn)
    output = net(*inputs)
    if not isinstance(expect, (list, tuple)):
        assert np.allclose(expect, output.asnumpy(), 1e-4, 1e-4)
    else:
        assert all(np.allclose(e, o.asnumpy(), 1e-4, 1e-4) for e, o in zip(expect, output))


def test_reshape():
    """
    Feature: KernelPacket
    Description: test kernelpacket with reshape
    Expectation: success
    """

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

    def calc(x, y):
        a = x.shape[0]
        y2 = np.reshape(y, (a, a))
        z = y2 + x
        return z

    x_dyn = Tensor(shape=[None], dtype=ms.float32)
    y_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    x = np.random.random([2]).astype(np.float32)
    y = np.random.random([2, 2]).astype(np.float32)
    helper(ReshapeNet(), (x_dyn, y_dyn), (Tensor(x), Tensor(y)), calc(x, y))


def test_reducesum():
    """
    Feature: KernelPacket
    Description: test kernelpacket with ReduceSum
    Expectation: success
    """

    class ReduceSumNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ops.Add()
            self.shape = ops.Shape()
            self.reducesum = ops.ReduceSum(True, True)

        def construct(self, x):
            shape = self.shape(x)
            b = shape[1]
            y = self.reducesum(x, b)
            return y

    def calc(x):
        return np.sum(x, x.shape[1], keepdims=True)

    x_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    x = np.array([[2], [1]], dtype=np.float32)
    helper(ReduceSumNet(), (x_dyn,), (Tensor(x),), calc(x))


@pytest.mark.parametrize('data_type', [(ms.float16, np.float16), (ms.float32, np.float32)])
def test_fuse_host_ops(data_type):
    """
    Feature: KernelPacket
    Description: test kernelpacket with host-device ops
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, y, z):
            # Shape-RealTupleGetItem-ScalarPow-ScalarToTensor-Mul
            t = ops.shape(x)[-1] ** 0.5
            # Shape-RealTupleGetItem-ScalarDiv-ScalarToTensor-Mul
            sy = ops.shape(y)
            t2 = sy[0] / sy[1]
            return t * z + t2 * z

    def calc(x, y, z):
        t = x.shape[-1] ** 0.5
        sy = y.shape
        t2 = sy[0] / sy[1]
        return t * z + t2 * z

    dyn = Tensor(shape=[None, None], dtype=data_type[0])
    x = np.random.random([32, 32]).astype(data_type[1])
    y = np.random.random([20, 15]).astype(data_type[1])
    z = np.random.random([16, 16]).astype(data_type[1])
    helper(Net(), (dyn, dyn, dyn), (Tensor(x), Tensor(y), Tensor(z)), calc(x, y, z))


def test_stridedslice():
    """
    Feature: KernelPacket
    Description: test kernelpacket with stridedslice
    Expectation: success
    """

    class SdNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.stack = ops.Stack()
            self.tensorshape = ops.operations._inner_ops.TensorShape()
            self.stridedslice = ops.StridedSlice(2, 2, 0, 0, 1)

        def construct(self, x):
            shape = self.tensorshape(x)
            shape2 = shape[1]
            a = Tensor(1, ms.int64)
            shape3 = self.stack([a, shape2])
            y = self.stridedslice(x, (0, 0), shape3, (1, 1))
            return y

    def calc(x):
        shape = x.shape
        return x[0:1, 0:shape[1]]

    x_dyn = Tensor(shape=[32, None], dtype=ms.float32)
    x = np.random.random([32, 16]).astype(np.float32)
    helper(SdNet(), (x_dyn,), (Tensor(x),), calc(x))


def test_matmul_only_shape():
    """
    Feature: KernelPacket
    Description: test kernelpacket to fuse the only-shape-depended ops.
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, p1, p2, p3, p4):
            m = ops.matmul(p1, p2)
            a = ops.add(m, p3)
            return ops.reshape(p4, ops.shape(a))

    def calc(p1, p2, p3, p4):
        m = np.matmul(p1, p2)
        a = np.add(m, p3)
        return p4.reshape(a.shape)

    p_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    p1 = np.random.random([1, 16]).astype(np.float32)
    p2 = np.random.random([16, 8]).astype(np.float32)
    p3 = np.random.random([32, 1]).astype(np.float32)
    p4 = np.random.random([64, 4]).astype(np.float32)
    ms.set_context(graph_kernel_flags="--enable_cluster_ops_only=Add")
    helper(Net(), (p_dyn, p_dyn, p_dyn, p_dyn), (Tensor(p1), Tensor(p2), Tensor(p3), Tensor(p4)), calc(p1, p2, p3, p4))


class GradNet(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad = ops.composite.GradOperation(get_all=True)

    def construct(self, *inputs):
        return self.grad(self.net)(*inputs)


def test_concat_grad():
    """
    Feature: KernelPacket
    Description: test kernelpacket with slice
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.abs = ops.Abs()

        def construct(self, x, y):
            t = ops.concat((self.abs(x), self.abs(y)), 0)
            return self.abs(t)

    def calc(x, y):
        return np.where(x > 0, 1, -1), np.where(y > 0, 1, -1)

    dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    x = np.random.rand(1, 3).astype(np.float32)
    y = np.random.rand(5, 3).astype(np.float32)
    helper(GradNet(Net()), (dyn, dyn), (Tensor(x), Tensor(y)), calc(x, y))


def test_stridedslice_grad():
    """
    Feature: KernelPacket
    Description: test kernelpacket with stridedslicegrad
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.stridedslice = ops.StridedSlice(0, 0, 0, 0, 0)

        def construct(self, x, y):
            shape = ops.shape(y)
            end = (10, shape[0])
            return self.stridedslice(x, (0, 0), end, (1, 1))

    def calc(x, y):
        dx = np.zeros_like(x)
        for i in range(10):
            for j in range(y.shape[0]):
                dx[i, j] = 1
        return dx, np.zeros_like(y)

    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    x = np.random.rand(32, 16).astype(np.float32)
    y = np.random.rand(10).astype(np.float32)
    helper(GradNet(Net()), (x_dyn, y_dyn), (Tensor(x), Tensor(y)), calc(x, y))
