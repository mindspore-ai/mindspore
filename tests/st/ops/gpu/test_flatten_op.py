# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F


class NetFlatten(nn.Cell):
    def __init__(self):
        super(NetFlatten, self).__init__()
        self.flatten = P.Flatten()

    def construct(self, x):
        return self.flatten(x)


class NetFlattenOps(nn.Cell):
    def __init__(self, order='C'):
        super(NetFlattenOps, self).__init__()
        self.order = order

    def construct(self, x, start_dim=1, end_dim=-1):
        return ops.flatten(x, self.order, start_dim=start_dim, end_dim=end_dim)


class NetFlattenTensor(nn.Cell):
    def construct(self, x, start_dim, end_dim):
        return x.flatten(), x.flatten(start_dim=start_dim, end_dim=end_dim)


class NetAllFlatten(nn.Cell):
    def __init__(self):
        super(NetAllFlatten, self).__init__()
        self.flatten = P.Flatten()

    def construct(self, x):
        loop_count = 4
        while loop_count > 0:
            x = self.flatten(x)
            loop_count = loop_count - 1
        return x


class NetFirstFlatten(nn.Cell):
    def __init__(self):
        super(NetFirstFlatten, self).__init__()
        self.flatten = P.Flatten()
        self.relu = P.ReLU()

    def construct(self, x):
        loop_count = 4
        while loop_count > 0:
            x = self.flatten(x)
            loop_count = loop_count - 1
        x = self.relu(x)
        return x


class NetLastFlatten(nn.Cell):
    def __init__(self):
        super(NetLastFlatten, self).__init__()
        self.flatten = P.Flatten()
        self.relu = P.ReLU()

    def construct(self, x):
        loop_count = 4
        x = self.relu(x)
        while loop_count > 0:
            x = self.flatten(x)
            loop_count = loop_count - 1
        return x


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_flatten():
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    expect = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    flatten = NetFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    flatten = NetFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_all_flatten():
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    expect = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    flatten = NetAllFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    flatten = NetAllFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_first_flatten():
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    expect = np.array([[0, 0.3, 3.6], [0.4, 0.5, 0]]).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    flatten = NetFirstFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    flatten = NetFirstFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_last_flatten():
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    expect = np.array([[0, 0.3, 3.6], [0.4, 0.5, 0]]).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    flatten = NetLastFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    flatten = NetLastFlatten()
    output = flatten(x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_flatten_tensor_interface():
    """
    Feature: test_flatten_tensor_interface.
    Description: test cases for tensor interface
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    in_np = np.random.randn(1, 16, 3, 1).astype(np.float32)
    in_tensor = Tensor(in_np)

    output_ms = in_tensor.flatten()
    output_np = in_np.flatten()

    np.testing.assert_allclose(output_ms.asnumpy(), output_np, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_flatten_functional_interface():
    """
    Feature: test_flatten_functional_interface.
    Description: test cases for functional interface.
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    in_np = np.random.randn(1, 16, 3, 1).astype(np.float32)
    in_tensor = Tensor(in_np)

    output_ms = F.flatten(in_tensor)
    output_np = np.reshape(in_np, (1, 48))

    np.testing.assert_allclose(output_ms.asnumpy(), output_np, rtol=1e-3)


def flatten_graph(x):
    return P.Flatten()(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_flatten_vmap():
    """
    Feature: test flatten vmap.
    Description: test cases for vmap.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    np.random.seed(0)
    in_np = np.random.rand(2, 3, 4, 5).astype(np.float32)
    output_np = np.reshape(in_np, (2, 3, 20))

    in_tensor = Tensor(in_np)
    vmap_round_net = ops.vmap(flatten_graph)
    output = vmap_round_net(in_tensor)
    np.testing.assert_allclose(output.asnumpy(), output_np, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                   np.uint32, np.uint64, np.float16, np.float32, np.float64,
                                   np.bool, np.complex64, np.complex128])
def test_flatten_op_dtype(mode, dtype):
    """
    Feature: gpu Flatten ops.
    Description: test flatten with the different types.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="GPU")

    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(dtype))
    expect = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(dtype)

    net = NetFlatten()
    out = net(x)

    assert np.allclose(expect, out.asnumpy())


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flatten_op_nn(mode):
    """
    Feature: gpu Flatten ops.
    Description: test flatten with nn interface.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="GPU")

    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    expect = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32)

    net = nn.Flatten()
    out = net(x)

    assert np.allclose(expect, out.asnumpy())


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_flatten(mode):
    """
    Feature: gpu Flatten ops.
    Description: test flatten with specified dimension.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="GPU")

    net = NetFlattenOps()
    x = Tensor(np.ones(shape=[1, 2, 3, 4]), mstype.int32)
    assert net(x).shape == (1, 24)
    assert net(x, start_dim=0).shape == (24,)
    assert net(x, start_dim=1).shape == (1, 24)
    assert net(x, start_dim=2).shape == (1, 2, 12)
    assert net(x, start_dim=1, end_dim=-1).shape == (1, 24)
    assert net(x, start_dim=1, end_dim=2).shape == (1, 6, 4)
    assert net(x, start_dim=2, end_dim=-2).shape == (1, 2, 3, 4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_nn_flatten(mode):
    """
    Feature: Flatten ops.
    Description: test nn.Flatten.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.ones(shape=[1, 2, 3, 4]), mstype.int32)
    out1 = nn.Flatten()(x)
    assert out1.shape == (1, 24)
    out2 = nn.Flatten(0, -1)(x)
    assert out2.shape == (24,)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_tensor_flatten(mode):
    """
    Feature: Flatten ops.
    Description: test tensor.flatten.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="GPU")

    net = NetFlattenTensor()
    x = Tensor(np.ones(shape=[1, 2, 3, 4]), mstype.int32)
    out1, out2 = net(x, 2, -1)
    assert out1.shape == (24,)
    assert out2.shape == (1, 2, 12)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flatten_order(mode):
    """
    Feature: Flatten ops.
    Description: test flatten with order argument.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="GPU")

    x = Tensor([[1, 2], [3, 4]], mstype.int32)
    net_c = NetFlattenOps('C')
    out_c = net_c(x, start_dim=0, end_dim=-1)
    net_f = NetFlattenOps('F')
    out_f = net_f(x, start_dim=0, end_dim=-1)
    assert np.all(out_c.asnumpy() == [1, 2, 3, 4])
    assert np.all(out_f.asnumpy() == [1, 3, 2, 4])


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flatten_single_element(mode):
    """
    Feature: gpu Flatten ops.
    Description: test flatten with single element.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="GPU")

    x = Tensor(3, mstype.int32)
    net1 = NetFlattenOps()
    out1 = net1(x)
    assert np.all(out1.asnumpy() == [3])

    y = Tensor([1, 2, 3], mstype.int32)
    net2 = NetFlattenOps()
    out2 = net2(y)
    assert np.all(out2.asnumpy() == y.asnumpy()) and out2.shape == (3,)

    with pytest.raises(ValueError):
        NetFlattenOps()(y, start_dim=2)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_ops_flatten_dynamic_shape():
    """
    Feature: Flatten ops.
    Description: test flatten with dynamic shape.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    net = NetFlattenOps()
    x = Tensor(np.ones(shape=[1, 2, 3, 4]), mstype.int32)
    x_dyn = Tensor(shape=[None for _ in x.shape], dtype=x.dtype)
    start_dim = 0
    end_dim = -1
    net.set_inputs(x_dyn, start_dim, end_dim)
    out = net(x, start_dim, end_dim)
    print(out.shape)


if __name__ == "__main__":
    test_flatten_tensor_interface()
    test_flatten_functional_interface()
    test_flatten_vmap()
