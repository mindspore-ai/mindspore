# Copyright 2019-2013 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, ops
from mindspore.ops import operations as P


class FlattenNet(nn.Cell):
    def __init__(self):
        super(FlattenNet, self).__init__()
        self.flatten = P.Flatten()

    def construct(self, tensor):
        return self.flatten(tensor)


class FlattenFuncNet(nn.Cell):
    def __init__(self, order='C'):
        super(FlattenFuncNet, self).__init__()
        self.order = order

    def construct(self, x, start_dim=1, end_dim=-1):
        return ops.flatten(x, self.order, start_dim=start_dim, end_dim=end_dim)


class FlattenTensorNet(nn.Cell):
    def construct(self, x, start_dim, end_dim):
        return x.flatten(), x.flatten(start_dim=start_dim, end_dim=end_dim)


def flatten_net(nptype):
    x = np.random.randn(1, 16, 1, 1).astype(nptype)
    net = FlattenNet()
    output = net(Tensor(x))
    print(output.asnumpy())
    assert np.all(output.asnumpy() == x.flatten())


def flatten_net_int8():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.int8)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.int8)


def flatten_net_uint8():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.uint8)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.uint8)


def flatten_net_int16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.int16)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.int16)


def flatten_net_uint16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.uint16)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.uint16)


def flatten_net_int32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.int32)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.int32)


def flatten_net_uint32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.uint32)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.uint32)


def flatten_net_int64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.int64)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.int64)


def flatten_net_uint64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.uint64)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.uint64)


def flatten_net_float16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.float16)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.float16)


def flatten_net_float32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net(np.float32)


def flatten_net_dynamic(np_type, ms_type):
    x = np.random.randn(1, 16, 3, 1).astype(np_type)
    x_dy = Tensor(shape=(1, None, 3, 1), dtype=ms_type)
    net = FlattenNet()
    net.set_inputs(x_dy)
    output = net(Tensor(x))
    print(output.asnumpy())
    assert np.all(output.asnumpy() == x.flatten())


def flatten_net_dynamic_float16():
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net_dynamic(np.float16, mindspore.float16)

    # pynative mode
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net_dynamic(np.float16, mindspore.float16)


def flatten_net_dynamic_float32():
    # graph mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    flatten_net_dynamic(np.float32, mindspore.float32)

    # pynative mode
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    flatten_net_dynamic(np.float32, mindspore.float32)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_flatten(mode):
    """
    Feature: Flatten ops.
    Description: test flatten with specified dimension.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="Ascend")

    net = FlattenFuncNet()
    x = Tensor(np.ones(shape=[1, 2, 3, 4]), mstype.int32)
    assert net(x).shape == (1, 24)
    assert net(x, start_dim=0).shape == (24,)
    assert net(x, start_dim=1).shape == (1, 24)
    assert net(x, start_dim=2).shape == (1, 2, 12)
    assert net(x, start_dim=1, end_dim=-1).shape == (1, 24)
    assert net(x, start_dim=1, end_dim=2).shape == (1, 6, 4)
    assert net(x, start_dim=2, end_dim=-2).shape == (1, 2, 3, 4)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_nn_flatten(mode):
    """
    Feature: Flatten ops.
    Description: test nn.Flatten.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="Ascend")
    x = Tensor(np.ones(shape=[1, 2, 3, 4]), mstype.int32)
    out1 = nn.Flatten()(x)
    assert out1.shape == (1, 24)
    out2 = nn.Flatten(0, -1)(x)
    assert out2.shape == (24,)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_tensor_flatten(mode):
    """
    Feature: Flatten ops.
    Description: test tensor.flatten.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="Ascend")

    net = FlattenTensorNet()
    x = Tensor(np.ones(shape=[1, 2, 3, 4]), mstype.int32)
    out1, out2 = net(x, 2, -1)
    assert out1.shape == (24,)
    assert out2.shape == (1, 2, 12)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_flatten_order(mode):
    """
    Feature: Flatten ops.
    Description: test flatten with order argument.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="Ascend")

    x = Tensor([[1, 2], [3, 4]], mstype.int32)
    net_c = FlattenFuncNet('C')
    out_c = net_c(x, start_dim=0, end_dim=-1)
    net_f = FlattenFuncNet('F')
    out_f = net_f(x, start_dim=0, end_dim=-1)
    assert np.all(out_c.asnumpy() == [1, 2, 3, 4])
    assert np.all(out_f.asnumpy() == [1, 3, 2, 4])


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_flatten_single_element(mode):
    """
    Feature: Flatten ops.
    Description: test flatten with single element.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="Ascend")

    x = Tensor(3, mstype.int32)
    net1 = FlattenFuncNet()
    out1 = net1(x)
    assert np.all(out1.asnumpy() == [3])

    y = Tensor([1, 2, 3], mstype.int32)
    net2 = FlattenFuncNet()
    out2 = net2(y)
    assert np.all(out2.asnumpy() == y.asnumpy()) and out2.shape == (3,)

    with pytest.raises(ValueError):
        FlattenFuncNet()(y, start_dim=2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ops_flatten_dynamic_shape():
    """
    Feature: Flatten ops.
    Description: test flatten with dynamic shape.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    net = FlattenFuncNet()
    x = Tensor(np.ones(shape=[1, 2, 3, 4]), mstype.int32)
    x_dyn = Tensor(shape=[None for _ in x.shape], dtype=x.dtype)
    start_dim = 0
    end_dim = -1
    net.set_inputs(x_dyn, start_dim, end_dim)
    out = net(x, start_dim, end_dim)
    print(out.shape)


if __name__ == "__main__":
    flatten_net_dynamic_float16()
    flatten_net_dynamic_float32()
