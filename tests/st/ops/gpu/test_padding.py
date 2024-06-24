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
from tests.mark_utils import arg_mark
import pytest
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner


class Net(nn.Cell):
    def __init__(self, pad_dim_size):
        super(Net, self).__init__()
        self.padding = P.Padding(pad_dim_size)

    def construct(self, x):
        return self.padding(x)


class PaddingDynamicShapeNet(nn.Cell):
    def __init__(self):
        super(PaddingDynamicShapeNet, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()

    def construct(self, x, pad_dim_size=4):
        x = self.test_dynamic(x)
        return F.padding(x, pad_dim_size)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2, 1), (2, 4, 1), (3, 4, 5, 1)])
@pytest.mark.parametrize('dtype', [np.bool_, np.uint32, np.float16, np.float32, np.complex64, np.complex128])
@pytest.mark.parametrize('pad_dim_size', [2, 4, 10])
def test_padding(mode, shape, dtype, pad_dim_size):
    """
    Feature: ALL To ALL
    Description: test cases for padding
    Expectation: the result match to numpy
    """
    context.set_context(mode=mode, device_target="GPU")
    prop = 100 if np.random.random() > 0.5 else -100
    x = (np.random.randn(*shape) * prop).astype(dtype)
    padding = Net(pad_dim_size)
    output = padding(Tensor(x))
    pad_width = [(0, 0) for _ in range(len(shape) - 1)]
    pad_width.append((0, pad_dim_size - 1))
    expect = np.pad(x, tuple(pad_width), 'constant', constant_values=0)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_padding_vmap(mode):
    """
    Feature: test padding vmap feature.
    Description: test padding vmap feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.array([[[-270.0144],
                          [19.09283],
                          [43.96024],
                          [257.01694]],
                         [[-104.56876],
                          [42.85809],
                          [-123.558815],
                          [54.194077]]], dtype=np.float32))
    # Case 1
    output = F.vmap(Net(4), 0, 0)(x)
    expect_output = np.array([[[-270.0144, 0, 0, 0],
                               [19.09283, 0, 0, 0],
                               [43.96024, 0, 0, 0],
                               [257.01694, 0, 0, 0]],
                              [[-104.56876, 0, 0, 0],
                               [42.85809, 0, 0, 0],
                               [-123.558815, 0, 0, 0],
                               [54.194077, 0, 0, 0]]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 2
    output = F.vmap(Net(4), 0, 1)(x)
    expect_output = np.array([[[-270.0144, 0., 0., 0.],
                               [-104.56876, 0., 0., 0.]],
                              [[19.09283, 0., 0., 0.],
                               [42.85809, 0., 0., 0.]],
                              [[43.96024, 0., 0., 0.],
                               [-123.558815, 0., 0., 0.]],
                              [[257.01694, 0., 0., 0.],
                               [54.194077, 0., 0., 0.]]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # # Case 3
    output = F.vmap(Net(4), 1, 0)(x)
    expect_output = np.array([[[-270.0144, 0., 0., 0.],
                               [-104.56876, 0., 0., 0.]],
                              [[19.09283, 0., 0., 0.],
                               [42.85809, 0., 0., 0.]],
                              [[43.96024, 0., 0., 0.],
                               [-123.558815, 0., 0., 0.]],
                              [[257.01694, 0., 0., 0.],
                               [54.194077, 0., 0., 0.]]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_padding_dynamic_shape(mode):
    """
    Feature: test padding dynamic_shape feature.
    Description: test padding dynamic_shape feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.array([[[-270.0144],
                          [19.09283],
                          [43.96024],
                          [257.01694]],
                         [[-104.56876],
                          [42.85809],
                          [-123.558815],
                          [54.194077]]], dtype=np.float32))
    output = PaddingDynamicShapeNet()(x)
    expect_output = np.array([[[-270.0144, 0, 0, 0],
                               [19.09283, 0, 0, 0],
                               [43.96024, 0, 0, 0],
                               [257.01694, 0, 0, 0]],
                              [[-104.56876, 0, 0, 0],
                               [42.85809, 0, 0, 0],
                               [-123.558815, 0, 0, 0],
                               [54.194077, 0, 0, 0]]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)
