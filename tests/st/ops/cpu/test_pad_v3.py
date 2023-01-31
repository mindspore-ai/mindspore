# Copyright 2022-2023 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.context as context

from mindspore import Tensor
from mindspore.ops.operations import nn_ops


class NetDynamic(nn.Cell):
    def __init__(self, mode, paddings_contiguous=True):
        super(NetDynamic, self).__init__()
        self.ops = nn_ops.PadV3(mode, paddings_contiguous)
        self.mode = mode

    def construct(self, x, paddings, value=0):
        if self.mode == "constant":
            out = self.ops(x, paddings, value)
        else:
            out = self.ops(x, paddings)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_padv3_circular_dynamic_shape_3d():
    """
    Feature: test padv3 x and padding dynamic shape
    Description: test padv3 dynamic shape
    Expectation: Success
    """
    context.set_context(device_target="CPU", save_graphs=False)
    x = Tensor(np.arange(9).reshape(1, 3, 3).astype(np.float32))
    padding = Tensor((1, 2), dtype=mindspore.int64)

    net = NetDynamic('circular')

    x_dyn = Tensor(shape=(1, 3, None), dtype=x.dtype)
    padding_dyn = Tensor(shape=(None,), dtype=padding.dtype)
    net.set_inputs(x_dyn, padding_dyn)

    out = net(x, padding)
    expect = np.array([[[2, 0, 1, 2, 0, 1],
                        [5, 3, 4, 5, 3, 4],
                        [8, 6, 7, 8, 6, 7]]]).astype(np.float32)
    np.testing.assert_almost_equal(expect, out.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_padv3_circular_dynamic_shape_4d():
    """
    Feature: test padv3 x and padding dynamic shape
    Description: test padv3 dynamic shape
    Expectation: Success
    """
    context.set_context(device_target="CPU", save_graphs=False)
    x = Tensor(np.arange(9).reshape(1, 1, 3, 3).astype(np.float64))
    padding = Tensor((1, -1, 1, 2), dtype=mindspore.int32)

    net = NetDynamic('circular')

    x_dyn = Tensor(shape=(1, 1, 3, None), dtype=x.dtype)
    padding_dyn = Tensor(shape=(None,), dtype=padding.dtype)
    net.set_inputs(x_dyn, padding_dyn)

    out = net(x, padding)
    expect = np.array([[[[7, 6, 7], [1, 0, 1], [4, 3, 4],
                         [7, 6, 7], [1, 0, 1], [4, 3, 4]]]]).astype(np.float64)
    np.testing.assert_almost_equal(expect, out.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_padv3_circular_dynamic_shape_5d():
    """
    Feature: test padv3 x and padding dynamic shape
    Description: test padv3 dynamic shape
    Expectation: Success
    """
    context.set_context(device_target="CPU", save_graphs=False)
    x = Tensor(np.arange(18).reshape(1, 1, 2, 3, 3).astype(np.float64))
    padding = Tensor((0, 1, 1, -1, 0, -1), dtype=mindspore.int32)

    net = NetDynamic('circular')

    x_dyn = Tensor(shape=(1, 1, None, 3, None), dtype=x.dtype)
    padding_dyn = Tensor(shape=(None,), dtype=padding.dtype)
    net.set_inputs(x_dyn, padding_dyn)

    out = net(x, padding)
    expect = np.array([[[[[3, 4, 5, 3,],
                          [0, 1, 2, 0,],
                          [3, 4, 5, 3,]]]]]).astype(np.float64)
    np.testing.assert_almost_equal(expect, out.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_padv3_x_padding_dynamic_shape():
    """
    Feature: test padv3 x and padding dynamic shape
    Description: test padv3 dynamic shape
    Expectation: Success
    """
    mode = 'constant'
    context.set_context(device_target="CPU", save_graphs=False)
    x = Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    padding = Tensor((1, 1, 2, 2), dtype=mindspore.int64)
    value = Tensor(1.5)

    net = NetDynamic(mode, True)

    x_dyn = Tensor(shape=(3, None), dtype=mindspore.float32)
    padding_dyn = Tensor(shape=(None,), dtype=padding.dtype)
    net.set_inputs(x_dyn, padding_dyn, value)

    out = net(x, padding, value)
    expect = np.array([[1.5, 1.5, 1.5, 1.5, 1.5],
                       [1.5, 1.5, 1.5, 1.5, 1.5],
                       [1.5, 0., 1., 2., 1.5],
                       [1.5, 3., 4., 5., 1.5],
                       [1.5, 6., 7., 8., 1.5],
                       [1.5, 1.5, 1.5, 1.5, 1.5],
                       [1.5, 1.5, 1.5, 1.5, 1.5]]).astype(np.float32)
    np.testing.assert_almost_equal(expect, out.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_padv3_x_dynamic_shape():
    """
    Feature: test padv3 x dynamic shape
    Description: test padv3 dynamic shape
    Expectation: Success
    """
    mode = 'constant'
    context.set_context(device_target="CPU", save_graphs=False)
    x = Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    padding = Tensor((1, 1, 2, 2), dtype=mindspore.int64)
    value = Tensor(1.5)

    net = NetDynamic(mode, True)

    x_dyn = Tensor(shape=(None, None), dtype=mindspore.float32)
    net.set_inputs(x_dyn, padding, value)

    out = net(x, padding, value)
    expect = np.array([[1.5, 1.5, 1.5, 1.5, 1.5],
                       [1.5, 1.5, 1.5, 1.5, 1.5],
                       [1.5, 0., 1., 2., 1.5],
                       [1.5, 3., 4., 5., 1.5],
                       [1.5, 6., 7., 8., 1.5],
                       [1.5, 1.5, 1.5, 1.5, 1.5],
                       [1.5, 1.5, 1.5, 1.5, 1.5]]).astype(np.float32)
    np.testing.assert_almost_equal(expect, out.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_padv3_padding_dynamic_shape():
    """
    Feature: test padv3 padding dynamic shape
    Description: test padv3 dynamic shape
    Expectation: Success
    """
    mode = 'constant'
    context.set_context(device_target="CPU", save_graphs=False)
    x = Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    padding = Tensor((1, 1, 2, 2), dtype=mindspore.int32)
    value = Tensor(1.5)

    net = NetDynamic(mode, True)

    padding_dyn = Tensor(shape=(None,), dtype=padding.dtype)
    net.set_inputs(x, padding_dyn, value)

    out = net(x, padding, value)
    expect = np.array([[1.5, 1.5, 1.5, 1.5, 1.5],
                       [1.5, 1.5, 1.5, 1.5, 1.5],
                       [1.5, 0., 1., 2., 1.5],
                       [1.5, 3., 4., 5., 1.5],
                       [1.5, 6., 7., 8., 1.5],
                       [1.5, 1.5, 1.5, 1.5, 1.5],
                       [1.5, 1.5, 1.5, 1.5, 1.5]]).astype(np.float32)
    np.testing.assert_almost_equal(expect, out.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_padv3_padding_list():
    """
    Feature: test padv3 padding dynamic shape
    Description: test padv3 dynamic shape
    Expectation: Success
    """
    mode = 'constant'
    context.set_context(device_target="CPU", save_graphs=True)
    x = Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    padding = (1, 1, 2, 2)
    value = Tensor(1.5)
    net = NetDynamic(mode, True)
    out = net(x, padding, value)
    expect = np.array([[1.5, 1.5, 1.5, 1.5, 1.5],
                       [1.5, 1.5, 1.5, 1.5, 1.5],
                       [1.5, 0., 1., 2., 1.5],
                       [1.5, 3., 4., 5., 1.5],
                       [1.5, 6., 7., 8., 1.5],
                       [1.5, 1.5, 1.5, 1.5, 1.5],
                       [1.5, 1.5, 1.5, 1.5, 1.5]]).astype(np.float32)
    np.testing.assert_almost_equal(expect, out.asnumpy())
