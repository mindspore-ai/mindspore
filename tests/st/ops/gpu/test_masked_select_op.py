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
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import vmap


def maskedselect():
    x = np.array([1, 2, 3, 4]).astype(np.int32)
    mask = np.array([[[0], [1], [0], [1]], [[0], [1], [0], [1]]]).astype(np.bool)
    net = P.MaskedSelect()
    return net(Tensor(x), Tensor(mask))


def maskedselect_func():
    x = np.array([1, 2, 3, 4]).astype(np.int32)
    mask = np.array([[[0], [1], [0], [1]], [[0], [1], [0], [1]]]).astype(np.bool)
    return F.masked_select(Tensor(x), Tensor(mask))


def maskedselect_tensor():
    x = np.array([1, 2, 3, 4]).astype(np.int32)
    mask = np.array([[[0], [1], [0], [1]], [[0], [1], [0], [1]]]).astype(np.bool)
    return Tensor(x).masked_select(Tensor(mask))


def vmap_case():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.masked_select = P.MaskedSelect()

        def construct(self, a, b):
            return self.masked_select(a, b)

    class WrapNet(nn.Cell):
        def __init__(self, net, in_axes, out_axes):
            super(WrapNet, self).__init__()
            self.net = net
            self.in_axes = in_axes
            self.out_axes = out_axes

        def construct(self, a, b):
            return vmap(self.net, self.in_axes, self.out_axes)(a, b)

    # batch dimension of x is 0, and batch dimension of y is None
    # the shape of x is (2, 3), and the mask is [False, True, True], bdim is 0, so the shape of output is (2, 2)
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    y = Tensor(np.array([False, True, True], dtype=np.bool))
    output = WrapNet(Net(), (0, None), 0)(x, y)
    expect = np.array([[2, 3], [5, 6]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect)


def vmap_case_nested():
    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.masked_select = P.MaskedSelect()

        def construct(self, a, b):
            return self.masked_select(a, b)

    class WrapNet2(nn.Cell):
        def __init__(self, net, in_axes, out_axes):
            super(WrapNet2, self).__init__()
            self.net = net
            self.in_axes = in_axes
            self.out_axes = out_axes

        def construct(self, a, b):
            return vmap(vmap(self.net, self.in_axes, self.out_axes), in_axes=(-1, None))(a, b)

    # the shape of x is (2, 3, 4), the bdim is nested -1
    # the shape of mask is [[False, False], [False, True]]
    # the shape of output is (4, 3, 1)
    x = Tensor(np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                         [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], dtype=np.float32))
    y = Tensor(np.array([[False, False], [False, True]], dtype=np.bool))
    output = WrapNet2(Net2(), (-1, None), 0)(x, y)
    expect = np.array([[[13], [17], [21]], [[14], [18], [22]], [[15], [19], [23]], [[16], [20], [24]]],
                      dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_masked_select_vmap_gpu():
    """
    Feature: test MaskedSelect vmap on GPU.
    Description: inputs with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    vmap_case()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_masked_select_vmap_nested_gpu():
    """
    Feature: test MaskedSelect vmap nested on GPU.
    Description: inputs with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    vmap_case_nested()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedselect():
    """
    Feature: MaskedSelect
    Description:  test cases for MaskedSelect operator.
    Expectation: the result match expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    y = maskedselect()
    expect = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    assert (y.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedselect_func():
    """
    Feature: MaskedSelect functional interface
    Description:  test cases for MaskedSelect operator.
    Expectation: the result match expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    y = maskedselect_func()
    expect = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    assert (y.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedselect_tensor():
    """
    Feature: MaskedSelect tensor interface
    Description:  test cases for MaskedSelect operator.
    Expectation: the result match expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    y = maskedselect_tensor()
    expect = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    assert (y.asnumpy() == expect).all()
