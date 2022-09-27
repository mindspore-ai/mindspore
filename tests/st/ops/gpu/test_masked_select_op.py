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


def maskedselect_dynamic_shape():
    x = np.array([1, 2, 3, 4, 1, 2, 3, 4]).astype(np.int32)
    mask = np.array([[[0], [1], [0], [1]], [[0], [1], [0], [1]]]).astype(np.bool)
    net = P.MaskedSelect()
    unique = P.Unique()
    unique_out, _ = unique(Tensor(x))
    return net(unique_out, Tensor(mask))


def maskedselect_for_type(x, mask):
    net = P.MaskedSelect()
    return net(Tensor(x), Tensor(mask))


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


@pytest.mark.level1
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


@pytest.mark.level1
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


@pytest.mark.level1
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedselect_dynamic_shape():
    """
    Feature: test MaskedSelect dynamic shape on GPU
    Description: the shape of input is dynamic
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    y = maskedselect_dynamic_shape()
    expect = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    assert (y.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedselect_bool_type():
    """
    Feature: test MaskedSelect bool type on GPU
    Description: the type of input is bool
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([0, 0, 1, 1]).astype(np.bool)
    mask = np.array([1, 0, 1, 0]).astype(np.bool)
    y = maskedselect_for_type(x, mask)
    expect = [False, True]
    assert (y.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedselect_complex64_type():
    """
    Feature: test MaskedSelect complex64 type on GPU
    Description: the type of input is complex64
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([1+2j, 2+3j, 3+4j, 4+5j]).astype(np.complex64)
    mask = np.array([1, 0, 1, 0]).astype(np.bool)
    y = maskedselect_for_type(x, mask)
    expect = np.array([1+2j, 3+4j]).astype(np.complex64)
    assert (y.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedselect_complex128_type():
    """
    Feature: test MaskedSelect complex128 type on GPU.
    Description: the type of input is complex128
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([1+2j, 2+3j, 3+4j, 4+5j]).astype(np.complex128)
    mask = np.array([1, 0, 1, 0]).astype(np.bool)
    y = maskedselect_for_type(x, mask)
    expect = np.array([1+2j, 3+4j]).astype(np.complex128)
    assert (y.asnumpy() == expect).all()


@pytest.mark.level1
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


@pytest.mark.level1
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
