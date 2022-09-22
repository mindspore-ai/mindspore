# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C


def maskedselect():
    x = np.array([1, 2, 3, 4]).astype(np.int32)
    mask = np.array([[[0], [1], [0], [1]], [[0], [1], [0], [1]]]).astype(np.bool)
    net = P.MaskedSelect()
    return net(Tensor(x), Tensor(mask))


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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maskedselect():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    y = maskedselect()
    expect = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    assert (y.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maskedselect_dynamic_shape():
    """
    Feature: test MaskedSelect dynamic shape on CPU
    Description: the shape of input is dynamic
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    y = maskedselect_dynamic_shape()
    expect = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    assert (y.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maskedselect_bool_type():
    """
    Feature: test MaskedSelect bool type on CPU
    Description: the type of input is bool
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = np.array([0, 0, 1, 1]).astype(np.bool)
    mask = np.array([1, 0, 1, 0]).astype(np.bool)
    y = maskedselect_for_type(x, mask)
    expect = [False, True]
    assert (y.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maskedselect_complex64_type():
    """
    Feature: test MaskedSelect complex64 type on CPU
    Description: the type of input is complex64
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = np.array([1+2j, 2+3j, 3+4j, 4+5j]).astype(np.complex64)
    mask = np.array([1, 0, 1, 0]).astype(np.bool)
    y = maskedselect_for_type(x, mask)
    expect = np.array([1+2j, 3+4j]).astype(np.complex64)
    assert (y.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maskedselect_complex128_type():
    """
    Feature: test MaskedSelect complex128 type on CPU.
    Description: the type of input is complex128
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = np.array([1+2j, 2+3j, 3+4j, 4+5j]).astype(np.complex128)
    mask = np.array([1, 0, 1, 0]).astype(np.bool)
    y = maskedselect_for_type(x, mask)
    expect = np.array([1+2j, 3+4j]).astype(np.complex128)
    assert (y.asnumpy() == expect).all()


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, x, mask, grad):
        gout = self.grad(self.network)(x, mask, grad)
        return gout


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = P.MaskedSelect()

    def construct(self, x, mask):
        return self.op(x, mask)


def masked_select_grad(data_type):
    x = np.array([1, 2, 3, 4]).astype(data_type)
    mask = np.array([[0], [1], [0], [1]]).astype(np.bool)
    dy = np.array([i for i in range(8)]).astype(data_type)
    grad = Grad(Net())
    return grad(Tensor(x), Tensor(mask), Tensor(dy))[0]


def masked_select_grad_dynamic_shape():
    x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
    mask = Tensor(np.array([[0], [1], [0], [1]]).astype(np.bool))
    dy = Tensor(np.array([i for i in range(8)]).astype(np.int32))
    x_dynamic_shape = Tensor(shape=[None], dtype=mindspore.int32)
    grad = Grad(Net())
    grad.set_inputs(x_dynamic_shape, mask, dy)
    return grad(x, mask, dy)[0]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_masked_select_grad():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dx = masked_select_grad(np.int32)
    expect = [4, 6, 8, 10]
    assert (dx.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_masked_select_grad_float64():
    """
    Feature: test MaskedSelectGrad complex64 type on CPU
    Description: the type of input is float64
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dx = masked_select_grad(np.float64)
    expect = [4, 6, 8, 10]
    assert (dx.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_masked_select_grad_dynamic_shape():
    """
    Feature: test MaskedSelectGrad dynamic shape on CPU
    Description: the shape of input is dynamic
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dx = masked_select_grad_dynamic_shape()
    expect = [4, 6, 8, 10]
    assert (dx.asnumpy() == expect).all()
