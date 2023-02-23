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

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F


class NetArgmaxWithValue(nn.Cell):

    def __init__(self):
        super(NetArgmaxWithValue, self).__init__()
        axis = 0
        self.argmax1 = P.ArgMaxWithValue(axis)

    def construct(self, x):
        return (self.argmax1(x), F.max(x, axis=-1), F.max(x))


class NetArgmaxWithValueBig(nn.Cell):

    def __init__(self, axis=0):
        super(NetArgmaxWithValueBig, self).__init__()
        self.argmax = P.ArgMaxWithValue(axis)

    def construct(self, x):
        return self.argmax(x)


def argmaxwithvalue_base(data_type):
    x = Tensor(
        np.array([[1., 20., 5.], [67., 8., 9.], [130., 24., 15.],
                  [0.3, -0.4, -15.]]).astype(data_type))
    expect1 = np.array([2, 2, 2]).astype(data_type)
    expect11 = np.array([130, 24, 15]).astype(data_type)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    argmax = NetArgmaxWithValue()
    output = argmax(x)
    assert (output[0][0].asnumpy() == expect1).all()
    assert (output[0][1].asnumpy() == expect11).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    argmax = NetArgmaxWithValue()
    output = argmax(x)
    assert (output[0][0].asnumpy() == expect1).all()
    assert (output[0][1].asnumpy() == expect11).all()


def argmaxwithvalue_3d(data_type, shape_x):
    np.random.seed(2)
    x_np = np.random.random(shape_x).astype(data_type)
    x = Tensor(x_np)

    argmax = NetArgmaxWithValueBig(0)
    output = argmax(x)
    expect1 = np.argmax(x_np, axis=0)
    expect2 = np.maximum.reduce(x_np, 0)
    assert (output[0].asnumpy() == expect1).all()
    assert (output[1].asnumpy() == expect2).all()

    argmax = NetArgmaxWithValueBig(1)
    output = argmax(x)
    expect1 = np.argmax(x_np, axis=1)
    expect2 = np.maximum.reduce(x_np, 1)
    assert (output[0].asnumpy() == expect1).all()
    assert (output[1].asnumpy() == expect2).all()

    argmax = NetArgmaxWithValueBig(2)
    output = argmax(x)
    expect1 = np.argmax(x_np, axis=2)
    expect2 = np.maximum.reduce(x_np, 2)
    assert (output[0].asnumpy() == expect1).all()
    assert (output[1].asnumpy() == expect2).all()


def dyn_case():
    net = NetArgmaxWithValueBig()

    x_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    net.set_inputs(x_dyn)

    x = Tensor(
        np.array([[1., 20., 5.], [67., 8., 9.], [130., 24., 15.],
                  [-0.5, 25, 100]]).astype(np.float32))
    out = net(x)

    expect_shape = (3,)
    for i in range(2):
        assert out[i].asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argmaxwithvalue_dyn():
    """
    Feature: test ArgmaxWithValue dynamic shape in cpu.
    Description: inputs is dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    dyn_case()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argmaxwithvalue_base_float32():
    argmaxwithvalue_base(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argmaxwithvalue_base_float16():
    argmaxwithvalue_base(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argmaxwithvalue_3d_float32():
    shape_x = (2, 32, 256)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    argmaxwithvalue_3d(np.float32, shape_x)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    argmaxwithvalue_3d(np.float32, shape_x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argmaxwithvalue_3d_float16():
    shape_x = (2, 64, 128)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    argmaxwithvalue_3d(np.float16, shape_x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_argmaxwithvalue_3d_big_float32():
    shape_x = (128, 1024, 1)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    argmaxwithvalue_3d(np.float32, shape_x)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    argmaxwithvalue_3d(np.float32, shape_x)
