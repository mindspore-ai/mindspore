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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class NetArgminWithValue(nn.Cell):
    def __init__(self):
        super(NetArgminWithValue, self).__init__()
        axis1 = 0
        axis2 = -1
        self.argmin1 = P.ArgMinWithValue(axis1)
        self.argmin2 = P.ArgMinWithValue(axis2)
        self.argmin3 = P.ArgMinWithValue()

    def construct(self, x):
        return (self.argmin1(x), self.argmin2(x), self.argmin3(x))


class NetArgminWithValueBig(nn.Cell):
    def __init__(self, axis=0):
        super(NetArgminWithValueBig, self).__init__()
        self.argmin = P.ArgMinWithValue(axis)

    def construct(self, x):
        return self.argmin(x)


def argminwithvalue_base(data_type):
    x = Tensor(np.array([[1., 20., 5.],
                         [67., 8., 9.],
                         [130., 24., 15.],
                         [0.3, -0.4, -15.]]).astype(data_type))
    expect1 = np.array([3, 3, 3]).astype(data_type)
    expect2 = np.array([0, 1, 2, 2]).astype(data_type)
    expect11 = np.array([0.3, -0.4, -15.]).astype(data_type)
    expect22 = np.array([1., 8., 15., -15.]).astype(data_type)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    argmin = NetArgminWithValue()
    output = argmin(x)
    assert (output[0][0].asnumpy() == expect1).all()
    assert (output[0][1].asnumpy() == expect11).all()
    assert (output[1][0].asnumpy() == expect2).all()
    assert (output[1][1].asnumpy() == expect22).all()
    assert (output[2][0].asnumpy() == expect1).all()
    assert (output[2][1].asnumpy() == expect11).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    argmin = NetArgminWithValue()
    output = argmin(x)
    assert (output[0][0].asnumpy() == expect1).all()
    assert (output[0][1].asnumpy() == expect11).all()
    assert (output[1][0].asnumpy() == expect2).all()
    assert (output[1][1].asnumpy() == expect22).all()
    assert (output[2][0].asnumpy() == expect1).all()
    assert (output[2][1].asnumpy() == expect11).all()


def argminwithvalue_3d(data_type, shape_x):
    np.random.seed(2)
    x_np = np.random.random(shape_x).astype(data_type)
    x = Tensor(x_np)

    argmin = NetArgminWithValueBig(0)
    output = argmin(x)
    expect1 = np.argmin(x_np, axis=0)
    expect2 = np.minimum.reduce(x_np, 0)
    assert (output[0].asnumpy() == expect1).all()
    assert (output[1].asnumpy() == expect2).all()

    argmin = NetArgminWithValueBig(1)
    output = argmin(x)
    expect1 = np.argmin(x_np, axis=1)
    expect2 = np.minimum.reduce(x_np, 1)
    assert (output[0].asnumpy() == expect1).all()
    assert (output[1].asnumpy() == expect2).all()

    argmin = NetArgminWithValueBig(2)
    output = argmin(x)
    expect1 = np.argmin(x_np, axis=2)
    expect2 = np.minimum.reduce(x_np, 2)
    assert (output[0].asnumpy() == expect1).all()
    assert (output[1].asnumpy() == expect2).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argminwithvalue_base_float32():
    argminwithvalue_base(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argminwithvalue_base_float16():
    argminwithvalue_base(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argminwithvalue_3d_float32():
    shape_x = (2, 32, 256)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    argminwithvalue_3d(np.float32, shape_x)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    argminwithvalue_3d(np.float32, shape_x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argminwithvalue_3d_float16():
    shape_x = (2, 64, 128)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    argminwithvalue_3d(np.float16, shape_x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argminwithvalue_3d_big_float32():
    shape_x = (128, 1024, 1)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    argminwithvalue_3d(np.float32, shape_x)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    argminwithvalue_3d(np.float32, shape_x)
