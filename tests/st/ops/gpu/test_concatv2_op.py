# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P


class ConcatV32(nn.Cell):
    def __init__(self, nptype):
        super(ConcatV32, self).__init__()

        self.cat = P.Concat(axis=2)
        self.x1 = Parameter(initializer(
            Tensor(np.arange(2 * 2 * 1).reshape(2, 2, 1).astype(nptype)), [2, 2, 1]), name='x1')
        self.x2 = Parameter(initializer(
            Tensor(np.arange(2 * 2 * 2).reshape(2, 2, 2).astype(nptype)), [2, 2, 2]), name='x2')

    @ms_function
    def construct(self):
        return self.cat((self.x1, self.x2))


def axis32(nptype):
    context.set_context(device_target='GPU')

    cat = ConcatV32(nptype)
    output = cat()
    expect = np.array([[[0., 0., 1.],
                        [1., 2., 3.]],
                       [[2., 4., 5.],
                        [3., 6., 7.]]]).astype(nptype)
    assert (output.asnumpy() == expect).all()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis32_float64():
    axis32(np.float64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis32_float32():
    axis32(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis32_int16():
    axis32(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis32_uint8():
    axis32(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis32_bool():
    axis32(np.bool)


class ConcatV43(nn.Cell):
    def __init__(self, nptype):
        super(ConcatV43, self).__init__()

        self.cat = P.Concat(axis=3)
        self.x1 = Parameter(initializer(
            Tensor(np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2).astype(nptype)), [2, 2, 2, 2]), name='x1')
        self.x2 = Parameter(initializer(
            Tensor(np.arange(2 * 2 * 2 * 3).reshape(2, 2, 2, 3).astype(nptype)), [2, 2, 2, 3]), name='x2')

    @ms_function
    def construct(self):
        return self.cat((self.x1, self.x2))


def axis43(nptype):
    context.set_context(device_target='GPU')

    cat = ConcatV43(nptype)
    output = cat()
    expect = np.array([[[[0., 1., 0., 1., 2.],
                         [2., 3., 3., 4., 5.]],
                        [[4., 5., 6., 7., 8.],
                         [6., 7., 9., 10., 11.]]],
                       [[[8., 9., 12., 13., 14.],
                         [10., 11., 15., 16., 17.]],
                        [[12., 13., 18., 19., 20.],
                         [14., 15., 21., 22., 23.]]]]).astype(nptype)
    assert (output.asnumpy() == expect).all()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis43_float64():
    axis43(np.float64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis43_float32():
    axis43(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis43_int16():
    axis43(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis43_uint8():
    axis43(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis43_bool():
    axis43(np.bool)


class ConcatV21(nn.Cell):
    def __init__(self, nptype):
        super(ConcatV21, self).__init__()

        self.cat = P.Concat(axis=1)
        self.x1 = Parameter(initializer(
            Tensor(np.arange(2 * 2).reshape(2, 2).astype(nptype)), [2, 2]), name='x1')
        self.x2 = Parameter(initializer(
            Tensor(np.arange(2 * 3).reshape(2, 3).astype(nptype)), [2, 3]), name='x2')

    @ms_function
    def construct(self):
        return self.cat((self.x1, self.x2))


def axis21(nptype):
    cat = ConcatV21(nptype)
    output = cat()
    expect = np.array([[0., 1., 0., 1., 2.],
                       [2., 3., 3., 4., 5.]]).astype(nptype)
    assert (output.asnumpy() == expect).all()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis21_float64():
    axis21(np.float64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis21_float32():
    axis21(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis21_int16():
    axis21(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis21_uint8():
    axis21(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis21_bool():
    axis21(np.bool)


class Concat3INet(nn.Cell):
    def __init__(self):
        super(Concat3INet, self).__init__()
        self.cat = P.Concat(axis=1)

    def construct(self, x1, x2, x3):
        return self.cat((x1, x2, x3))


def concat_3i(nptype):
    cat = Concat3INet()

    x1_np = np.random.randn(32, 4, 224, 224).astype(nptype)
    x2_np = np.random.randn(32, 8, 224, 224).astype(nptype)
    x3_np = np.random.randn(32, 10, 224, 224).astype(nptype)
    output_np = np.concatenate((x1_np, x2_np, x3_np), axis=1)

    x1_ms = Tensor(x1_np)
    x2_ms = Tensor(x2_np)
    x3_ms = Tensor(x3_np)
    output_ms = cat(x1_ms, x2_ms, x3_ms)

    error = np.ones(shape=output_np.shape) * 10e-6
    diff = output_ms.asnumpy() - output_np
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_3i_float64():
    concat_3i(np.float64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_3i_float32():
    concat_3i(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_3i_int16():
    concat_3i(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_3i_uint8():
    concat_3i(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_3i_bool():
    cat = Concat3INet()

    x1_np = np.random.choice([True, False], (32, 4, 224, 224)).astype(np.bool)
    x2_np = np.random.choice([True, False], (32, 8, 224, 224)).astype(np.bool)
    x3_np = np.random.choice([True, False], (32, 10, 224, 224)).astype(np.bool)
    output_np = np.concatenate((x1_np, x2_np, x3_np), axis=1)

    x1_ms = Tensor(x1_np)
    x2_ms = Tensor(x2_np)
    x3_ms = Tensor(x3_np)
    output_ms = cat(x1_ms, x2_ms, x3_ms)

    assert (output_ms.asnumpy() == output_np).all()


class Concat4INet(nn.Cell):
    def __init__(self):
        super(Concat4INet, self).__init__()
        self.cat = P.Concat(axis=1)

    def construct(self, x1, x2, x3, x4):
        return self.cat((x1, x2, x3, x4))


def concat_4i(nptype):
    cat = Concat4INet()

    x1_np = np.random.randn(32, 4, 224, 224).astype(nptype)
    x2_np = np.random.randn(32, 8, 224, 224).astype(nptype)
    x3_np = np.random.randn(32, 10, 224, 224).astype(nptype)
    x4_np = np.random.randn(32, 5, 224, 224).astype(nptype)
    output_np = np.concatenate((x1_np, x2_np, x3_np, x4_np), axis=1)

    x1_ms = Tensor(x1_np)
    x2_ms = Tensor(x2_np)
    x3_ms = Tensor(x3_np)
    x4_ms = Tensor(x4_np)
    output_ms = cat(x1_ms, x2_ms, x3_ms, x4_ms)

    error = np.ones(shape=output_np.shape) * 10e-6
    diff = output_ms.asnumpy() - output_np
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_4i_float64():
    concat_4i(np.float64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_4i_float32():
    concat_4i(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_4i_int16():
    concat_4i(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_4i_uint8():
    concat_4i(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_4i_bool():
    cat = Concat4INet()

    x1_np = np.random.choice([True, False], (32, 4, 224, 224)).astype(np.bool)
    x2_np = np.random.choice([True, False], (32, 8, 224, 224)).astype(np.bool)
    x3_np = np.random.choice([True, False], (32, 10, 224, 224)).astype(np.bool)
    x4_np = np.random.choice([True, False], (32, 5, 224, 224)).astype(np.bool)
    output_np = np.concatenate((x1_np, x2_np, x3_np, x4_np), axis=1)

    x1_ms = Tensor(x1_np)
    x2_ms = Tensor(x2_np)
    x3_ms = Tensor(x3_np)
    x4_ms = Tensor(x4_np)
    output_ms = cat(x1_ms, x2_ms, x3_ms, x4_ms)

    assert (output_ms.asnumpy() == output_np).all()
