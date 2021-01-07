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

import pytest
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

class ConcatV10(nn.Cell):
    def __init__(self, nptype):
        super(ConcatV10, self).__init__()

        self.cat = P.Concat(axis=2)
        self.x1 = Tensor(np.array([[[0., 0., 1.],
                                    [1., 2., 3.]],
                                   [[2., 4., 5.],
                                    [3., 6., 7.]]]).astype(nptype))

    def construct(self):
        return self.cat((self.x1,))


def axis10(nptype):
    cat = ConcatV10(nptype)
    output = cat()
    expect = np.array([[[0., 0., 1.],
                        [1., 2., 3.]],
                       [[2., 4., 5.],
                        [3., 6., 7.]]]).astype(nptype)
    print(output)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis10_float32():
    axis10(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis10_int32():
    axis10(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis10_bool():
    axis10(np.bool)

class ConcatV32(nn.Cell):
    def __init__(self, nptype):
        super(ConcatV32, self).__init__()

        self.cat = P.Concat(axis=2)
        self.x1 = Tensor(np.arange(2 * 2 * 1).reshape(2, 2, 1).astype(nptype))
        self.x2 = Tensor(np.arange(2 * 2 * 2).reshape(2, 2, 2).astype(nptype))

    def construct(self):
        return self.cat((self.x1, self.x2))


def axis32(nptype):
    cat = ConcatV32(nptype)
    output = cat()
    expect = np.array([[[0., 0., 1.],
                        [1., 2., 3.]],
                       [[2., 4., 5.],
                        [3., 6., 7.]]]).astype(nptype)
    print(output)
    assert (output.asnumpy() == expect).all()

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis32_float32():
    axis32(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis32_int32():
    axis32(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis32_bool():
    axis32(np.bool)


class ConcatV43(nn.Cell):
    def __init__(self, nptype):
        super(ConcatV43, self).__init__()

        self.cat = P.Concat(axis=3)
        self.x1 = Tensor(np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2).astype(nptype))
        self.x2 = Tensor(np.arange(2 * 2 * 2 * 3).reshape(2, 2, 2, 3).astype(nptype))

    def construct(self):
        return self.cat((self.x1, self.x2))


def axis43(nptype):
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
    print(output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis43_float32():
    axis43(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis43_int32():
    axis43(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis43_bool():
    axis43(np.bool)


class ConcatV21(nn.Cell):
    def __init__(self, nptype):
        super(ConcatV21, self).__init__()

        self.cat = P.Concat(axis=1)
        self.x1 = Tensor(np.arange(2 * 2).reshape(2, 2).astype(nptype))
        self.x2 = Tensor(np.arange(2 * 3).reshape(2, 3).astype(nptype))

    def construct(self):
        return self.cat((self.x1, self.x2))


def axis21(nptype):
    cat = ConcatV21(nptype)
    output = cat()
    expect = np.array([[0., 1., 0., 1., 2.],
                       [2., 3., 3., 4., 5.]]).astype(nptype)
    assert (output.asnumpy() == expect).all()
    print(output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis21_float32():
    axis21(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis21_int32():
    axis21(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
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
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_3i_float32():
    concat_3i(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_3i_int32():
    concat_3i(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
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
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_4i_float32():
    concat_4i(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_4i_int32():
    concat_4i(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_4i_int8():
    concat_4i(np.int8)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_4i_uint64():
    concat_4i(np.uint64)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
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
