# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.common.api import ms_function
import numpy as np
import mindspore.context as context
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

context.set_context(device_target='GPU')


class ConcatV32(nn.Cell):
    def __init__(self):
        super(ConcatV32, self).__init__()

        self.cat = P.Concat(axis=2)
        self.x1 = Parameter(initializer(
            Tensor(np.arange(2 * 2 * 1).reshape(2, 2, 1).astype(np.float32)), [2, 2, 1]), name='x1')
        self.x2 = Parameter(initializer(
            Tensor(np.arange(2 * 2 * 2).reshape(2, 2, 2).astype(np.float32)), [2, 2, 2]), name='x2')

    @ms_function
    def construct(self):
        return self.cat((self.x1, self.x2))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis32():
    cat = ConcatV32()
    output = cat()
    expect = [[[0., 0., 1.],
               [1., 2., 3.]],
              [[2., 4., 5.],
               [3., 6., 7.]]]
    print(output)
    assert (output.asnumpy() == expect).all()


class ConcatV43(nn.Cell):
    def __init__(self):
        super(ConcatV43, self).__init__()

        self.cat = P.Concat(axis=3)
        self.x1 = Parameter(initializer(
            Tensor(np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2).astype(np.float32)), [2, 2, 2, 2]), name='x1')
        self.x2 = Parameter(initializer(
            Tensor(np.arange(2 * 2 * 2 * 3).reshape(2, 2, 2, 3).astype(np.float32)), [2, 2, 2, 3]), name='x2')

    @ms_function
    def construct(self):
        return self.cat((self.x1, self.x2))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis43():
    cat = ConcatV43()
    output = cat()
    expect = [[[[0., 1., 0., 1., 2.],
                [2., 3., 3., 4., 5.]],
               [[4., 5., 6., 7., 8.],
                [6., 7., 9., 10., 11.]]],
              [[[8., 9., 12., 13., 14.],
                [10., 11., 15., 16., 17.]],
               [[12., 13., 18., 19., 20.],
                [14., 15., 21., 22., 23.]]]]
    assert (output.asnumpy() == expect).all()
    print(output)


class ConcatV21(nn.Cell):
    def __init__(self):
        super(ConcatV21, self).__init__()

        self.cat = P.Concat(axis=1)
        self.x1 = Parameter(initializer(
            Tensor(np.arange(2 * 2).reshape(2, 2).astype(np.float32)), [2, 2]), name='x1')
        self.x2 = Parameter(initializer(
            Tensor(np.arange(2 * 3).reshape(2, 3).astype(np.float32)), [2, 3]), name='x2')

    @ms_function
    def construct(self):
        return self.cat((self.x1, self.x2))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_axis21():
    cat = ConcatV21()
    output = cat()
    expect = [[0., 1., 0., 1., 2.],
              [2., 3., 3., 4., 5.]]
    assert (output.asnumpy() == expect).all()
    print(output)


class Concat3INet(nn.Cell):
    def __init__(self):
        super(Concat3INet, self).__init__()
        self.cat = P.Concat(axis=1)

    def construct(self, x1, x2, x3):
        return self.cat((x1, x2, x3))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_3i():
    cat = Concat3INet()

    x1_np = np.random.randn(32, 4, 224, 224).astype(np.float32)
    x2_np = np.random.randn(32, 8, 224, 224).astype(np.float32)
    x3_np = np.random.randn(32, 10, 224, 224).astype(np.float32)
    output_np = np.concatenate((x1_np, x2_np, x3_np), axis=1)

    x1_ms = Tensor(x1_np)
    x2_ms = Tensor(x2_np)
    x3_ms = Tensor(x3_np)
    output_ms = cat(x1_ms, x2_ms, x3_ms)

    error = np.ones(shape=output_np.shape) * 10e-6
    diff = output_ms.asnumpy() - output_np
    assert np.all(diff < error)


class Concat4INet(nn.Cell):
    def __init__(self):
        super(Concat4INet, self).__init__()
        self.cat = P.Concat(axis=1)

    def construct(self, x1, x2, x3, x4):
        return self.cat((x1, x2, x3, x4))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_concat_4i():
    cat = Concat4INet()

    x1_np = np.random.randn(32, 4, 224, 224).astype(np.float32)
    x2_np = np.random.randn(32, 8, 224, 224).astype(np.float32)
    x3_np = np.random.randn(32, 10, 224, 224).astype(np.float32)
    x4_np = np.random.randn(32, 5, 224, 224).astype(np.float32)
    output_np = np.concatenate((x1_np, x2_np, x3_np, x4_np), axis=1)

    x1_ms = Tensor(x1_np)
    x2_ms = Tensor(x2_np)
    x3_ms = Tensor(x3_np)
    x4_ms = Tensor(x4_np)
    output_ms = cat(x1_ms, x2_ms, x3_ms, x4_ms)

    error = np.ones(shape=output_np.shape) * 10e-6
    diff = output_ms.asnumpy() - output_np
    assert np.all(diff < error)
