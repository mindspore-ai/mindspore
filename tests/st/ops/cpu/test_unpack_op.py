# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore.ops.operations.array_ops as P
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter


class Net(nn.Cell):
    def __init__(self, nptype):
        super(Net, self).__init__()

        self.unstack = P.Unstack(axis=3)
        self.data_np = np.array([[[[[0, 0],
                                    [-2, -1]],
                                   [[0, 0],
                                    [0, 1]]],
                                  [[[0, 0],
                                    [2, 3]],
                                   [[0, 0],
                                    [4, 5]]],
                                  [[[0, 0],
                                    [6, 7]],
                                   [[0, 0],
                                    [8, 9]]]],
                                 [[[[0, 0],
                                    [10, 11]],
                                   [[0, 0],
                                    [12, 13]]],
                                  [[[0, 0],
                                    [14, 15]],
                                   [[0, 0],
                                    [16, 17]]],
                                  [[[0, 0],
                                    [18, 19]],
                                   [[0, 0],
                                    [20, 21]]]],
                                 [[[[0, 0],
                                    [22, 23]],
                                   [[0, 0],
                                    [24, 25]]],
                                  [[[0, 0],
                                    [26, 27]],
                                   [[0, 0],
                                    [28, 29]]],
                                  [[[0, 0],
                                    [30, 31]],
                                   [[0, 0],
                                    [32, 33]]]]]).astype(nptype)
        self.x1 = Parameter(initializer(Tensor(self.data_np), [3, 3, 2, 2, 2]), name='x1')

    @ms_function
    def construct(self):
        return self.unstack(self.x1)


def unpack(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    unpack_ = Net(nptype)
    output = unpack_()
    expect = (np.reshape(np.array([0] * 36).astype(nptype), (3, 3, 2, 2)),
              np.arange(-2, 34, 1).reshape(3, 3, 2, 2).astype(nptype))

    for i, exp in enumerate(expect):
        assert (output[i].asnumpy() == exp).all()


def unpack_pynative(nptype):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    x1 = np.array([[[[[0, 0],
                      [-2, -1]],
                     [[0, 0],
                      [0, 1]]],
                    [[[0, 0],
                      [2, 3]],
                     [[0, 0],
                      [4, 5]]],
                    [[[0, 0],
                      [6, 7]],
                     [[0, 0],
                      [8, 9]]]],
                   [[[[0, 0],
                      [10, 11]],
                     [[0, 0],
                      [12, 13]]],
                    [[[0, 0],
                      [14, 15]],
                     [[0, 0],
                      [16, 17]]],
                    [[[0, 0],
                      [18, 19]],
                     [[0, 0],
                      [20, 21]]]],
                   [[[[0, 0],
                      [22, 23]],
                     [[0, 0],
                      [24, 25]]],
                    [[[0, 0],
                      [26, 27]],
                     [[0, 0],
                      [28, 29]]],
                    [[[0, 0],
                      [30, 31]],
                     [[0, 0],
                      [32, 33]]]]]).astype(nptype)
    x1 = Tensor(x1)
    expect = (np.reshape(np.array([0] * 36).astype(nptype), (3, 3, 2, 2)),
              np.arange(-2, 34, 1).reshape(3, 3, 2, 2).astype(nptype))
    output = P.Unstack(axis=3)(x1)

    for i, exp in enumerate(expect):
        assert (output[i].asnumpy() == exp).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_unpack_graph_float32():
    unpack(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_unpack_graph_float16():
    unpack(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_unpack_graph_int32():
    unpack(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_unpack_graph_int16():
    unpack(np.int16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_unpack_graph_uint8():
    unpack(np.uint8)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_unpack_graph_bool():
    unpack(np.bool)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_unpack_pynative_float32():
    unpack_pynative(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_unpack_pynative_float16():
    unpack_pynative(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_unpack_pynative_int32():
    unpack_pynative(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_unpack_pynative_int16():
    unpack_pynative(np.int16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_unpack_pynative_uint8():
    unpack_pynative(np.uint8)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_unpack_pynative_bool():
    unpack_pynative(np.bool)
