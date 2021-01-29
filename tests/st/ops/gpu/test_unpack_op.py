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


class UnstackNet(nn.Cell):
    def __init__(self, nptype):
        super(UnstackNet, self).__init__()

        self.unstack = P.Unstack(axis=3)
        self.data_np = np.array([[[[[0, 0],
                                    [0, 1]],
                                   [[0, 0],
                                    [2, 3]]],
                                  [[[0, 0],
                                    [4, 5]],
                                   [[0, 0],
                                    [6, 7]]]],
                                 [[[[0, 0],
                                    [8, 9]],
                                   [[0, 0],
                                    [10, 11]]],
                                  [[[0, 0],
                                    [12, 13]],
                                   [[0, 0],
                                    [14, 15]]]]]).astype(nptype)
        self.x1 = Parameter(initializer(Tensor(self.data_np), [2, 2, 2, 2, 2]), name='x1')

    @ms_function
    def construct(self):
        return self.unstack(self.x1)


def unstack(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    unstack_ = UnstackNet(nptype)
    output = unstack_()
    expect = (np.reshape(np.array([0] * 16).astype(nptype), (2, 2, 2, 2)),
              np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2).astype(nptype))

    for i, exp in enumerate(expect):
        assert (output[i].asnumpy() == exp).all()


def unstack_pynative(nptype):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x1 = np.array([[[[[0, 0],
                      [0, 1]],
                     [[0, 0],
                      [2, 3]]],
                    [[[0, 0],
                      [4, 5]],
                     [[0, 0],
                      [6, 7]]]],
                   [[[[0, 0],
                      [8, 9]],
                     [[0, 0],
                      [10, 11]]],
                    [[[0, 0],
                      [12, 13]],
                     [[0, 0],
                      [14, 15]]]]]).astype(nptype)
    x1 = Tensor(x1)
    expect = (np.reshape(np.array([0] * 16).astype(nptype), (2, 2, 2, 2)),
              np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2).astype(nptype))
    output = P.Unstack(axis=3)(x1)

    for i, exp in enumerate(expect):
        assert (output[i].asnumpy() == exp).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unstack_graph_float32():
    unstack(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unstack_graph_float16():
    unstack(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unstack_graph_int32():
    unstack(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unstack_graph_int16():
    unstack(np.int16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unstack_graph_uint8():
    unstack(np.uint8)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unstack_graph_bool():
    unstack(np.bool)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unstack_pynative_float32():
    unstack_pynative(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unstack_pynative_float16():
    unstack_pynative(np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unstack_pynative_int32():
    unstack_pynative(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unstack_pynative_int16():
    unstack_pynative(np.int16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unstack_pynative_uint8():
    unstack_pynative(np.uint8)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unstack_pynative_bool():
    unstack_pynative(np.bool)
