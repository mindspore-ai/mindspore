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


class StackNet(nn.Cell):
    def __init__(self, nptype):
        super(StackNet, self).__init__()

        self.stack = P.Stack(axis=2)
        self.data_np = np.array([0] * 16).astype(nptype)
        self.data_np = np.reshape(self.data_np, (2, 2, 2, 2))
        self.x1 = Parameter(initializer(
            Tensor(self.data_np), [2, 2, 2, 2]), name='x1')
        self.x2 = Parameter(initializer(
            Tensor(np.arange(16).reshape(2, 2, 2, 2).astype(nptype)), [2, 2, 2, 2]), name='x2')

    @ms_function
    def construct(self):
        return self.stack((self.x1, self.x2))


def stack(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    stack_ = StackNet(nptype)
    output = stack_()
    expect = np.array([[[[[0, 0],
                          [0, 0]],
                         [[0, 1],
                          [2, 3]]],
                        [[[0, 0],
                          [0, 0]],
                         [[4, 5],
                          [6, 7]]]],
                       [[[[0, 0],
                          [0, 0]],
                         [[8, 9],
                          [10, 11]]],
                        [[[0, 0],
                          [0, 0]],
                         [[12, 13],
                          [14, 15]]]]]).astype(nptype)
    assert (output.asnumpy() == expect).all()

def stack_pynative(nptype):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x1 = np.array([0] * 16).astype(nptype)
    x1 = np.reshape(x1, (2, 2, 2, 2))
    x1 = Tensor(x1)
    x2 = Tensor(np.arange(16).reshape(2, 2, 2, 2).astype(nptype))
    expect = np.array([[[[[0, 0],
                          [0, 0]],
                         [[0, 1],
                          [2, 3]]],
                        [[[0, 0],
                          [0, 0]],
                         [[4, 5],
                          [6, 7]]]],
                       [[[[0, 0],
                          [0, 0]],
                         [[8, 9],
                          [10, 11]]],
                        [[[0, 0],
                          [0, 0]],
                         [[12, 13],
                          [14, 15]]]]]).astype(nptype)
    output = P.Stack(axis=2)((x1, x2))
    assert (output.asnumpy() == expect).all()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stack_graph_float32():
    stack(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stack_graph_float16():
    stack(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stack_graph_int32():
    stack(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stack_graph_int16():
    stack(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stack_graph_uint8():
    stack(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stack_graph_bool():
    stack(np.bool)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stack_pynative_float32():
    stack_pynative(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stack_pynative_float16():
    stack_pynative(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stack_pynative_int32():
    stack_pynative(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stack_pynative_int16():
    stack_pynative(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stack_pynative_uint8():
    stack_pynative(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_stack_pynative_bool():
    stack_pynative(np.bool)
