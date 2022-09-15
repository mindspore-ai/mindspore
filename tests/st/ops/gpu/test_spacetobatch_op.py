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
import mindspore.ops.operations.array_ops as P
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

class SpaceToBatchNet(nn.Cell):
    def __init__(self, nptype, block_size=2, input_shape=(1, 1, 4, 4)):
        super(SpaceToBatchNet, self).__init__()
        self.SpaceToBatch = P.SpaceToBatch(block_size=block_size, paddings=[[0, 0], [0, 0]])
        input_size = 1
        for i in input_shape:
            input_size = input_size*i
        data_np = np.arange(input_size).reshape(input_shape).astype(nptype)
        self.x1 = Parameter(initializer(Tensor(data_np), input_shape), name='x1')


    @ms_function
    def construct(self):
        y1 = self.SpaceToBatch(self.x1)
        return y1


def SpaceToBatch(nptype, block_size=2, input_shape=(1, 1, 4, 4)):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_size = 1
    for i in input_shape:
        input_size = input_size*i
    expect = np.array([[[[0, 2],
                         [8, 10]]],
                       [[[1, 3],
                         [9, 11]]],
                       [[[4, 6],
                         [12, 14]]],
                       [[[5, 7],
                         [13, 15]]]]).astype(nptype)

    dts = SpaceToBatchNet(nptype, block_size, input_shape)
    output = dts()

    assert (output.asnumpy() == expect).all()

def SpaceToBatch_pynative(nptype, block_size=2, input_shape=(1, 1, 4, 4)):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_size = 1
    for i in input_shape:
        input_size = input_size*i
    expect = np.array([[[[0, 2],
                         [8, 10]]],
                       [[[1, 3],
                         [9, 11]]],
                       [[[4, 6],
                         [12, 14]]],
                       [[[5, 7],
                         [13, 15]]]]).astype(nptype)

    dts = P.SpaceToBatch(block_size=block_size, paddings=[[0, 0], [0, 0]])
    arr_input = Tensor(np.arange(input_size).reshape(input_shape).astype(nptype))
    output = dts(arr_input)

    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetobatch_graph_float32():
    SpaceToBatch(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetobatch_graph_float16():
    SpaceToBatch(np.float16)


class SpaceToBatchDynNet(nn.Cell):
    def __init__(self, block_size=2):
        super(SpaceToBatchDynNet, self).__init__()
        self.net = P.SpaceToBatch(block_size=block_size, paddings=[[0, 0], [0, 0]])

    @ms_function
    def construct(self, input_x):
        y1 = self.net(input_x)
        return y1


def test_spacetobatch_dyn_shape():
    """
    Feature: op dynamic shape
    Description: set input_shape None and input real tensor
    Expectation: success
    """

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x = Tensor(np.arange(16).reshape((1, 1, 4, 4)).astype(np.float32))
    input_x_dyn = Tensor(shape=[1, None, None, None], dtype=input_x.dtype)
    net = SpaceToBatchDynNet(2)
    net.set_inputs(input_x_dyn)
    output = net(input_x)
    expect = np.array([[[[0, 2],
                         [8, 10]]],
                       [[[1, 3],
                         [9, 11]]],
                       [[[4, 6],
                         [12, 14]]],
                       [[[5, 7],
                         [13, 15]]]]).astype(np.float32)
    assert output.asnumpy().shape == expect.shape
