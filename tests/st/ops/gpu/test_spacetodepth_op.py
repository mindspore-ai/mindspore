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

class SpaceToDepthNet(nn.Cell):
    def __init__(self, nptype):
        super(SpaceToDepthNet, self).__init__()
        self.SpaceToDepth = P.SpaceToDepth(2)

        data_np = np.array([[[[0, 3],
                              [6, 9]],
                             [[1, 4],
                              [7, 10]],
                             [[2, 5],
                              [8, 11]]]]).astype(nptype)
        self.data_np = data_np
        self.x = Parameter(initializer(Tensor(self.data_np), (1, 3, 2, 2)), name='x')

    @ms_function
    def construct(self):
        return self.SpaceToDepth(self.x)


def SpaceToDepth(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    expect = np.arange(12).reshape((1, 12, 1, 1)).astype(nptype)
    std = SpaceToDepthNet(nptype)
    output = std()
    assert (output.asnumpy() == expect).all()

def SpaceToDepth_pynative(nptype):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    expect = np.arange(12).reshape((1, 12, 1, 1)).astype(nptype)

    std = P.SpaceToDepth(2)
    data_np = np.array([[[[0, 3],
                          [6, 9]],
                         [[1, 4],
                          [7, 10]],
                         [[2, 5],
                          [8, 11]]]]).astype(nptype)
    tensor_input = Tensor(data_np)
    output = std(tensor_input)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_float32():
    SpaceToDepth(np.float32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_float16():
    SpaceToDepth(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_int32():
    SpaceToDepth(np.int32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_int64():
    SpaceToDepth(np.int64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_int8():
    SpaceToDepth(np.int8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_int16():
    SpaceToDepth(np.int16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_uint8():
    SpaceToDepth(np.uint8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_uint16():
    SpaceToDepth(np.uint16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_uint32():
    SpaceToDepth(np.uint32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_uint64():
    SpaceToDepth(np.uint64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_float32():
    SpaceToDepth_pynative(np.float32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_float16():
    SpaceToDepth_pynative(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_int32():
    SpaceToDepth_pynative(np.int32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_int64():
    SpaceToDepth_pynative(np.int64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_int8():
    SpaceToDepth_pynative(np.int8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_int16():
    SpaceToDepth_pynative(np.int16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_uint8():
    SpaceToDepth_pynative(np.uint8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_uint16():
    SpaceToDepth_pynative(np.uint16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_uint32():
    SpaceToDepth_pynative(np.uint32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_uint64():
    SpaceToDepth_pynative(np.uint64)


class SpaceToDepthDynNet(nn.Cell):
    def __init__(self, block_size=2):
        super(SpaceToDepthDynNet, self).__init__()
        self.net = P.SpaceToDepth(block_size)

    @ms_function
    def construct(self, input_x):
        y1 = self.net(input_x)
        return y1


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_dyn_shape():
    """
    Feature: op dynamic shape
    Description: set input_shape None and input real tensor
    Expectation: success
    """

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_x = Tensor(np.arange(12).reshape((1, 3, 2, 2)).astype(np.float32))
    input_x_dyn = Tensor(shape=[1, None, None, None], dtype=input_x.dtype)
    net = SpaceToDepthDynNet(2)
    net.set_inputs(input_x_dyn)
    output = net(input_x)
    expect_shape = (1, 12, 1, 1)
    assert output.asnumpy().shape == expect_shape


test_spacetodepth_graph_float32()
test_spacetodepth_pynative_int32()
