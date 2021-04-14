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

def DepthToSpaceNumpy(arr, block_size):
    '''
     DepthToSpace ops use numpy
     DepthToSpace ops is reverse ops to SpaceToDepth ops
     therefore DepthToSpace's output can be SpaceToDepth's input
    '''
    tmpshape = arr.shape
    newshape = []
    newshape.append(tmpshape[0])
    newshape.append(tmpshape[1]//block_size//block_size)
    newshape.append(tmpshape[2]*block_size)
    newshape.append(tmpshape[3]*block_size)
    output = arr.reshape(newshape[0], newshape[1], block_size, block_size, tmpshape[2], tmpshape[3])
    output = np.transpose(output, (0, 1, 4, 2, 5, 3))
    output = output.reshape(newshape)
    return output

class SpaceToDepthNet(nn.Cell):
    def __init__(self, nptype, block_size=2, input_shape=(1, 4, 3, 3)):
        super(SpaceToDepthNet, self).__init__()
        self.SpaceToDepth = P.SpaceToDepth(block_size)
        input_size = 1
        for i in input_shape:
            input_size = input_size*i

        data_np = np.arange(input_size).reshape(input_shape).astype(nptype)# data_np shape is (N,C,H,W)
        data_np = DepthToSpaceNumpy(data_np, block_size)#now data_np shape is (N,C/(block_size*block_size),H*block_size,W*block_size)
        self.data_np = data_np
        new_shape = []
        new_shape.append(input_shape[0])
        new_shape.append(input_shape[1]//(block_size*block_size))
        new_shape.append(input_shape[2]*block_size)
        new_shape.append(input_shape[3]*block_size)
        self.x = Parameter(initializer(Tensor(self.data_np), new_shape), name='x')

    @ms_function
    def construct(self):
        return self.SpaceToDepth(self.x)


def SpaceToDepth(nptype, block_size=2, input_shape=(1, 4, 3, 3)):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_size = 1
    for i in input_shape:
        input_size = input_size*i

    expect = np.arange(input_size).reshape(input_shape).astype(nptype)

    std = SpaceToDepthNet(nptype, block_size, input_shape)
    output = std()
    assert (output.asnumpy() == expect).all()

def SpaceToDepth_pynative(nptype, block_size=2, input_shape=(1, 4, 3, 3)):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_size = 1
    for i in input_shape:
        input_size = input_size*i
    expect = np.arange(input_size).reshape(input_shape).astype(nptype)
    arrinput = DepthToSpaceNumpy(expect, block_size)

    std = P.SpaceToDepth(block_size)
    arrinput = Tensor(arrinput)
    output = std(arrinput)

    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_float32():
    SpaceToDepth(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_float16():
    SpaceToDepth(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_int32():
    SpaceToDepth(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_int64():
    SpaceToDepth(np.int64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_int8():
    SpaceToDepth(np.int8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_int16():
    SpaceToDepth(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_uint8():
    SpaceToDepth(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_uint16():
    SpaceToDepth(np.uint16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_uint32():
    SpaceToDepth(np.uint32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_graph_uint64():
    SpaceToDepth(np.uint64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_float32():
    SpaceToDepth_pynative(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_float16():
    SpaceToDepth_pynative(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_int32():
    SpaceToDepth_pynative(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_int64():
    SpaceToDepth_pynative(np.int64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_int8():
    SpaceToDepth_pynative(np.int8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_int16():
    SpaceToDepth_pynative(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_uint8():
    SpaceToDepth_pynative(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_uint16():
    SpaceToDepth_pynative(np.uint16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_uint32():
    SpaceToDepth_pynative(np.uint32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_spacetodepth_pynative_uint64():
    SpaceToDepth_pynative(np.uint64)
