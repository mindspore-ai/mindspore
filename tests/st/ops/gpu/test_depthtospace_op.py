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

class DepthToSpaceNet(nn.Cell):
    def __init__(self, nptype, block_size=2, input_shape=(1, 4, 3, 3)):
        super(DepthToSpaceNet, self).__init__()
        self.DepthToSpace = P.DepthToSpace(2)
        input_size = 1
        for i in input_shape:
            input_size = input_size*i
        self.data_np = np.arange(input_size).reshape(input_shape).astype(nptype)
        self.x = Parameter(initializer(Tensor(self.data_np), input_shape), name='x')

    @ms_function
    def construct(self):
        return self.DepthToSpace(self.x)


def DepthToSpace(nptype, block_size=2, input_shape=(1, 4, 3, 3)):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_size = 1
    for i in input_shape:
        input_size = input_size*i
    expect = np.arange(input_size).reshape(input_shape).astype(nptype)
    expect = DepthToSpaceNumpy(expect, block_size)

    dts = DepthToSpaceNet(nptype, block_size, input_shape)
    output = dts()
    assert (output.asnumpy() == expect).all()

def DepthToSpace_pynative(nptype, block_size=2, input_shape=(1, 4, 3, 3)):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_size = 1
    for i in input_shape:
        input_size = input_size*i
    expect = np.arange(input_size).reshape(input_shape).astype(nptype)
    expect = DepthToSpaceNumpy(expect, block_size)

    dts = P.DepthToSpace(2)
    arr_input = Tensor(np.arange(input_size).reshape(input_shape).astype(nptype))
    output = dts(arr_input)

    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_float32():
    DepthToSpace(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_float16():
    DepthToSpace(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_int32():
    DepthToSpace(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_int64():
    DepthToSpace(np.int64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_int8():
    DepthToSpace(np.int8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_int16():
    DepthToSpace(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_uint8():
    DepthToSpace(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_uint16():
    DepthToSpace(np.uint16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_uint32():
    DepthToSpace(np.uint32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_uint64():
    DepthToSpace(np.uint64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_float32():
    DepthToSpace_pynative(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_float16():
    DepthToSpace_pynative(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_int32():
    DepthToSpace_pynative(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_int64():
    DepthToSpace_pynative(np.int64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_int8():
    DepthToSpace_pynative(np.int8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_int16():
    DepthToSpace_pynative(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_uint8():
    DepthToSpace_pynative(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_uint16():
    DepthToSpace_pynative(np.uint16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_uint32():
    DepthToSpace_pynative(np.uint32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_uint64():
    DepthToSpace_pynative(np.uint64)
