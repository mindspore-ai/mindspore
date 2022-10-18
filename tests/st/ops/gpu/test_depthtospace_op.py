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
from mindspore.common.api import jit
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

class DepthToSpaceNet(nn.Cell):
    def __init__(self, nptype, block_size=2, input_shape=(1, 12, 1, 1)):
        super(DepthToSpaceNet, self).__init__()
        self.DepthToSpace = P.DepthToSpace(2)
        input_size = 1
        for i in input_shape:
            input_size = input_size*i
        data_np = np.arange(input_size).reshape(input_shape).astype(nptype)
        self.x1 = Parameter(initializer(Tensor(data_np), input_shape), name='x1')


    @jit
    def construct(self):
        y1 = self.DepthToSpace(self.x1)
        return y1


def DepthToSpace(nptype, block_size=2, input_shape=(1, 12, 1, 1)):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_size = 1
    for i in input_shape:
        input_size = input_size*i
    expect = np.array([[[[0, 3],
                         [6, 9]],
                        [[1, 4],
                         [7, 10]],
                        [[2, 5],
                         [8, 11]]]]).astype(nptype)

    dts = DepthToSpaceNet(nptype, block_size, input_shape)
    output = dts()
    print(output)
    assert (output.asnumpy() == expect).all()

def DepthToSpace_pynative(nptype, block_size=2, input_shape=(1, 12, 1, 1)):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_size = 1
    for i in input_shape:
        input_size = input_size*i
    expect = np.array([[[[0, 3],
                         [6, 9]],
                        [[1, 4],
                         [7, 10]],
                        [[2, 5],
                         [8, 11]]]]).astype(nptype)

    dts = P.DepthToSpace(2)
    arr_input = Tensor(np.arange(input_size).reshape(input_shape).astype(nptype))
    output = dts(arr_input)

    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_float32():
    DepthToSpace(np.float32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_float16():
    DepthToSpace(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_int32():
    DepthToSpace(np.int32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_int64():
    DepthToSpace(np.int64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_int8():
    DepthToSpace(np.int8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_int16():
    DepthToSpace(np.int16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_uint8():
    DepthToSpace(np.uint8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_uint16():
    DepthToSpace(np.uint16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_uint32():
    DepthToSpace(np.uint32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_graph_uint64():
    DepthToSpace(np.uint64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_float32():
    DepthToSpace_pynative(np.float32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_float16():
    DepthToSpace_pynative(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_int32():
    DepthToSpace_pynative(np.int32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_int64():
    DepthToSpace_pynative(np.int64)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_int8():
    DepthToSpace_pynative(np.int8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_int16():
    DepthToSpace_pynative(np.int16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_uint8():
    DepthToSpace_pynative(np.uint8)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_uint16():
    DepthToSpace_pynative(np.uint16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_uint32():
    DepthToSpace_pynative(np.uint32)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_depthtospace_pynative_uint64():
    DepthToSpace_pynative(np.uint64)
