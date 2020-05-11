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

import pytest
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
import mindspore.nn as nn
import mindspore.context as context
from mindspore.common import dtype as mstype

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
class BatchMatMulNet(nn.Cell):
    def __init__(self, transpose_a=False, transpose_b=False):
        super(BatchMatMulNet, self).__init__()
        self.batch_matmul = P.BatchMatMul(transpose_a, transpose_b)

    def construct(self, x, y):
        return self.batch_matmul(x, y)

def test_4D():
    input_x = Tensor(np.arange(2 * 4 * 1 * 3).reshape(2, 4, 1, 3), mstype.float32)
    input_y = Tensor(np.arange(2 * 4 * 3 * 4).reshape(2, 4, 3, 4), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BatchMatMulNet()
    output = net(input_x, input_y)
    expect = [[[[  20,   23,   26,   29]],
              [[ 200,  212,  224,  236]],
              [[ 596,  617,  638,  659]],
              [[1208, 1238, 1268, 1298]]],

              [[[2036, 2075, 2114, 2153]],
              [[3080, 3128, 3176, 3224]],
              [[4340, 4397, 4454, 4511]],
              [[5816, 5882, 5948, 6014]]]]
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_4D_transpose_a():
    input_x = Tensor(np.arange(2*4*3*1).reshape(2,4,3,1), mstype.float32)
    input_y = Tensor(np.arange(2*4*3*4).reshape(2,4,3,4), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BatchMatMulNet(transpose_a=True)
    output = net(input_x, input_y)
    expect = [[[[  20,   23,   26,   29]],
              [[ 200,  212,  224,  236]],
              [[ 596,  617,  638,  659]],
              [[1208, 1238, 1268, 1298]]],

              [[[2036, 2075, 2114, 2153]],
              [[3080, 3128, 3176, 3224]],
              [[4340, 4397, 4454, 4511]],
              [[5816, 5882, 5948, 6014]]]]
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_4D_transpose_b():
    input_x = Tensor(np.arange(2*4*1*3).reshape(2,4,1,3), mstype.float32)
    input_y = Tensor(np.arange(2*4*4*3).reshape(2,4,4,3), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BatchMatMulNet(transpose_b=True)
    output = net(input_x, input_y)
    expect = [[[[   5,   14,   23,   32]],
              [[ 158,  194,  230,  266]],
              [[ 527,  590,  653,  716]],
              [[1112, 1202, 1292, 1382]]],

              [[[1913, 2030, 2147, 2264]],
              [[2930, 3074, 3218, 3362]],
              [[4163, 4334, 4505, 4676]],
              [[5612, 5810, 6008, 6206]]]]
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_4D_transpose_ab():
    input_x = Tensor(np.arange(2*4*3*1).reshape(2,4,3,1), mstype.float32)
    input_y = Tensor(np.arange(2*4*4*3).reshape(2,4,4,3), mstype.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BatchMatMulNet(transpose_a=True, transpose_b=True)
    output = net(input_x, input_y)
    expect = [[[[   5,   14,   23,  32]],
              [[ 158,  194,  230,  266]],
              [[ 527,  590,  653,  716]],
              [[1112, 1202, 1292, 1382]]],

              [[[1913, 2030, 2147, 2264]],
              [[2930, 3074, 3218, 3362]],
              [[4163, 4334, 4505, 4676]],
              [[5612, 5810, 6008, 6206]]]]
    assert (output.asnumpy() == expect).all()

class BatchMatMulNet(nn.Cell):
    def __init__(self, transpose_a=False, transpose_b=False):
        super(BatchMatMulNet, self).__init__()
        self.batch_matmul = P.BatchMatMul(transpose_a, transpose_b)

    def construct(self, x, y):
        return self.batch_matmul(x, y)

def test_4D_fp16():
    input_x = Tensor(np.arange(2 * 4 * 1 * 3).reshape(2, 4, 1, 3), mstype.float16)
    input_y = Tensor(np.arange(2 * 4 * 3 * 4).reshape(2, 4, 3, 4), mstype.float16)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BatchMatMulNet()
    output = net(input_x, input_y)
    expect = [[[[  20,   23,   26,   29]],
              [[ 200,  212,  224,  236]],
              [[ 596,  617,  638,  659]],
              [[1208, 1238, 1268, 1298]]],

              [[[2036, 2075, 2114, 2153]],
              [[3080, 3128, 3176, 3224]],
              [[4340, 4397, 4454, 4511]],
              [[5816, 5882, 5948, 6014]]]]
    assert (output.asnumpy() == expect).all()
