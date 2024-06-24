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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.array_ops as P
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

class BatchToSpaceNet(nn.Cell):
    def __init__(self, nptype, block_size=2, input_shape=(4, 1, 2, 2)):
        super(BatchToSpaceNet, self).__init__()
        self.BatchToSpace = P.BatchToSpace(block_size=block_size, crops=[[0, 0], [0, 0]])
        input_size = 1
        for i in input_shape:
            input_size = input_size*i
        data_np = np.arange(input_size).reshape(input_shape).astype(nptype)
        self.x1 = Parameter(initializer(Tensor(data_np), input_shape), name='x1')


    @jit
    def construct(self):
        y1 = self.BatchToSpace(self.x1)
        return y1


def BatchToSpace(nptype, block_size=2, input_shape=(4, 1, 2, 2)):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    input_size = 1
    for i in input_shape:
        input_size = input_size*i
    expect = np.array([[[[0, 4, 1, 5],
                         [8, 12, 9, 13],
                         [2, 6, 3, 7],
                         [10, 14, 11, 15]]]]).astype(nptype)

    dts = BatchToSpaceNet(nptype, block_size, input_shape)
    output = dts()

    assert (output.asnumpy() == expect).all()

def BatchToSpace_pynative(nptype, block_size=2, input_shape=(4, 1, 2, 2)):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_size = 1
    for i in input_shape:
        input_size = input_size*i
    expect = np.array([[[[0, 4, 1, 5],
                         [8, 12, 9, 13],
                         [2, 6, 3, 7],
                         [10, 14, 11, 15]]]]).astype(nptype)

    dts = P.BatchToSpace(block_size=block_size, crops=[[0, 0], [0, 0]])
    arr_input = Tensor(np.arange(input_size).reshape(input_shape).astype(nptype))
    output = dts(arr_input)

    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_batchtospace_graph_float32():
    BatchToSpace(np.float32)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_batchtospace_graph_float16():
    BatchToSpace(np.float16)
