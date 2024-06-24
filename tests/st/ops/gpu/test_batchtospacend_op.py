# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore.ops.operations as ops
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter


class BatchToSpaceNDNet(nn.Cell):
    def __init__(self, nptype, block_shape=2, input_shape=(4, 1, 1, 1)):
        super(BatchToSpaceNDNet, self).__init__()
        self.batch_to_space_nd = ops.BatchToSpaceND(block_shape=block_shape, crops=[[0, 0], [0, 0]])
        input_size = np.prod(input_shape)
        data_np = np.arange(input_size).reshape(input_shape).astype(nptype)
        self.x1 = Parameter(initializer(Tensor(data_np), input_shape), name='x1')

    @ms_function
    def construct(self):
        y1 = self.batch_to_space_nd(self.x1)
        return y1


def batch_to_space_nd_test_case(nptype, block_shape=2, input_shape=(4, 1, 1, 1)):
    expect = np.array([[[[0, 1],
                         [2, 3]]]]).astype(nptype)

    dts = BatchToSpaceNDNet(nptype, block_shape, input_shape)
    output = dts()

    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_batch_to_space_nd_graph():
    """
    Feature: test BatchToSpaceND function interface.
    Description: test interface.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    batch_to_space_nd_test_case(np.float32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_batch_to_space_nd_pynative():
    """
    Feature: test BatchToSpaceND function interface.
    Description: test interface.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    batch_to_space_nd_test_case(np.float32)
