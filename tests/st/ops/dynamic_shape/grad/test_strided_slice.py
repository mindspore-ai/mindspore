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

import pytest
import numpy as np
import mindspore as ms
from mindspore import nn, context, Tensor
from mindspore.ops.operations import array_ops as P
from .test_grad_of_dynamic import TestDynamicGrad
from mindspore.nn import Cell
from mindspore import ops as op

context.set_context(mode=context.PYNATIVE_MODE)


class NetStridedSlice(nn.Cell):

    def __init__(self):
        super(NetStridedSlice, self).__init__()
        self.op = P.StridedSlice()

    def construct(self, input_x, begin, end, strides):
        return self.op(input_x, begin, end, strides)


def strided_slice_test(is_dyn_rank):
    input_x = Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],
                      [[5, 5, 5], [6, 6, 6]]],
                     dtype=ms.float32)
    begin = (1, 0, 2)
    end = (3, 1, 3)
    strides = (1, 1, 1)
    tester = TestDynamicGrad(NetStridedSlice())
    tester.test_dynamic_grad_net([input_x, begin, end, strides], is_dyn_rank)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_dyn_shape():
    """
    Feature: StridedSlice Grad DynamicShape.
    Description: Test case of dynamic shape for StridedSlice grad operator.
    Expectation: success.
    """
    strided_slice_test(False)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_dyn_rank():
    """
    Feature: StridedSlice Grad DynamicRank.
    Description: Test case of dynamic rank for StridedSlice grad operator.
    Expectation: success.
    """
    strided_slice_test(True)


class StridedSliceNet(Cell):
    def __init__(self,
                 begin_mask,
                 end_mask,
                 ellipsis_mask,
                 new_axis_mask,
                 shrink_axis_mask):
        super().__init__()
        self.strided_slice = op.StridedSlice(begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                             shrink_axis_mask)

    def construct(self, input_x, begin, end, strides):
        out = self.strided_slice(input_x, begin, end, strides)
        return op.add(out, out)

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_strided_slice_insert_cast_for_tuple_input():
    """
    Feature: StridedSlice Grad DynamicShape in pynative mode.
    Description: The input is a tuple of bprop of strided slice, should be converted to tensor in pynative mode.
    Expectation: No exception raised.
    """
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE)
    net = StridedSliceNet(0, 0, 0, 0, 0)
    grad_net = op.grad(net)
    x = ms.Tensor(np.ones([5, 6, 7]).astype(np.float32))
    x_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)
    begin = ms.Tensor(np.array([0, 1, 2]).astype(np.int32))
    end = ms.Tensor(np.array([4, 5, 6]).astype(np.int32))
    strides = ms.Tensor(np.array([1, 1, 2]).astype(np.int32))
    net.set_inputs(x_dyn, begin, end, strides)
    grad_net(x, begin, end, strides)
