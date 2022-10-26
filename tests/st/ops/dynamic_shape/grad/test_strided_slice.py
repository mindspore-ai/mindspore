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

import pytest

import mindspore as ms
from mindspore import nn, context, Tensor
from mindspore.ops.operations import array_ops as P
from .test_grad_of_dynamic import TestDynamicGrad

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


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu
def test_strided_slice_dyn_shape():
    """
    Feature: StridedSlice Grad DynamicShape.
    Description: Test case of dynamic shape for StridedSlice grad operator.
    Expectation: success.
    """
    strided_slice_test(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu
def test_strided_slice_dyn_rank():
    """
    Feature: StridedSlice Grad DynamicRank.
    Description: Test case of dynamic rank for StridedSlice grad operator.
    Expectation: success.
    """
    strided_slice_test(True)
