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
"""Tests for tensor and dynamic sequence conversions."""

from tests.mark_utils import arg_mark

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops.operations import _sequence_ops as seq
from mindspore import context
from mindspore import Tensor, mutable
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class Net(nn.Cell):
    def __init__(self, axis=0):
        super().__init__()
        self.tensor_to_tuple = seq.TensorToTuple()
        self.tensor_to_scalar = seq.TensorToScalar()

    def construct(self, x, y):
        return self.tensor_to_tuple(x), self.tensor_to_scalar(y)


def dyn_case():
    x = Tensor([1, 2, 3], mstype.int64)
    y = Tensor(1, mstype.float32)
    expect_x = (1, 2, 3)
    expect_y = 1.0
    net = Net()
    res_x, res_y = net(x, y)
    assert expect_x == res_x
    assert expect_y == res_y


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard')
def test_seq_to_tensor():
    """
    Feature: test TensorToxx.
    Description: inputs is dynamic sequence or scalar.
    Expectation: the result match with numpy result
    """
    dyn_case()


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard', mem_peak=0.01)
def test_tensor_to_tuple_bool():
    """
    Feature: TensorToTuple.
    Description: Test bool Tensor conversion to tuple in graph mode.
    Expectation: The output tuple contains the input bool values.
    """
    tensor_to_tuple = seq.TensorToTuple()
    output = tensor_to_tuple(Tensor([True, False], mstype.bool_))
    assert output == (True, False)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', mem_peak=0.01)
def test_dynamic_list_to_tensor_cpu():
    """
    Feature: ListToTensor.
    Description: Test dynamic list conversion to Tensor through the CPU kernel in graph mode.
    Expectation: The output Tensor contains the input values cast to float32.
    """
    class ListTensorNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.list_to_tensor = seq.ListToTensor()

        def construct(self, value):
            return self.list_to_tensor(value, mstype.float32)

    output = ListTensorNet()(mutable([1, 2], True))
    assert output.asnumpy().tolist() == [1.0, 2.0]
