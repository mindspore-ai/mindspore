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
import pytest
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
@pytest.mark.parametrize(
    "value, output_dtype, expected",
    [
        pytest.param([1.5, 2.5], mstype.float32, [1.5, 2.5], id="float64_to_float32"),
        pytest.param([1.5, 2.5], mstype.float64, [1.5, 2.5], id="float64_to_float64"),
        pytest.param([1.5, 2.5], mstype.int32, [1, 2], id="float64_to_int32"),
        pytest.param([1.5, 2.5], mstype.int64, [1, 2], id="float64_to_int64"),
        pytest.param([1, 2], mstype.float32, [1.0, 2.0], id="int64_to_float32"),
        pytest.param([1, 2], mstype.float64, [1.0, 2.0], id="int64_to_float64"),
        pytest.param([1, 2], mstype.int32, [1, 2], id="int64_to_int32"),
        pytest.param([1, 2], mstype.int64, [1, 2], id="int64_to_int64"),
        pytest.param([True, False], mstype.float32, [1.0, 0.0], id="bool_to_float32"),
        pytest.param([True, False], mstype.float64, [1.0, 0.0], id="bool_to_float64"),
        pytest.param([True, False], mstype.int32, [1, 0], id="bool_to_int32"),
        pytest.param([True, False], mstype.int64, [1, 0], id="bool_to_int64"),
        pytest.param([True, False], mstype.bool_, [True, False], id="bool_to_bool"),
    ],
)
def test_dynamic_list_to_tensor_cpu(value, output_dtype, expected):
    """
    Feature: ListToTensor.
    Description: Test dynamic list conversion to Tensor through the CPU kernel in graph mode.
    Expectation: The output Tensor has the expected dtype and values.
    """
    class ListTensorNet(nn.Cell):
        def __init__(self, dtype):
            super().__init__()
            self.list_to_tensor = seq.ListToTensor()
            self.dtype = dtype

        def construct(self, value):
            return self.list_to_tensor(value, self.dtype)

    output = ListTensorNet(output_dtype)(mutable(value, True))
    assert output.dtype == output_dtype
    assert output.asnumpy().tolist() == expected


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', mem_peak=0.01)
@pytest.mark.parametrize(
    "input_values, input_dtype, output_dtype, expected",
    [
        pytest.param((1.5, 2.5), mstype.float32, mstype.float32, [1.5, 2.5], id="float32_to_float32"),
        pytest.param((1.5, 2.5), mstype.float32, mstype.float64, [1.5, 2.5], id="float32_to_float64"),
        pytest.param((1.5, 2.5), mstype.float32, mstype.int32, [1, 2], id="float32_to_int32"),
        pytest.param((1.5, 2.5), mstype.float32, mstype.int64, [1, 2], id="float32_to_int64"),
        pytest.param((1, 2), mstype.int32, mstype.float32, [1.0, 2.0], id="int32_to_float32"),
        pytest.param((1, 2), mstype.int32, mstype.float64, [1.0, 2.0], id="int32_to_float64"),
        pytest.param((1, 2), mstype.int32, mstype.int32, [1, 2], id="int32_to_int32"),
        pytest.param((1, 2), mstype.int32, mstype.int64, [1, 2], id="int32_to_int64"),
    ],
)
def test_tensor_scalar_list_to_tensor_cpu(input_values, input_dtype, output_dtype, expected):
    """
    Feature: ListToTensor.
    Description: Test float32 and int32 scalar lists converted to Tensor through the CPU kernel in graph mode.
    Expectation: The output Tensor has the expected dtype and values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    class ListTensorNet(nn.Cell):
        def __init__(self, dtype):
            super().__init__()
            self.tensor_to_scalar = seq.TensorToScalar()
            self.list_to_tensor = seq.ListToTensor()
            self.dtype = dtype

        def construct(self, x, y):
            value = [self.tensor_to_scalar(x), self.tensor_to_scalar(y)]
            return self.list_to_tensor(value, self.dtype)

    output = ListTensorNet(output_dtype)(Tensor(input_values[0], input_dtype), Tensor(input_values[1], input_dtype))
    assert output.dtype == output_dtype
    assert output.asnumpy().tolist() == expected
