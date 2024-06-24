# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
import pytest


class Net(nn.Cell):
    """Net definition"""

    def __init__(self, input_data):
        super(Net, self).__init__()
        self.assign_add = P.AssignAdd()
        self.inputdata = input_data
        print("inputdata: ", self.inputdata)

    def construct(self, x):
        self.assign_add(self.inputdata, x)
        return self.inputdata


def assign_add_forward_test(ms_type, np_type):
    variable = Parameter(initializer('ones', [1], ms_type), name="global_step")
    value = Tensor(np.array([4.0]), ms_type)
    assign_add = Net(variable)
    output = assign_add(value)
    expected = np.array([5.0]).astype(np_type)
    if ms_type == mindspore.bfloat16:
        output_np = output.float().asnumpy()
        np.testing.assert_array_almost_equal(output_np, expected, decimal=3)
    else:
        output_np = output.asnumpy()
        np.testing.assert_array_almost_equal(output_np, expected, decimal=6)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_assign_add_forward_fp32(mode):
    """
    Feature: test assign add forward with mstype.float32, on mode GRAPH & PYNATIVE
    Description: test inputs using given mindspore type and data type
    Expectation: the result match with the expected result
    """
    context.set_context(mode=mode, device_target="Ascend")
    assign_add_forward_test(mindspore.float32, np.float32)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_assign_add_forward_bf16(mode):
    """
    Feature: test assign add forward with mstype.bfloat16, on mode GRAPH & PYNATIVE
    Description: test inputs using given mindspore type and data type
    Expectation: the result match with the expected result
    """
    context.set_context(mode=mode, device_target="Ascend")
    assign_add_forward_test(mindspore.bfloat16, np.float32)
