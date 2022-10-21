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

import platform
import numpy as np
import pytest

from mindspore import nn
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore import jit, context
import mindspore.ops as ops


class PyNet(nn.Cell):
    def __init__(self):
        super(PyNet, self).__init__()
        self.w1 = Parameter(Tensor(np.ones((2, 2), np.float32)), name="w1")

    @jit
    def construct(self, param_a, list_a, tuple_a, tensor_a, dict_a, param_b, tensor_b):
        output = param_a + list_a[0] + tuple_a[1] - tensor_a - dict_a["x"] - param_b + tensor_b
        output = output * self.w1
        return output


class GraphNet(nn.Cell):
    def __init__(self):
        super(GraphNet, self).__init__()
        self.w2 = Parameter(Tensor(np.ones((2, 2), np.float32)), name="w2")

    def construct(self, param_x, list_x, tuple_x, tensor_x, dict_x, param_y, tensor_y):
        output = param_x + list_x[0] + tuple_x[1] - tensor_x - dict_x["x"] - param_y + tensor_y
        output = output * self.w2
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_ms_function_support_parameter_as_input():
    """
    Feature: PyNative ms_function support parameter as input.
    Description: PyNative ms_function support parameter as input.
    Expectation: The calculation result is correct.
    """
    if platform.system() == 'Windows':
        return

    tensor_a = Tensor(np.ones((2, 2), np.float32))
    tensor_b = Tensor(np.ones((2, 2), np.float32) * 2)
    tuple_a = (Tensor(np.ones((2, 2), np.float32) * 3), Tensor(np.ones((2, 2), np.float32) * 4))
    list_a = [Tensor(np.ones((2, 2), np.float32) * 5), Tensor(np.ones((2, 2), np.float32) * 6)]
    dict_a = {"x": Tensor(np.ones((2, 2), np.float32) * 7), "y": Tensor(np.ones((2, 2), np.float32) * 8)}
    param_a = Parameter(Tensor(np.ones((2, 2), np.float32)), name="param1")
    param_b = Parameter(Tensor(np.ones((2, 2), np.float32) * 2), name="param2")

    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    context.set_context(mode=context.PYNATIVE_MODE)
    net1 = PyNet()
    output1 = grad_op(net1, ParameterTuple(net1.trainable_params()))(param_a, list_a, tuple_a, tensor_a, dict_a,
                                                                     param_b, tensor_b)
    context.set_context(mode=context.GRAPH_MODE)
    net2 = GraphNet()
    output2 = grad_op(net2, ParameterTuple(net2.trainable_params()))(param_a, list_a, tuple_a, tensor_a, dict_a,
                                                                     param_b, tensor_b)
    assert np.allclose(output1[0][0].asnumpy(), output2[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(output1[0][1].asnumpy(), output2[0][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(output1[0][2].asnumpy(), output2[0][2].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(output1[0][3].asnumpy(), output2[0][3].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(output1[1][0].asnumpy(), output2[1][0].asnumpy(), 0.000001, 0.000001)
