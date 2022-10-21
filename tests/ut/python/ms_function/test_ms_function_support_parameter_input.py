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
import numpy as np

import mindspore.nn as nn
from mindspore.common import dtype as mstype
from mindspore import context, Tensor, Parameter, ParameterTuple, jit
import mindspore.ops as ops


class NetWithParamInput(nn.Cell):
    def __init__(self):
        super(NetWithParamInput, self).__init__()
        self.w = Parameter(Tensor([6], mstype.float32))

    @jit
    def construct(self, x, y):
        return (x + y) * self.w


def test_ms_func_parameter_input():
    """
    Feature: Functions decorated with jit support parameter as input in PyNative Mode.
    Description: Using parameter as input for functions decorated with jit.
    Expectation: Calculation result is correct.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor([1], mstype.float32)
    input_param = Parameter(Tensor([2], mstype.float32), name="param")
    net = NetWithParamInput()
    # check forward run
    out = net(input_x, input_param)
    assert np.allclose(out.asnumpy(), 18, 0.000001, 0.000001)
    # check grad
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    gradient = grad_op(net, ParameterTuple(net.trainable_params()))(input_x, input_param)
    assert np.allclose(gradient[0][0].asnumpy(), 6, 0.000001, 0.000001)
    assert np.allclose(gradient[0][1].asnumpy(), 6, 0.000001, 0.000001)
    assert np.allclose(gradient[1][0].asnumpy(), 3, 0.000001, 0.000001)
