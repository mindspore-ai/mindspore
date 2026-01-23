# Copyright 2019 Huawei Technologies Co., Ltd
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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops, Parameter
import mindspore


class Net(nn.Cell):
    def __init__(self, var, m, v):
        super(Net, self).__init__()
        self.apply_adam = ops.Adam()
        self.var = Parameter(var, name="var")
        self.m = Parameter(m, name="m")
        self.v = Parameter(v, name="v")

    def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
        out = self.apply_adam(self.var, self.m, self.v, beta1_power,
                              beta2_power, lr, beta1, beta2, epsilon, grad)
        return out


def get_output(var, m, v, grad, enable_graph_kernel=False):
    if enable_graph_kernel:
        context.set_context(jit_level='O1', graph_kernel_flags="--enable_expand_ops=Adam")
    else:
        context.set_context(jit_level='O0')
    net = Net(var, m, v)
    output = net(0.9, 0.999, 0.001, 0.9, 0.999, 1e-8, grad)
    return output


def run_basic(dtype):
    np.random.seed(42)
    shape = [10, 10]
    var = Tensor(np.random.random(shape), dtype=dtype)
    m = Tensor(np.random.random(shape), dtype=dtype)
    v = Tensor(np.random.random(shape), dtype=dtype)
    grad = Tensor(np.random.random(shape), dtype=dtype)
    expect = get_output(var, m, v, grad, False)
    output = get_output(var, m, v, grad, True)

    expect_np = expect[0].asnumpy().copy()
    output_np = output[0].asnumpy().copy()
    assert np.allclose(expect_np, output_np)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_adam_expander_float32(nptype=np.float32, mstype=None):
    """
    Feature: test O1 adam expander float32.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic(mindspore.float32)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_adam_expander_float16():
    """
    Feature: test O1 adam expander float16.
    Description: test float16 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic(mindspore.float16)
