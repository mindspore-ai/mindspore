# Copyright 2025 Huawei Technologies Co., Ltd
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
"""pynative function"""
import numpy as np
import os
import pytest
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.graph.api import jit
from mindspore.ops.composite import GradOperation
from .ms_function_export_mindir import excute_export_mindir
from .utils import GradOfAllInputs
from .utils import allclose_nparray
from tests.mark_utils import arg_mark


@jit
def square(x):
    fun = P.Square()(x)
    return fun


def square0(x):
    fun = P.Square()(x)
    return fun


@jit
def squeeze(x, axis):
    fun = P.Squeeze(axis)(x)
    return fun


def squeeze0(x, axis):
    fun = P.Squeeze(axis)(x)
    return fun


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.02)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
@pytest.mark.offline_infer_executor
def test_pynative_ms_function_double_ms_fun(mode):
    """
    Feature: pynative function.
    Description: execute function in pynative mode twice.
    Expectation: the result is same.
    """
    ms.set_context(mode=mode)

    class Net0(Cell):
        def __init__(self):
            super().__init__()
            self.square = P.Square()

        @jit
        def neg(self, x):
            x = P.Neg()(x)
            return x

        @jit
        def construct(self, x):
            x = self.square(x)
            x = self.neg(x)
            return x

    class Net1(Cell):
        def __init__(self):
            super().__init__()
            self.square = P.Square()

        def neg(self, x):
            x = P.Neg()(x)
            return x

        def construct(self, x):
            x = self.square(x)
            x = self.neg(x)
            return x

    inputs = Tensor(np.random.randn(2, 2).astype(np.float32))

    net0 = Net0()
    out_ms_fun = net0(inputs)
    grad_net0 = GradOfAllInputs(net0)
    grad_net0.set_train()
    input_grad0 = grad_net0(inputs, out_ms_fun)

    net1 = Net1()
    out_pynative = net1(inputs)
    grad_net1 = GradOfAllInputs(net1)
    grad_net1.set_train()
    input_grad1 = grad_net1(inputs, out_ms_fun)

    allclose_nparray(out_ms_fun.asnumpy(), out_pynative.asnumpy(), 0.001, 0.001)
    allclose_nparray(input_grad0[0].asnumpy(), input_grad1[0].asnumpy(), 0.001, 0.001)

    # 导出@jit的mindir文件
    if 'CONTEXT_MODE' in os.environ:
        out1 = net0.neg(inputs)
        out_me = excute_export_mindir(func=Net0.neg, input_data=(None, inputs),
                                      name="test_pynative_ms_function_double_ms_fun_not0_neg",
                                      input_num=1, out_num=1)
        a = list(range(len(out1)))
        for i in a:
            allclose_nparray(out1[i].asnumpy(), out_me[i].asnumpy(), 0, 0)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=1024.03)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_pynative_ms_function_for_fun_same_scalar(mode):
    """
    Feature: pynative function.
    Description: execute squeeze function in pynative mode twice.
    Expectation: the result is same.
    """
    ms.set_context(mode=mode)

    inputs = Tensor(np.random.randn(1, 1, 1, 2).astype(np.float32))
    axis = 0

    def net0(x, axis):
        return squeeze(squeeze(square(x), axis), axis)

    out_ms_fun = net0(inputs, axis)
    grad_net0 = GradOperation(get_all=True)
    input_grad0 = grad_net0(net0)(inputs, axis)

    def net1(x, axis):
        return squeeze0(squeeze0(square0(x), axis), axis)

    out_pynative = net1(inputs, axis)
    grad_net1 = GradOperation(get_all=True)
    input_grad1 = grad_net1(net1)(inputs, axis)

    allclose_nparray(out_ms_fun.asnumpy(), out_pynative.asnumpy(), 0.001, 0.001)
    allclose_nparray(input_grad0[0].asnumpy(), input_grad1[0].asnumpy(), 0.001, 0.001)
