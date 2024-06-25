# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
""" test graph fallback """
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor, jit, context
from mindspore import ops, tensor
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)


class ControlNet(nn.Cell):
    def inner_function_1(self, a, b):
        return a + b

    def inner_function_2(self, a, b):
        return a - b

    def construct(self, x):
        a = Tensor(np.array(4), mstype.int32)
        b = Tensor(np.array(5), mstype.int32)
        if a + b > x:
            return self.inner_function_1(a, b)
        return self.inner_function_2(a, b)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_control_sink_tensor():
    """
    Feature: Fallback feature: support define Tensor in Class construct.
    Description: Fallback feature: support define Tensor in Class construct.
    Expectation: Fallback feature: support define Tensor in Class construct.
    """
    x = Tensor(np.array(1), mstype.int32)
    net = ControlNet()
    output = net(x)
    output_expect = Tensor(9, mstype.int32)
    assert output == output_expect


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_np_tensor_list():
    """
    Feature: Fallback feature
    Description: support Basic method of Tensor list.
    Expectation: No exception.
    """

    @jit
    def np_tensor_list():
        a = Tensor(np.array(4), mstype.int32)
        b = Tensor(np.array(5), mstype.int32)
        c = Tensor(np.array(6), mstype.int32)
        tensor_list = [a, b]
        for x in tensor_list:
            print(x)
        tensor_list.append(tensor_list[-1] + c)
        return tensor_list

    tensor_list = np_tensor_list()
    print("tensor_list:", tensor_list)
    assert len(tensor_list) == 3


@jit
def np_fallback_func_tensor_index(x):
    array_x = tuple([2, 3, 4, 5])
    np_x = np.array(array_x).astype(np.float32)
    me_x = Tensor(np_x)
    me_x = me_x + me_x
    return me_x[x]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_np_fallback_func_tensor_index():
    """
    Feature: Fallback feature: support Tensor index.
    Description: Fallback feature: support Tensor index.
    Expectation: Fallback feature: support Tensor index.
    """
    x = Tensor(1, mstype.int32)
    output = np_fallback_func_tensor_index(x)
    output_expect = Tensor(6, mstype.float32)
    assert output == output_expect


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_tensor_compare_with_variable():
    """
    Feature: Fallback feature
    Description: Test ms.Tensor() in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        while x > Tensor([0]):
            x = x - abs(Tensor([-1]))
        return x

    res = foo(Tensor([6]))
    assert res == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_np_tensor_add():
    """
    Feature: Fallback feature
    Description: support Tensor add.
    Expectation: No exception.
    """

    @jit
    def np_tensor_add():
        a = Tensor(np.array(4))
        b = Tensor(np.array(5))
        tensor_list = [a, b]
        for x in tensor_list:
            print(x)
        x = 6
        np_x = np.array(x)
        c = Tensor(np_x)
        d = tensor_list[-1] + c
        tensor_list.append(d)
        return tensor_list

    tensor_list = np_tensor_add()
    print("tensor_list:", tensor_list)
    assert tensor_list[-1] == 11


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_user_define_bprop_using_fallback():
    """
    Feature: Fallback feature
    Description: user define bprop support jit fallback.
    Expectation: No exception.
    """
    class TestBpropCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.const_value = 1

        def construct(self, x):
            x = x * self.const_value
            x = x.asnumpy()
            x = (x + x) * x
            return tensor(x, mstype.float32)

        def bprop(self, x, out, dout):
            x = dout.asnumpy()
            x = 2 * (x * x) * (np.log(x) + 1)
            return (tensor(x, mstype.float32),)

    class TestCell(nn.Cell):
        def __init__(self):
            super().__init__()
            self.user_define_bprop = TestBpropCell()

        def construct(self, x):
            x = 2 * x
            x = self.user_define_bprop(x)
            x = x + 1
            x = 2 * x
            return x

    test_cell = TestCell()
    input_x = Tensor([1, 2, 3, 4], mstype.float32)
    graph_output = ops.grad(test_cell)(input_x)

    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_out = ops.grad(test_cell)(input_x)
    context.set_context(mode=context.GRAPH_MODE)

    assert np.allclose(graph_output.asnumpy(), pynative_out.asnumpy())
