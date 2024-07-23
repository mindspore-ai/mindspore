# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.train import Model

context.set_context(device_target="Ascend")
grad = C.GradOperation(get_all=True, sens_param=True)


class Select(Cell):
    def __init__(self):
        super(Select, self).__init__()
        self.select = P.Select()

    def construct(self, cond, inputa, inputb):
        return self.select(cond, inputa, inputb)


class GradWrap(Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, cond, inputa, inputb, out_grad):
        gout = grad(self.network)(cond, inputa, inputb, out_grad)
        return gout


def me_select(cond, inputa, inputb):
    net = Select()
    net.set_train()
    model = Model(net)
    if isinstance(inputa, np.ndarray):
        inputa = Tensor(inputa)
    if isinstance(inputb, np.ndarray):
        inputb = Tensor(inputb)
    if isinstance(cond, np.bool_):
        cond = np.array(cond)

    out = model.predict(Tensor(cond), inputa, inputb)
    return out.asnumpy()


def cmp_select(input_cond, inputa, inputb):
    cond = input_cond > 0.5
    out_me = me_select(cond, inputa, inputb)
    print(input_cond)
    print(cond)
    print(inputa)
    print(inputb)
    print(out_me)


def test_select_2_2():
    input_cond = np.random.rand(2, 2)
    inputa = np.random.randn(2, 2).astype(np.float32)
    inputb = np.random.randn(2, 2).astype(np.float32)
    cmp_select(input_cond, inputa, inputb)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_functional_select_scalar():
    """
    Feature: Test functional select operator. Support x or y is a int/float.
    Description: Operator select's input `x` is a Tensor with int32 type, input `y` is a int.
    Expectation: Assert result.
    """
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    x = np.array([[12, 1], [1, 0]]).astype(np.int32)
    y = 2
    output = ops.select(Tensor(cond), Tensor(x), y)
    expect = [[12, 2], [1, 2]]
    error = np.ones(shape=[2, 2]) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_functional_select_bf16():
    """
    Feature: Test functional select operator. Support x and y is a bfloat16 tensor.
    Description: Operator select's input `x` and 'y' both are Tensor with bfloat16 type.
    Expectation: Assert result compare with torch.
    """
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    input_x_me = Tensor([[2.45, -0.38], [0.91, 0.23]], ms.bfloat16)
    input_y_me = Tensor([[0.83, 4.72], [1.89, 0.96]], ms.bfloat16)
    output_me = Select()(Tensor(cond), input_x_me, input_y_me)
    except_result = np.array([[2.45, 4.72], [0.91, 0.96]]).astype(np.float32)
    assert np.allclose(output_me.float().asnumpy(), except_result, 0.004, 0.004)
