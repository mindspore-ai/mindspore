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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

from mindspore import context, nn, set_seed
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)
context.set_context(device_target="Ascend")
set_seed(2)


class AdamApplyOneWithDecayNet(nn.Cell):
    def __init__(self):
        super(AdamApplyOneWithDecayNet, self).__init__()
        self.add = P.Add()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.real_div = P.RealDiv()
        self.sqrt = P.Sqrt()
        self.square = P.Square()

    def construct(self, input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
        mul0 = self.mul(mul0_x, input2)
        mul1 = self.mul(mul1_x, input0)
        square0 = self.square(input0)
        add0 = self.add(mul0, mul1)
        mul2 = self.mul(mul2_x, input1)
        mul3 = self.mul(mul3_x, square0)
        add1 = self.add(mul2, mul3)
        sqrt0 = self.sqrt(add1)
        add2 = self.add(add2_y, sqrt0)
        mul4 = self.mul(mul4_x, input3)
        real_div0 = self.real_div(add0, add2)
        add3 = self.add(mul4, real_div0)
        mul5 = self.mul(input4, add3)
        sub0 = self.sub(input3, mul5)
        return add1, add0, sub0


def adam_apply_one_with_decay_np(input0, input1, input2, input3, input4,
                                 mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y):
    mul0 = mul0_x * input2
    mul1 = mul1_x * input0
    square0 = input0 * input0
    add0 = mul0 + mul1
    mul2 = mul2_x * input1
    mul3 = mul3_x * square0
    add1 = mul2 + mul3
    sqrt0 = np.sqrt(add1)
    add2 = add2_y + sqrt0
    mul4 = mul4_x * input3
    real_div0 = np.true_divide(add0, add2)
    add3 = mul4 + real_div0
    mul5 = input4 * add3
    sub0 = input3 - mul5
    return add1, add0, sub0


def compute_func(ms_net, np_net, is_dyn=False):
    if is_dyn:
        inputs = Tensor(shape=[2, None], dtype=mstype.float32)
        ms_net.set_inputs(inputs, inputs, inputs, inputs, inputs,
                          inputs, inputs, inputs, inputs, inputs, inputs)
    input0 = np.array([[0.1, 0.3, 3.6], [0.4, 0.5, 3.2]]).astype(np.float32)
    out0, out1, out2 = ms_net(Tensor(input0), Tensor(input0), Tensor(input0),
                              Tensor(input0), Tensor(input0), Tensor(input0),
                              Tensor(input0), Tensor(input0), Tensor(input0),
                              Tensor(input0), Tensor(input0))
    np0, np1, np2 = np_net(input0, input0, input0, input0, input0,
                           input0, input0, input0, input0, input0, input0)
    assert np.all(out0.asnumpy() == np0)
    assert np.all(out1.asnumpy() == np1)
    assert np.all(out2.asnumpy() == np2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_adam_apply_one_with_decay():
    """
    Feature: Test AdamApplyOneWithDecay.
    Description: The input shape is static.
    Expectation: Assert that results are consistent with numpy.
    """
    ms_net = AdamApplyOneWithDecayNet()
    np_net = adam_apply_one_with_decay_np
    compute_func(ms_net, np_net)
