# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_return_const_or_parameter """

import numpy as np

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor

context.set_context(mode=context.GRAPH_MODE)


class ChooseOneParam(nn.Cell):
    def __init__(self, flag):
        super(ChooseOneParam, self).__init__()
        self.flag = flag

    def construct(self, x, y):
        if self.flag == 0:
            return x
        return y


class ChooseOneConst(nn.Cell):
    def __init__(self, flag, x, y):
        super(ChooseOneConst, self).__init__()
        self.flag = flag
        self.x = x
        self.y = y

    def construct(self):
        if self.flag == 0:
            return self.x
        return self.y


def test_choose_input_x():
    choose = ChooseOneParam(0)
    tensor_x = Tensor(np.zeros(2), dtype=mstype.int32)
    tensor_y = Tensor(np.ones(2), dtype=mstype.int32)
    out = choose(tensor_x, tensor_y)
    assert np.allclose(tensor_x.asnumpy(), out.asnumpy())


def test_choose_input_y():
    choose = ChooseOneParam(1)
    tensor_x = Tensor(1, dtype=mstype.int32)
    tensor_y = Tensor(2, dtype=mstype.int32)
    out = choose(tensor_x, tensor_y)
    assert np.allclose(tensor_y.asnumpy(), out.asnumpy())


def test_choose_const_x():
    tensor_x = Tensor(np.zeros(2), dtype=mstype.int32)
    tensor_y = Tensor(np.ones(2), dtype=mstype.int32)
    choose = ChooseOneConst(0, tensor_x, tensor_y)
    out = choose()
    assert np.allclose(tensor_x.asnumpy(), out.asnumpy())


def test_choose_const_y():
    tensor_x = Tensor(np.zeros(2), dtype=mstype.int32)
    tensor_y = Tensor(np.ones(2), dtype=mstype.int32)
    choose = ChooseOneConst(1, tensor_x, tensor_y)
    out = choose()
    assert np.allclose(tensor_y.asnumpy(), out.asnumpy())
