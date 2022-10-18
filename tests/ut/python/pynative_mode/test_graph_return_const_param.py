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
from mindspore.common.api import jit
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


class ChooseInitParameter(nn.Cell):
    def __init__(self):
        super(ChooseInitParameter, self).__init__()
        self.x = Parameter(Tensor(np.ones(2), dtype=mstype.int32), name='x')

    @jit
    def construct(self):
        return self.x


class ChooseInitParameterWithInput(nn.Cell):
    def __init__(self):
        super(ChooseInitParameterWithInput, self).__init__()
        self.x = Parameter(Tensor(np.ones(2), dtype=mstype.int32), name='x')

    @jit
    def construct(self, input_data):
        return self.x


def test_choose_init_param():
    choose = ChooseInitParameter()
    expect = Tensor(np.ones(2), dtype=mstype.int32)
    out = choose()
    assert np.allclose(out.asnumpy(), expect.asnumpy())


def test_choose_param_with_input():
    choose = ChooseInitParameterWithInput()
    input_data = Tensor(np.zeros(2), dtype=mstype.int32)
    expect = Tensor(np.ones(2), dtype=mstype.int32)
    out = choose(input_data)
    assert np.allclose(expect.asnumpy(), out.asnumpy())
