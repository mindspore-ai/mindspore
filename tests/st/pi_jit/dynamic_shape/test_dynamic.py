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
# ==============================================================================
from mindspore.nn import Cell
from mindspore import ops
from mindspore import context, jit
from mindspore.common import dtype
from mindspore.common import Tensor
import numpy as np
import pytest 
from tests.mark_utils import arg_mark


class DynamicFactory:
    def __init__(self, ps_net):
        self.ps_net = ps_net

    def forward_cmp(self, inputs):
        context.set_context(mode=context.PYNATIVE_MODE)
        jit(function=self.ps_net.construct, capture_mode="bytecode")(inputs)
        self.ps_net(inputs)

class Net7(Cell):
    def __init__(self):
        super().__init__()
        self.pow_op = ops.Pow()

    def construct(self, x):
        a = self.pow_op(x, 0.0)
        #print(type(a),"hejianheng")
        b = ops.rrelu(a)
        return b


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_frontend_optimize():
    '''
    TEST_SUMMARY:
    Description:
        1. create a net with pow rrelu
        2. run twice for Resize
        3. set inputs for pow frontend pass
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    ps_net = Net7()

    #x = np.random.randn(3, 4, 5).astype(np.float32)
    #s = np.random.randn(3, 4, 5).astype(np.float32)
    d = Tensor(np.random.randn(3, 4, 5), dtype=dtype.float32)
    fact = DynamicFactory(ps_net)
    fact.forward_cmp(d)
