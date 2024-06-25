# Copyright 2024 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
import mindspore.context as context
from mindspore import Tensor, jit
from mindspore.nn import Cell
from mindspore._extends.parse import compile_config
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_for_half_unroll_basic():
    """
    Feature: Half unroll compile optimization for for statement.
    Description: Only test for statement.
    Expectation: Correct result and no exception.
    """
    class ForLoopBasic(Cell):
        def __init__(self):
            super().__init__()
            self.array = (Tensor(np.array(10).astype(np.int32)), Tensor(np.array(5).astype(np.int32)))

        @jit(mode="PIJit")
        def construct(self, x):
            output = x
            for i in self.array:
                output += i

            return output

    context.set_context(mode=context.PYNATIVE_MODE)
    net = ForLoopBasic()
    x = Tensor(np.array(10).astype(np.int32))
    compile_config.FOR_HALF_UNROLL = 1
    res = net(x)
    compile_config.FOR_HALF_UNROLL = ''
    assert res == 25


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_for_half_unroll_if():
    """
    Feature: Half unroll compile optimization for for statement.
    Description: Test for-in statements.
    Expectation: Correct result and no exception.
    """
    class ForLoopIf(Cell):
        def __init__(self):
            super().__init__()
            self.array = (Tensor(np.array(10).astype(np.int32)), Tensor(np.array(5).astype(np.int32)))

        @jit(mode="PIJit")
        def construct(self, x):
            output = x
            for i in self.array:
                if i < 10:
                    output += i

            return output

    context.set_context(mode=context.PYNATIVE_MODE)
    net = ForLoopIf()
    x = Tensor(np.array(10).astype(np.int32))
    compile_config.FOR_HALF_UNROLL = 1
    res = net(x)
    compile_config.FOR_HALF_UNROLL = ''
    assert res == 15
