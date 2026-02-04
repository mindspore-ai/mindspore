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
"""Test python super()"""

import mindspore
from mindspore import jit, nn

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_equal, assert_executed_by_graph_mode


class IdentityOp(nn.Cell):
    def construct(self, x, *args, **kwargs):
        return x


class IdentityFuncOp(IdentityOp):
    def construct(self, x, *args, **kwargs):
        return super().construct


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_super_v1():
    """
    Feature: Calling super().
    Description: Use super() in child class, to call the parent class method.
    Expectation: No graph break.
    """

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = IdentityFuncOp()

        def construct(self, x, y):
            z = self.op(x, y)(x)  # return x
            return z * 2

    model = Model()
    x = mindspore.tensor([1, 2])
    y = mindspore.tensor([3, 4])

    o1 = model(x, y)

    model.construct = jit(model.construct, capture_mode='bytecode', fullgraph=True)
    o2 = model(x, y)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(model.construct)
