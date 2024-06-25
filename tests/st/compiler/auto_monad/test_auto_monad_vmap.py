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
# ==============================================================================
import pytest
import mindspore.nn as nn
from mindspore import context, Tensor, Parameter, jit
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_monad_vmap():
    """
    Feature: Auto monad feature:auto monad eliminate.
    Description: If exist special node, should not replace update_state for the load node.
    Expectation: No exception.
    """
    class AssignNet(nn.Cell):
        def __init__(self):
            super(AssignNet, self).__init__()
            self.assign = P.Assign()
            self.value = Tensor([3, 4], mstype.int32)

        def construct(self, x):
            x = self.assign(x, self.value)
            return x

    vampfunc = F.vmap(AssignNet())

    @jit
    def test_monad(a):
        c = Tensor([[1, 2], [3, 4], [5, 6]], mstype.int32)
        out = vampfunc(a)
        c = a + c
        P.AssignAdd()(a, c)
        out2 = a
        return out, out2

    a = Parameter(Tensor([[1, 2], [3, 4], [5, 6]], mstype.int32), name='param_a')
    out = test_monad(a)
    assert (out[0].asnumpy() == [[3, 4], [3, 4], [3, 4]]).all()
    assert (out[1].asnumpy() == [[7, 10], [9, 12], [11, 14]]).all()
