# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class NpuFloatNet(nn.Cell):
    """ NpuFloat definition, base on the related code in test_math_ops.py."""

    def __init__(self):
        super(NpuFloatNet, self).__init__()
        self.mul = P.Mul()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_status = P.NPUClearFloatStatus()
        self.shape_op = P.Shape()
        self.select = P.Select()
        self.less = P.Less()
        self.cast = P.Cast()
        self.dtype = P.DType()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.sub = P.Sub()
        self.neg = P.Neg()

    def construct(self, x):
        init = self.alloc_status()
        clear_status = self.clear_status(init)
        x = F.depend(x, clear_status)  # let x depend on clear_status
        res = self.sub(x, self.neg(x))
        init = F.depend(init, res)  # let get_status depend on res
        get_status = self.get_status(init)
        # let reduce_sum depend on get_statusk
        init = F.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        base = self.cast(F.fill(self.dtype(res), self.shape_op(res), 0.0),
                         self.dtype(flag_sum))
        cond = self.less(base, flag_sum)
        out = self.select(cond, self.cast(base, self.dtype(res)), res)
        return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_float_not_overflow():
    """
    Feature: Auto monad feature.
    Description: Verify overflow operator.
    Expectation: No exception.
    """
    input_data = Tensor(np.full((8, 5, 3, 1), 655, dtype=np.float16), dtype=mstype.float16)
    net = NpuFloatNet()
    out = net(input_data)
    # not overflow, we should got expected output.
    expect = Tensor(np.full((8, 5, 3, 1), 655 * 2,
                            dtype=np.float16), dtype=mstype.float16)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_float_overflow():
    """
    Feature: Auto monad feature.
    Description: Verify overflow operator.
    Expectation: No exception.
    """
    input_data = Tensor(np.full((8, 5, 3, 1), 65504, dtype=np.float16), dtype=mstype.float16)
    net = NpuFloatNet()
    out = net(input_data)
    # all zero if overflowed.
    assert np.all(out.asnumpy() == 0)
