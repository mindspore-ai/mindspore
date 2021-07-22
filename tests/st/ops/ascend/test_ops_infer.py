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
""" test nn ops """
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype

from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_cast_op_attr():
    class CastNet(nn.Cell):
        def __init__(self):
            super(CastNet, self).__init__()
            self.cast = P.Cast()

        def construct(self, x, t):
            return self.cast(x, t)

    class CastTypeTest(nn.Cell):
        def __init__(self, net):
            super(CastTypeTest, self).__init__()
            self.net = net
            self.cast = P.Cast()

        def construct(self, x, y, z):
            cast_op = self.cast
            t1 = cast_op(x, mstype.float32)
            t2 = cast_op(y, mstype.int32)
            cast_net = self.net
            t3 = cast_net(x, mstype.float16)
            t4 = cast_net(y, mstype.int32)
            t5 = cast_net(z, mstype.float16)
            return (t1, t2, t3, t4, t5)

    net = CastTypeTest(CastNet())
    t1 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.int32))
    t2 = Tensor(np.ones([1, 16, 1, 3840]).astype(np.float32))
    t3 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.int32))
    out = net(t1, t2, t3)
    assert out[0].asnumpy().dtype == np.float32
    assert out[1].asnumpy().dtype == np.int32
    assert out[2].asnumpy().dtype == np.float16
    assert out[3].asnumpy().dtype == np.int32
    assert out[4].asnumpy().dtype == np.float16
