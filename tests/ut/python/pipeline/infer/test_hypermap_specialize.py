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
""" test_hypermap_partial """
import os
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.common.api import jit
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)


def test_hypermap_specialize_param():
    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.mul = P.Mul()

        def construct(self, x, y):
            ret = self.mul(x, y)
            return ret

    factor1 = Tensor(5, dtype=mstype.int32)
    x = Tensor(np.ones([1]).astype(np.int32))
    y = Tensor(np.ones([2]).astype(np.int32))
    net = Net()
    hypermap = C.HyperMap()

    @jit
    def hypermap_specialize_param():
        ret1 = hypermap(F.partial(net, factor1), (x, y))
        # List will be converted to Tuple in SimlifyDataStructurePass.
        ret2 = hypermap(F.partial(net, factor1), [x, y])
        return ret1, ret2

    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    expected_ret = (Tensor(np.full(1, 5).astype(np.int32)), Tensor(np.full(2, 5).astype(np.int32)))
    ret = hypermap_specialize_param()
    assert ret[0][0].asnumpy() == expected_ret[0].asnumpy()
    assert np.all(ret[0][1].asnumpy() == expected_ret[1].asnumpy())
    assert ret[1][0].asnumpy() == list(expected_ret[0].asnumpy())
    assert np.all(ret[1][1].asnumpy() == list(expected_ret[1].asnumpy()))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'
