# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_fallback_runtime_map():
    """
    Feature: JIT Fallback
    Description: Test map() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo():
        x = Tensor(np.array([1, 2, 3, 4])).asnumpy()
        y = Tensor(np.array([1, 1, 1, 1])).asnumpy()
        ret = map(lambda x, y: x + y, x, y)
        return ret

    out = foo()
    assert isinstance(out, map)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_map_cust_class():
    """
    Feature: JIT Fallback
    Description: Test map() in fallback runtime
    Expectation: No exception
    """

    class GetattrClass():
        def __init__(self):
            self.attr1 = (1, 2, 3, 4)
            self.attr2 = 6

        def method1(self, x):
            return x + self.attr2

    class GetattrClassNet(ms.nn.Cell):
        def __init__(self):
            super(GetattrClassNet, self).__init__()
            self.cls = GetattrClass()

        def construct(self):
            x = self.cls.attr1
            out = map(self.cls.method1, x)
            return out

    net = GetattrClassNet()
    out = net()
    assert out == (7, 8, 9, 10)
