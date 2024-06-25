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
from mindspore import context, Tensor
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_round_cust_class():
    """
    Feature: JIT Fallback
    Description: Test round() in fallback runtime
    Expectation: No exception
    """

    class GetattrClass():
        def __init__(self):
            self.attr1 = 99.909
            self.attr2 = 1

        def method1(self, x):
            return x + self.attr2

    class GetattrClassNet(ms.nn.Cell):
        def __init__(self):
            super(GetattrClassNet, self).__init__()
            self.cls = GetattrClass()

        def construct(self):
            return round(self.cls.method1(self.cls.attr1), 2), round(100.909, 2)

    net = GetattrClassNet()
    out = net()
    assert np.allclose(out[0], 100.91, 0.0001, 0.0001)
    assert np.allclose(out[1], 100.91, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_round_var_tensor_set_digit():
    """
    Feature: JIT Fallback
    Description: Test round() in fallback runtime
    Expectation: No exception
    """
    @ms.jit
    def foo(x):
        return round(x, 1)

    x = Tensor([-1, -2, -3])
    with pytest.raises(TypeError) as e:
        foo(x)
    assert "When applying round() to tensor, only one tensor is supported as input" in str(e.value)
