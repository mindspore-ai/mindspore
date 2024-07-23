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
# ============================================================================
""" test dtype and shape as attr"""
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_type_not_have_the_attr_runtime():
    """
    Feature: Support getattr in JIT Fallback.
    Description: Test getattr.
    Expectation: No exception.
    """
    class Net(nn.Cell):

        def construct(self, x):
            shape = x.shapes
            return shape

    net = Net()
    x = Tensor(np.ones([1, 2, 3], np.int32))
    with pytest.raises(AttributeError):
        net(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_type_not_have_the_method_runtime():
    """
    Feature: Support getattr in JIT Fallback.
    Description: Test getattr.
    Expectation: No exception.
    """
    class Net(nn.Cell):

        def construct(self, x):
            shape = x.dtypes()
            return shape

    net = Net()
    x = Tensor(np.ones([1, 2, 3], np.int32))
    with pytest.raises(AttributeError):
        net(x)
