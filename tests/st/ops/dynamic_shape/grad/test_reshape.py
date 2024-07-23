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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
from mindspore import nn
from mindspore.ops import operations as P
from mindspore import context
from mindspore import Tensor
from .test_grad_of_dynamic import TestDynamicGrad


class NetReshape(nn.Cell):
    def __init__(self):
        super(NetReshape, self).__init__()
        self.reshape = P.Reshape()
        self.target = (3, 2)

    def construct(self, x):
        return self.reshape(x, self.target)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_reshape_shape():
    """
    Feature: Reshape Grad DynamicShape.
    Description: Test case of dynamic shape for Reshape grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetReshape())
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    test_dynamic.test_dynamic_grad_net(x)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_dynamic_reshape_rank():
    """
    Feature: Reshape Grad DynamicRank.
    Description: Test case of dynamic rank for Reshape grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetReshape())
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype(np.float32))
    test_dynamic.test_dynamic_grad_net(x, True)
