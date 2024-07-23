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
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype


class CumSumDyNet(nn.Cell):
    def __init__(self, exclusive=False, reverse=False, axis=0):
        super(CumSumDyNet, self).__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.cumsum = P.CumSum(exclusive=exclusive, reverse=reverse)
        self.cast = P.Cast()
        self.axis = axis

    def construct(self, indices, x, axis):
        unique_indices, _ = self.unique(indices)
        x_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        real_x = self.gather(x, unique_indices, self.axis)
        real_x = self.cast(real_x, x_dtype)
        return real_x, self.cumsum(real_x, axis)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("data_type", [np.float32, np.float16, np.int32])
def test_dynamic_shape_cumsum(axis, data_type):
    """
    Feature: CumSum DynamicShape.
    Description: Test case of dynamic shape for CumSum operator.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    indices = Tensor(np.random.randint(0, 8, size=8)).astype(np.int32)
    x = Tensor(np.random.randint(-10, 10, size=(8, 4, 5)).astype(data_type))

    dy_net = CumSumDyNet()
    real_x, output = dy_net(indices, x, axis)

    real_x_np = real_x.asnumpy()
    expect = real_x_np.cumsum(axis)
    np.testing.assert_allclose(expect, output.asnumpy())
