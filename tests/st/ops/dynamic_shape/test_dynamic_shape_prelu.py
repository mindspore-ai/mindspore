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

import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype


class GetDynamicInputNet(nn.Cell):
    def __init__(self, axis=0):
        super(GetDynamicInputNet, self).__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.cast = P.Cast()
        self.axis = axis

    def construct(self, x, indices):
        unique_indices, _ = self.unique(indices)
        x_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        real_x = self.gather(x, unique_indices, self.axis)
        return self.cast(real_x, x_dtype)


class PReLUDyNet(nn.Cell):
    def __init__(self):
        super(PReLUDyNet, self).__init__()
        self.op = P.PReLU()
        self.transformer = GetDynamicInputNet()

    def construct(self, indices, x, weight):
        real_x = self.transformer(x, indices)
        out = self.op(real_x, weight)
        return real_x, weight, out


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_shape", [((8, 6, 7), (1,))])
@pytest.mark.parametrize("data_type", [np.float16, np.float32])
def test_dynamic_shape_prelu(data_shape, data_type):
    """
    Feature: PReLU DynamicShape.
    Description: Test case of dynamic shape for PReLU operator.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x_shape, weight_shape = data_shape
    x = Tensor(np.random.random(size=x_shape).astype(data_type))
    weight = Tensor(np.random.random(size=weight_shape).astype(data_type))
    indices = Tensor(np.random.randint(0, x_shape[0], size=(5,)).astype(np.int32))

    dy_net = PReLUDyNet()
    real_x, real_weight, output = dy_net(indices, x, weight)
    x, weight = real_x.asnumpy(), real_weight.asnumpy()
    expect = np.where(x >= 0, x, weight * x)

    np.testing.assert_allclose(expect, output.asnumpy())
