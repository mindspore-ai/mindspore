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
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


class CholeskyFuncNet(nn.Cell):
    def construct(self, x):
        return F.cholesky(x, upper=False)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_functional_cholesky(mode):
    """
    Feature: Test cholesky functional api.
    Description: Test cholesky functional api for Graph and PyNative modes.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor([[1.0, 1.0], [1.0, 2.0]], mstype.float32)
    net = CholeskyFuncNet()
    output = net(x)
    expect_output = np.array([[1., 0.], [1., 1.]])
    assert np.allclose(output.asnumpy(), expect_output)
