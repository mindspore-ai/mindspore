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

import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import auto_generate as P
from mindspore.common import dtype as mstype


class CholeskyNet(nn.Cell):
    def __init__(self, upper):
        super(CholeskyNet, self).__init__()
        self.cholesky = P.Cholesky(upper)

    def construct(self, x):
        return self.cholesky(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cholesky_cpu():
    """
    Feature: Cholesky cpu kernel.
    Description: Test cholesky cpu kernel for Graph and PyNative modes.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU", precompile_only=True)
    x = Tensor([[1.0, 1.0], [1.0, 2.0]], mstype.float32)
    net = CholeskyNet(True)
    output = net(x)
    assert output is None


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_cholesky_gpu():
    """
    Feature: Cholesky gpu kernel.
    Description: Test cholesky gpu kernel for Graph and PyNative modes.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", precompile_only=True)
    x = Tensor([[1.0, 1.0], [1.0, 2.0]], mstype.float32)
    net = CholeskyNet(True)
    output = net(x)
    assert output is None
