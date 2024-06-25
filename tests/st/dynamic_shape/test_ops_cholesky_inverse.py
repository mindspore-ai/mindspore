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
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import auto_generate as P
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark


class CholeskyInverseNet(nn.Cell):
    def __init__(self, upper):
        super(CholeskyInverseNet, self).__init__()
        self.cholesky_inverse = P.CholeskyInverse(upper)

    def construct(self, x):
        return self.cholesky_inverse(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_cholesky_inverse_cpu():
    """
    Feature: Test cholesky_inverse cpu kernel.
    Description: Test cholesky_inverse cpu kernel for Graph and PyNative modes.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = Tensor([[1., 1.], [1., 2.]], mstype.float32)
    net = CholeskyInverseNet(True)
    output = net(x)
    expect = np.array([[5., -3.], [-3., 2.]], np.float32)
    error = 1e-3
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=error)
