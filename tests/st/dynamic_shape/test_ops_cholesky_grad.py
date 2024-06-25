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
from tests.st.utils import test_utils
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
from mindspore.common import dtype as mstype
from mindspore.ops import auto_generate as P
from tests.mark_utils import arg_mark


class CholeskyGradNet(nn.Cell):
    def __init__(self):
        super(CholeskyGradNet, self).__init__()
        self.cholesky_grad = P.CholeskyGrad()

    def construct(self, x, dx):
        return self.cholesky_grad(x, dx)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@test_utils.run_test_with_On
def test_cholesky_grad_ascend():
    """
    Feature: Cholesky ascend kernel.
    Description: Test cholesky ascend kernel for Graph modes.
    Expectation: the result match with expected result.
    """
    expect = np.array([[0.5, 0.0], [0.0, 0.5]]).astype(np.float32)
    error = 1e-3
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = Tensor([[4.0, 2.0], [2.0, 3.0]], mstype.float32)
    dx = Tensor([[4.0, 2.0], [2.0, 3.0]], mstype.float32)
    net = CholeskyGradNet()
    output = net(x, dx)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=error)
