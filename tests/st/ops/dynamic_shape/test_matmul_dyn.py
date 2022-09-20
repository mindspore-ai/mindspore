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

"""test matmul dynamic shape"""

import numpy as np
import pytest

from mindspore import dtype as mstype
from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P


class MatMulNet(nn.Cell):
    def __init__(self, transpose_a=False, transpose_b=False):
        super(MatMulNet, self).__init__()
        self.matmul = P.MatMul(transpose_a, transpose_b)

    def construct(self, x, y):
        return self.matmul(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('trans_a', [True, False])
@pytest.mark.parametrize('trans_b', [True, False])
def test_matmul_matrix(trans_a, trans_b):
    """
    Feature: test matmul op.
    Description: test the ops in dynamic shape.
    Expectation: expect correct output shape.
    """
    m, k, n = 5, 3, 4
    a = np.random.random((m, k)).astype(np.float32)
    b = np.random.random((k, n)).astype(np.float32)
    if trans_a:
        a = a.T
    if trans_b:
        b = b.T
    a_dyn = Tensor(shape=(None, None), dtype=mstype.float32)
    b_dyn = Tensor(shape=(None, None), dtype=mstype.float32)
    expected_output_shape = (5, 4)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = MatMulNet(transpose_a=trans_a, transpose_b=trans_b)
    net.set_inputs(a_dyn, b_dyn)
    output = net(Tensor(a), Tensor(b))

    assert output.asnumpy().shape == expected_output_shape
