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
from mindspore import nn, context, Tensor
from mindspore.ops.operations.math_ops import MatrixExp
from .test_grad_of_dynamic import TestDynamicGrad


class NetMatrixExp(nn.Cell):
    def __init__(self):
        super(NetMatrixExp, self).__init__()
        self.matrix_exp = MatrixExp()

    def construct(self, x):
        return self.matrix_exp(x)


def matrix_exp_test(is_dyn_rank):
    x = Tensor(np.array([[-1.0, 4.0], [2.0, -5.0]]).astype(np.float16))
    tester = TestDynamicGrad(NetMatrixExp())
    tester.test_dynamic_grad_net(x, is_dyn_rank)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.skip(reason="I69UYY")
def test_matrix_exp_dyn_shape():
    """
    Feature: MatrixExp Grad DynamicShape.
    Description: Test case of dynamic shape for MatrixExp grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    matrix_exp_test(False)
    context.set_context(mode=context.GRAPH_MODE)
    matrix_exp_test(False)


@pytest.mark.level2
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.skip(reason="I69UYY")
def test_matrix_exp_dyn_rank():
    """
    Feature: MatrixExp Grad DynamicRank.
    Description: Test case of dynamic rank for MatrixExp grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    matrix_exp_test(True)
    context.set_context(mode=context.GRAPH_MODE)
    matrix_exp_test(True)
