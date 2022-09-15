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
import mindspore as ms
from mindspore import nn, context, Tensor
from mindspore.ops.operations.math_ops import MatrixSolve
from .test_grad_of_dynamic import TestDynamicGrad


class NetMatrixSolve(nn.Cell):
    def __init__(self):
        super(NetMatrixSolve, self).__init__()
        self.sol = MatrixSolve()

    def construct(self, matrix, rhs):
        return self.sol(matrix, rhs)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bprop_matrix_solve_dynamic_shape():
    """
    Features: ensure that matrix_solve can support [dynamic shape] while undergoing its gradient backprogation(bprop)
    Description: the test hides the complete shape info so that operations only infer the exact shape until runtime
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetMatrixSolve(), skip_convert_out_ids=[0])
    x = Tensor([[5, 4], [3, 1]], ms.float32)
    rhs = Tensor([[7], [2]], ms.float32)
    test_dynamic.test_dynamic_grad_net([x, rhs])


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bprop_matrix_solve_dynamic_rank():
    """
    Features: ensure that matrix_solve can support [dynamic rank] while undergoing its gradient backprogation(bprop)
    Description: the test hides the complete rank(the least amount of needed info when referring values in a tensor)
            information so that operations only infer the exact shape until runtime
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetMatrixSolve(), skip_convert_out_ids=[0])
    x = Tensor([[5, 4], [3, 1]], ms.float32)
    rhs = Tensor([[7], [2]], ms.float32)
    test_dynamic.test_dynamic_grad_net([x, rhs], True)
