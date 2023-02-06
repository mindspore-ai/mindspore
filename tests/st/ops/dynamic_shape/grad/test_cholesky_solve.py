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
from mindspore import nn, Tensor, context
from mindspore.common import dtype as mstype
from mindspore.ops.operations.math_ops import CholeskySolve
from .test_grad_of_dynamic import TestDynamicGrad


class CholeskySolveNet(nn.Cell):
    def __init__(self):
        super(CholeskySolveNet, self).__init__()
        self.cholesky = CholeskySolve()

    def construct(self, x1, x2):
        return self.cholesky(x1, x2)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(CholeskySolveNet())
    x1 = Tensor(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), mstype.float32)
    x2 = Tensor(np.array([[2, 0, 0], [4, 1, 0], [-1, 1, 2]]), mstype.float32)
    test_dynamic.test_dynamic_grad_net([x1, x2], is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_choleskysolve_dynamic_shape():
    """
    Feature: Test CholeskySolve on CPU.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    grad_dyn_case(False)


@pytest.mark.skip(reason="dynamic rank feature is under developing.")
@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_choleskysolve_dynamic_rank():
    """
    Feature: Test CholeskySolve on CPU.
    Description:  The rank of inputs is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    grad_dyn_case(True)
