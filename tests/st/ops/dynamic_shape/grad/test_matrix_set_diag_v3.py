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
from mindspore import nn
from mindspore import Tensor, context
from mindspore.ops.operations.array_ops import MatrixSetDiagV3
from mindspore import dtype as mstype
from .test_grad_of_dynamic import TestDynamicGrad


class MatrixSetDiagV3Net(nn.Cell):
    def __init__(self):
        super(MatrixSetDiagV3Net, self).__init__()
        self.matrix_set_diag_v3 = MatrixSetDiagV3()

    def construct(self, x, diagonal, k):
        return self.matrix_set_diag_v3(x, diagonal, k)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(MatrixSetDiagV3Net())
    input_x = Tensor(np.array([[[5, 5, 5, 5],
                                [5, 5, 5, 5],
                                [5, 5, 5, 5]],
                               [[5, 5, 5, 5],
                                [5, 5, 5, 5],
                                [5, 5, 5, 5]]]), mstype.float32)
    diagonal = Tensor(np.array([[1, 2, 3],
                                [4, 5, 6]]), mstype.float32)
    k = Tensor(1, mstype.int32)
    test_dynamic.test_dynamic_grad_net([input_x, diagonal, k], is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_shape():
    """
    Feature: test MatrixSetDiagV3 dynamic shape on GPU, CPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(False)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_rank():
    """
    Feature: test MatrixSetDiagV3 dynamic rank on GPU, CPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(True)
