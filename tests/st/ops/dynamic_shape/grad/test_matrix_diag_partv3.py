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
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from .test_grad_of_dynamic import TestDynamicGrad


class MatrixDiagPartV3Net(nn.Cell):
    def __init__(self, align='LEFT_RIGHT'):
        super(MatrixDiagPartV3Net, self).__init__()
        self.matrix_diag_dart_v3 = P.array_ops.MatrixDiagPartV3(align=align)

    def construct(self, x, k, padding_value):
        return self.matrix_diag_dart_v3(x, k, padding_value)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_dynamic_shape_matrix_diag_partv3():
    """
    Feature: MatrixDiagPartV3 Grad DynamicShape.
    Description: Test case of dynamic shape for MatrixDiagPartV3 grad operator on CPU and GPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    align = 'RIGHT_LEFT'
    test_dynamic = TestDynamicGrad(MatrixDiagPartV3Net(align))
    input_x = Tensor(np.array([[[1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 8, 7, 6]],
                               [[5, 4, 3, 2],
                                [1, 2, 3, 4],
                                [5, 6, 7, 8]]]), mstype.float32)
    k = Tensor(1, mstype.int32)
    padding_value = Tensor(0, mstype.float32)
    test_dynamic.test_dynamic_grad_net([input_x, k, padding_value], False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_dynamic_rank_matrix_diag_partv3():
    """
    Feature: MatrixDiagPartV3 Grad DynamicShape.
    Description: Test case of dynamic rank for MatrixDiagPartV3 grad operator on CPU and GPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    align = 'RIGHT_LEFT'
    test_dynamic = TestDynamicGrad(MatrixDiagPartV3Net(align))
    input_x = Tensor(np.array([[[1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 8, 7, 6]],
                               [[5, 4, 3, 2],
                                [1, 2, 3, 4],
                                [5, 6, 7, 8]]]), mstype.float32)
    k = Tensor(1, mstype.int32)
    padding_value = Tensor(0, mstype.float32)
    test_dynamic.test_dynamic_grad_net([input_x, k, padding_value], True)
