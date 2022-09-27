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
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops.operations.array_ops import MatrixDiagV3
from .test_grad_of_dynamic import TestDynamicGrad


class MatrixDiagV3Net(nn.Cell):
    def __init__(self):
        super(MatrixDiagV3Net, self).__init__()
        self.matrix_diag_v3 = MatrixDiagV3(align='LEFT_RIGHT')

    def construct(self, x, k, num_rows, num_cols, padding_value):
        return self.matrix_diag_v3(x, k, num_rows, num_cols, padding_value)


def run_dynamic_shape():
    test_dynamic = TestDynamicGrad(MatrixDiagV3Net())
    x = Tensor(np.array([[8, 9, 0], [1, 2, 3], [0, 4, 5]]), ms.float32)
    k = Tensor(np.array([-1, 1]), ms.int32)
    num_rows = Tensor(np.array(3), ms.int32)
    num_cols = Tensor(np.array(3), ms.int32)
    padding_value = Tensor(np.array(11), ms.float32)
    test_dynamic.test_dynamic_grad_net(
        [x, k, num_rows, num_cols, padding_value])


def run_dynamic_rank():
    test_dynamic = TestDynamicGrad(MatrixDiagV3Net())
    x = Tensor(np.array([[8, 9, 0],
                         [1, 2, 3],
                         [0, 4, 5]]), ms.float32)
    k = Tensor(np.array([-1, 1]), ms.int32)
    num_rows = Tensor(np.array(3), ms.int32)
    num_cols = Tensor(np.array(3), ms.int32)
    padding_value = Tensor(np.array(11), ms.float32)
    test_dynamic.test_dynamic_grad_net(
        [x, k, num_rows, num_cols, padding_value], True)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_matrix_diag_v3_cpu():
    """
    Feature: MatrixDiagV3 Grad DynamicShape.
    Description: Test case of dynamic shape for  MatrixDiagV3 grad operator on CPU.
    Expectation: success.
    """
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    run_dynamic_shape()
    run_dynamic_rank()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_matrix_diag_v3_gpu():
    """
    Feature: MatrixDiagV3 Grad DynamicShape.
    Description: Test case of dynamic shape for  MatrixDiagV3 grad operator on GPU.
    Expectation: success.
    """
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    run_dynamic_shape()
    run_dynamic_rank()
