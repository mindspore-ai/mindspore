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
import mindspore
import mindspore.ops.operations.math_ops as M
from mindspore import nn, context, Tensor
from .test_grad_of_dynamic import TestDynamicGrad


class NetBatchMatMul(nn.Cell):
    def __init__(self):
        super(NetBatchMatMul, self).__init__()
        self.batchmatmul = M.BatchMatMul()

    def construct(self, x, y):
        return self.batchmatmul(x, y)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_batch_matmul_dynamic_shape():
    """
    Feature: BatchMatMul Grad DynamicShape.
    Description: Test case of dynamic shape for BatchMatMul grad operator on GPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetBatchMatMul(), skip_convert_out_ids=[0])
    x = Tensor(np.ones(shape=[2, 4, 1, 3]), mindspore.float32)
    y = Tensor(np.ones(shape=[2, 4, 3, 4]), mindspore.float32)
    inputs = [x, y]
    test_dynamic.test_dynamic_grad_net(inputs, False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_batch_matmul_dynamic_rank():
    """
    Feature: BatchMatMul Grad DynamicShape.
    Description: Test case of dynamic rank for BatchMatMul grad operator on GPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetBatchMatMul(), skip_convert_out_ids=[0])
    x = Tensor(np.ones(shape=[2, 4, 1, 3]), mindspore.float32)
    y = Tensor(np.ones(shape=[2, 4, 3, 4]), mindspore.float32)
    inputs = [x, y]
    test_dynamic.test_dynamic_grad_net(inputs, True)
