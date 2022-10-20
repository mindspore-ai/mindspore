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


class NetTrace(nn.Cell):
    def __init__(self):
        super(NetTrace, self).__init__()
        self.trace = M.Trace()

    def construct(self, x):
        return self.trace(x)



@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_trace_dynamic_shape():
    """
    Feature: Trace Grad DynamicShape.
    Description: Test case of dynamic shape for Trace grad operator on CPU and GPU.
    Expectation: success.
    """
    for device in ['GPU', 'CPU']:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=device)
        test_dynamic = TestDynamicGrad(NetTrace())
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        test_dynamic.test_dynamic_grad_net(x, False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_trace_dynamic_shape_rank():
    """
    Feature: Trace Grad DynamicShape.
    Description: Test case of dynamic rank for Trace grad operator on CPU and GPU.
    Expectation: success.
    """
    for device in ['GPU', 'CPU']:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=device)
        test_dynamic = TestDynamicGrad(NetTrace())
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        test_dynamic.test_dynamic_grad_net(x, True)
