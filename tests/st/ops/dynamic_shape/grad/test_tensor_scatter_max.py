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
from mindspore import ops, nn, context, Tensor
from .test_grad_of_dynamic import TestDynamicGrad


class TestMax(nn.Cell):
    def __init__(self):
        super(TestMax, self).__init__()
        self.ops = ops.TensorScatterMax()

    def construct(self, input_x, indices, updates):
        return self.ops(input_x, indices, updates)


class TestMin(nn.Cell):
    def __init__(self):
        super(TestMin, self).__init__()
        self.ops = ops.TensorScatterMin()

    def construct(self, input_x, indices, updates):
        return self.ops(input_x, indices, updates)


def tensor_scatter_max_min_dynamic_shape(is_dyn_rank=False):
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
    indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
    updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
    x = [input_x, indices, updates]
    test_dynamic_max = TestDynamicGrad(TestMax())
    test_dynamic_max.test_dynamic_grad_net(x, is_dyn_rank)

    test_dynamic_min = TestDynamicGrad(TestMin())
    test_dynamic_min.test_dynamic_grad_net(x, is_dyn_rank)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_tensor_scatter_max_min_dynamic_shape():
    """
    Feature: TensorScatterMax/Min Grad DynamicShape.
    Description: Test case of dynamic shape for TensorScatterMax/Min grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    tensor_scatter_max_min_dynamic_shape()


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_tensor_scatter_max_min_dynamic_rank():
    """
    Feature: TensorScatterMax/Min Grad DynamicShape.
    Description: Test case of dynamic rank for TensorScatterMax/Min grad operator.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    tensor_scatter_max_min_dynamic_shape(True)
