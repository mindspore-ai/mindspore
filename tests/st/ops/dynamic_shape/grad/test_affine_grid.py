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
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops.operations.array_ops import AffineGrid
from .test_grad_of_dynamic import TestDynamicGrad


class AffineGridNet(nn.Cell):
    def __init__(self):
        super(AffineGridNet, self).__init__()
        self.affine_grid = AffineGrid()

    def construct(self, theta, output_size):
        return self.affine_grid(theta, output_size)


def dynamic_shape():
    test_dynamic = TestDynamicGrad(AffineGridNet())
    theta_4d = Tensor(np.array([[[0.8, 0.5, 0], [-0.5, 0.8, 0]]]).astype(np.float32))
    output_size_4d = Tensor(np.array([1, 3, 2, 3]).astype(np.int32))
    test_dynamic.test_dynamic_grad_net((theta_4d, output_size_4d), is_save_graphs=True)

    theta_5d = Tensor(np.array([[[0.8, 0.5, 0, 1], [-0.5, 0.8, 0, 1], [-0.2, 0.9, 0, 2]]]).astype(np.float32))
    output_size_5d = Tensor(np.array([1, 3, 2, 3, 2]).astype(np.int32))
    test_dynamic.test_dynamic_grad_net((theta_5d, output_size_5d), is_save_graphs=True)


def dynamic_rank():
    test_dynamic = TestDynamicGrad(AffineGridNet())
    theta_4d = Tensor(np.array([[[0.8, 0.5, 0], [-0.5, 0.8, 0]]]).astype(np.float32))
    output_size_4d = Tensor(np.array([1, 3, 2, 3]).astype(np.int32))
    test_dynamic.test_dynamic_grad_net((theta_4d, output_size_4d), True, is_save_graphs=True)

    theta_5d = Tensor(np.array([[[0.8, 0.5, 0, 1], [-0.5, 0.8, 0, 1], [-0.2, 0.9, 0, 2]]]).astype(np.float32))
    output_size_5d = Tensor(np.array([1, 3, 2, 3, 2]).astype(np.int32))
    test_dynamic.test_dynamic_grad_net((theta_5d, output_size_5d), True)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_dynamic_affine_grid_cpu():
    """
    Feature: AffineGrid Grad DynamicShape.
    Description: Test case of dynamic shape for AffineGrid grad operator on CPU.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dynamic_shape()
    dynamic_rank()
    # PyNative mode

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    dynamic_shape()
    dynamic_rank()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
def test_dynamic_affine_grid_gpu():
    """
    Feature: AffineGrid Grad DynamicShape.
    Description: Test case of dynamic shape for AffineGrid grad operator on GPU.
    Expectation: success.
    """
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dynamic_shape()
    dynamic_rank()
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dynamic_shape()
    dynamic_rank()
