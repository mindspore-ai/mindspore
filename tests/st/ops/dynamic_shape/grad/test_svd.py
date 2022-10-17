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
from mindspore.ops.operations import linalg_ops
from .test_grad_of_dynamic import TestDynamicGrad


class SvdNet(nn.Cell):
    def __init__(self):
        super(SvdNet, self).__init__()
        self.svd = linalg_ops.Svd(full_matrices=False, compute_uv=False)

    def construct(self, a):
        s, _, _ = self.svd(a)
        return s


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(SvdNet())
    np.random.seed(1)
    a = np.random.rand(3, 2).astype(np.float32)
    tensor_a = Tensor(a)
    test_dynamic.test_dynamic_grad_net(tensor_a, is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grad_dynamic_shape():
    """
    Feature: test Svd grad dynamic shape.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(False)


@pytest.mark.skip(reason="MatrixDiagV3在GRAPH_MODE下不支持`num_rows`动态shape")
@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grad_dynamic_shape_2():
    """
    Feature: test Svd grad dynamic shape.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.GRAPH_MODE)
    grad_dyn_case(False)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grad_dynamic_rank():
    """
    Feature: test Svd grad dynamic rank.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(True)
    context.set_context(mode=context.GRAPH_MODE)
    grad_dyn_case(True)
