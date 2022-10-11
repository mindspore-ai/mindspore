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
from mindspore.ops.operations import _inner_ops as inner
from .test_grad_of_dynamic import TestDynamicGrad


class NetClipByNorm(nn.Cell):
    def __init__(self, axis):
        super(NetClipByNorm, self).__init__()
        self.op = inner.ClipByNorm(axis)

    def construct(self, x, y):
        return self.op(x, y)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(NetClipByNorm(0))
    np.random.seed(1)
    x = Tensor(np.random.rand(3, 16, 5, 4).astype(np.float32))
    y = Tensor(np.array([0.01]).astype(np.float32))
    test_dynamic.test_dynamic_grad_net([x, y], is_dynamic_rank)


@pytest.mark.skip(reason="Dependent on [ClipByNorm infer shape/CPU kernel dynamic shape] and [_dyn_reduced_shape]")
@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_dynamic_shape():
    """
    Feature: test ClipByNorm dynamic shape.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(False)


@pytest.mark.skip(reason="Dependent on [ClipByNorm infer shape/CPU kernel dynamic shape] and [_dyn_reduced_shape]")
@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_grad_dynamic_rank():
    """
    Feature: test ClipByNorm dynamic rank.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(True)
