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
from mindspore import ops

from .test_grad_of_dynamic import TestDynamicGrad


class NetArgMaxWithValue(nn.Cell):
    def __init__(self, keep_dims=False, axis=0):
        super(NetArgMaxWithValue, self).__init__()
        self.op = ops.ArgMaxWithValue(keep_dims=keep_dims, axis=axis)

    def construct(self, x):
        return self.op(x)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_shape_arg_max_with_value():
    """
    Feature: ArgMaxWithValue Grad DynamicShape.
    Description: Test case of dynamic shape for ArgMaxWithValue grad.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetArgMaxWithValue(False, -1))
    x = Tensor(np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)).astype(np.float16))
    test_dynamic.test_dynamic_grad_net(x)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_rank_arg_max_with_value():
    """
    Feature: ArgMaxWithValue Grad DynamicRank.
    Description: Test case of dynamic rank for ArgMaxWithValue grad.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetArgMaxWithValue(True, 3))
    x = Tensor(np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)).astype(np.float16))
    test_dynamic.test_dynamic_grad_net(x, True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_rank_arg_max_with_value_neg_axis():
    """
    Feature: ArgMaxWithValue Grad DynamicRank.
    Description: Test case of dynamic rank for ArgMaxWithValue grad.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetArgMaxWithValue(True, -2))
    x = Tensor(np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)).astype(np.float16))
    test_dynamic.test_dynamic_grad_net(x, True)
