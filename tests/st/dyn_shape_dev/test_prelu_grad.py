# Copyright 2023 Huawei Technologies Co., Ltd
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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops
import test_utils


@test_utils.run_with_cell
def prelu_grad_func(y, x, weight):
    return ops.auto_generate.prelu_grad(y, x, weight)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_prelu_grad(mode):
    """
    Feature: prelu_grad ops.
    Description: test ops prelu_grad.
    Expectation: output right results.
    """
    context.set_context(mode=mode)
    dy = Tensor(np.array([[[-0.6, -0.5],
                           [-2.4, -1.8],
                           [0.6, 0.3]],
                          [[0., 1.],
                           [2., 3.],
                           [4., 5.]]]).astype(np.float32))
    x = Tensor(np.arange(-6, 6).reshape((2, 3, 2)).astype(np.float32))
    weight = Tensor(np.array([0.1, 0.6, -0.3]).astype(np.float32))
    output = prelu_grad_func(dy, x, weight)
