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
import pytest
import numpy as np
from mindspore import Tensor, context
from mindspore.ops import auto_generate as P
import test_utils


@test_utils.run_with_cell
def celu_forward_func(x, alpha):
    return P.CeLU(alpha)(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_celu_op_cpu(data_type):
    """
    Feature: Celu cpu kernel
    Description: test the celu alpha = 1.0.
    Expectation: match to np benchmark.
    """
    x = Tensor(np.array([-2.0, -1.0, 1.0, 2.0]).astype(data_type))
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU', precompile_only=True)
    output = celu_forward_func(x, 1.)
    print(output)
    assert output is None
