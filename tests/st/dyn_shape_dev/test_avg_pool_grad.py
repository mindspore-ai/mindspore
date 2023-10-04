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
import numpy as np
import mindspore as ms
from mindspore.ops import auto_generate as ops
import test_utils
import pytest


@test_utils.run_with_cell
def avg_pool_grad_forward_func(x, out, dout):
    return ops.AvgPoolGrad(kernel_size=2, strides=1, pad_mode="VALID", data_format="NCHW")(x, out, dout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
@pytest.mark.skip(reason="Not ready")
def test_avg_pool_grad(mode):
    """
    Feature: DynamicShape.
    Description: Create AvgPoolGrad instance with constant arguaments.
    Expectation: No exception.
    """
    ms.context.set_context(precompile_only=True)
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    out = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    dout = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    out = avg_pool_grad_forward_func(x, out, dout)
    print("out:", out)
