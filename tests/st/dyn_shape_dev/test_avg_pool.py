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
import pytest
import test_utils

from mindspore import ops
import mindspore as ms
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

@test_utils.run_with_cell
def avg_pool_forward_func(x):
    return ops.auto_generate.avg_pool(x, kernel_size=2, strides=2, pad_mode="VALID", data_format="NCHW")


@test_utils.run_with_cell
def avg_pool_backward_func(x):
    return ops.grad(avg_pool_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_avg_pool_forward(mode):
    """
    Feature: Ops.
    Description: test op avg pool.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    ms.context.set_context(precompile_only=True)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    out = avg_pool_forward_func(x)
    print("out:", out)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_avg_pool_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op avg pool.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    ms.context.set_context(precompile_only=True)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    grads = avg_pool_backward_func(x)
    print("grads:", grads)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_avg_pool_vmap(mode):
    """
    Feature: test vmap function.
    Description: test avgpool op vmap.
    Expectation: expect correct result.
    """
    in_axes = -1
    ms.context.set_context(mode=mode)
    ms.context.set_context(precompile_only=True)
    x = ms.Tensor(np.random.randn(1, 1, 6, 6, 3, 6).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(avg_pool_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    print("out:", out)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_avg_pool_dynamic():
    """
    Feature: test dynamic tensor and dynamic scalar of avg pool.
    Description: test dynamic tensor and dynamic scalar of avg pool.
    Expectation: expect correct result.
    """
    in1 = [ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32), 2, 2, "VALID", "NCHW"]
    in2 = [ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32), 2, 2, "VALID", "NCHW"]
    TEST_OP(ops.auto_generate.avg_pool, [in1, in2], dump_ir=True, grad=False)
