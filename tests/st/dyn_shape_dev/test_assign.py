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
import test_utils
import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops


@test_utils.run_with_cell
def assign_forward_func(x, y):
    return ops.auto_generate.assign(x, y)


@test_utils.run_with_cell
def assign_backward_func(x, y):
    return ops.grad(assign_forward_func, (0, 1))(x, y)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_assign_forward_cpu_gpu(mode):
    """
    Feature: assign ops.
    Description: test ops assign.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    variable = ms.Parameter(Tensor(np.array([1.0]).astype(np.float32)))
    value = Tensor(np.array([2.0]).astype(np.float32))
    output = assign_forward_func(variable, value)
    expect_output = np.asarray([2.0]).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_assign_forward_ascend(mode):
    """
    Feature: assign ops.
    Description: test ops assign.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    variable = ms.Parameter(Tensor(np.array([1.0]).astype(np.float32)))
    value = Tensor(np.array([2.0]).astype(np.float32))
    output = assign_forward_func(variable, value)
    print("output:", output)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_assign_backward(mode):
    """
    Feature: assign ops.
    Description: test auto grad of ops assign.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    variable = ms.Parameter(Tensor(np.array([1.0]).astype(np.float32)))
    value = Tensor(np.array([2.0]).astype(np.float32))
    dvariable, dvalue = assign_backward_func(variable, value)
    except_dvariable = np.asarray([1.]).astype(np.float32)
    except_dvalue = np.asarray([0.]).astype(np.float32)
    np.testing.assert_array_almost_equal(dvariable.asnumpy(), except_dvariable, decimal=4)
    np.testing.assert_array_almost_equal(dvalue.asnumpy(), except_dvalue, decimal=4)
