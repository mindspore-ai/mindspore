# Copyright 2020 Huawei Technologies Co., Ltd
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
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from mindspore import ops, Tensor
from mindspore.ops.function.math_func import linspace_ext
import mindspore as ms
from mindspore.common import mutable

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

@test_utils.run_with_cell
def lin_space_ext_forward_func(start, end, steps, dtype=None):
    return linspace_ext(start, end, steps, dtype=dtype)

@test_utils.run_with_cell
def lin_space_ext_backward_func(start, end, steps, dtype=None):
    return ops.grad(lin_space_ext_forward_func, (0, 1))(start, end, steps, dtype)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.float32])
def test_lin_space_ext_normal(mode, dtype):
    """
    Feature: Ops.
    Description: test op LinSpaceExt forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    ms.set_context(jit_level='O0')
    ## forward
    start_scalar, end_scalar, steps_scalar = 5, 25, 5
    start_tensor, end_tensor, steps_tensor = ms.Tensor(start_scalar), ms.Tensor(end_scalar), ms.Tensor(steps_scalar)
    output1 = lin_space_ext_forward_func(start_scalar, end_scalar, steps_scalar, dtype)
    expect1 = np.linspace(start_scalar, end_scalar, steps_scalar, axis=-1)
    assert np.allclose(output1.asnumpy(), expect1)
    output2 = lin_space_ext_forward_func(start_tensor, end_tensor, steps_tensor, dtype)
    expect2 = np.linspace(start_scalar, end_scalar, steps_scalar, axis=-1)
    assert np.allclose(output2.asnumpy(), expect2)

    start_scalar, end_scalar, steps_scalar = 1.0, 25.0, 20
    start_tensor, end_tensor, steps_tensor = ms.Tensor(start_scalar), ms.Tensor(end_scalar), ms.Tensor(steps_scalar)
    dtype = ms.float32
    output3 = lin_space_ext_forward_func(start_scalar, end_scalar, steps_scalar, dtype)
    expect3 = np.linspace(start_scalar, end_scalar, steps_scalar, axis=-1)
    assert np.allclose(output3.asnumpy(), expect3)
    output4 = lin_space_ext_forward_func(start_tensor, end_tensor, steps_tensor, dtype)
    expect4 = np.linspace(start_scalar, end_scalar, steps_scalar, axis=-1)
    assert np.allclose(output4.asnumpy(), expect4)

    start_scalar, end_scalar, steps_scalar = 5.0, 250, 14
    start_tensor, end_tensor, steps_tensor = ms.Tensor(start_scalar), ms.Tensor(end_scalar), ms.Tensor(steps_scalar)
    dtype = ms.float32
    output5 = lin_space_ext_forward_func(start_scalar, end_scalar, steps_scalar, dtype)
    expect5 = np.linspace(start_scalar, end_scalar, steps_scalar, axis=-1)
    assert np.allclose(output5.asnumpy(), expect5)
    output6 = lin_space_ext_forward_func(start_tensor, end_tensor, steps_tensor, dtype)
    expect6 = np.linspace(start_scalar, end_scalar, steps_scalar, axis=-1)
    assert np.allclose(output6.asnumpy(), expect6)

    ## backward
    start, end, steps = -115, 251, 101
    dtype = ms.float32
    grads = lin_space_ext_backward_func(ms.Tensor(start, ms.float32), ms.Tensor(end, ms.float32), steps, dtype)
    grads_ = [out.asnumpy() for out in grads]
    expect = [0, 0]
    assert np.allclose(grads_, expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.bfloat16])
def test_lin_space_ext_bfloat16(mode, dtype):
    """
    Feature: Ops.
    Description: test op LinSpaceExt.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    ms.set_context(jit_level='O0')

    start_scalar, end_scalar, steps_scalar = 5, 25, 5
    start_tensor, end_tensor, steps_tensor = ms.Tensor(start_scalar), ms.Tensor(end_scalar), ms.Tensor(steps_scalar)
    output1 = lin_space_ext_forward_func(start_scalar, end_scalar, steps_scalar, dtype)
    expect1 = np.linspace(start_scalar, end_scalar, steps_scalar, axis=-1)
    assert np.allclose(output1.float().asnumpy(), expect1)
    output2 = lin_space_ext_forward_func(start_tensor, end_tensor, steps_tensor, dtype)
    expect2 = np.linspace(start_scalar, end_scalar, steps_scalar, axis=-1)
    assert np.allclose(output2.float().asnumpy(), expect2)

    start_scalar, end_scalar, steps_scalar = 1.0, 25.0, 20
    start_tensor, end_tensor, steps_tensor = ms.Tensor(start_scalar), ms.Tensor(end_scalar), ms.Tensor(steps_scalar)
    dtype = ms.float32
    output3 = lin_space_ext_forward_func(start_scalar, end_scalar, steps_scalar, dtype)
    expect3 = np.linspace(start_scalar, end_scalar, steps_scalar, axis=-1)
    assert np.allclose(output3.float().asnumpy(), expect3)
    output4 = lin_space_ext_forward_func(start_tensor, end_tensor, steps_tensor, dtype)
    expect4 = np.linspace(start_scalar, end_scalar, steps_scalar, axis=-1)
    assert np.allclose(output4.float().asnumpy(), expect4)

    start_scalar, end_scalar, steps_scalar = 5.0, 250, 14
    start_tensor, end_tensor, steps_tensor = ms.Tensor(start_scalar), ms.Tensor(end_scalar), ms.Tensor(steps_scalar)
    dtype = ms.float32
    output5 = lin_space_ext_forward_func(start_scalar, end_scalar, steps_scalar, dtype)
    expect5 = np.linspace(start_scalar, end_scalar, steps_scalar, axis=-1)
    assert np.allclose(output5.float().asnumpy(), expect5)
    output6 = lin_space_ext_forward_func(start_tensor, end_tensor, steps_tensor, dtype)
    expect6 = np.linspace(start_scalar, end_scalar, steps_scalar, axis=-1)
    assert np.allclose(output6.float().asnumpy(), expect6)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
def test_lin_space_ext_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op concat.
    Expectation: expect tile result.
    """
    input_case1 = (Tensor([5]), Tensor([23]), Tensor([5]))
    input_case2 = (Tensor([-4]), Tensor([40]), Tensor([6]))
    TEST_OP(lin_space_ext_forward_func, [[*input_case1], [*input_case2]], '', disable_yaml_check=True,
            disable_input_check=True, disable_mode=['GRAPH_MODE'], disable_nontensor_dynamic_type='BOTH')

    input_case3 = (5, 50.23, mutable(5), ms.int32)
    input_case4 = (-5, 43.97, mutable(13), ms.float32)
    TEST_OP(lin_space_ext_forward_func, [[*input_case3], [*input_case4]], '', disable_yaml_check=True,
            disable_input_check=True, disable_mode=['GRAPH_MODE'], disable_nontensor_dynamic_type='BOTH',
            disable_resize=True)
