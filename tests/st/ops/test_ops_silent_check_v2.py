# Copyright 2024 Huawei Technologies Co., Ltd
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
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops


def silent_check_v2(val, input_grad, sfda, step, c_min_steps=7, c_thresh_l1=1000000.,
                    c_coeff_l1=100000., c_thresh_l2=10000., c_coeff_l2=5000., npu_asd_detect=1):
    op = ops.auto_generate.silent_check_v2_op
    return op(val, input_grad, sfda, step, c_min_steps, c_thresh_l1,
              c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect)


@test_utils.run_with_cell
def silent_check_v2_forward_func(val, input_grad, sfda, step, c_min_steps=7,
                                 c_thresh_l1=1000000., c_coeff_l1=100000.,
                                 c_thresh_l2=10000., c_coeff_l2=5000., npu_asd_detect=1):
    return silent_check_v2(val, input_grad, sfda, step, c_min_steps, c_thresh_l1,
                           c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect)


def set_mode(mode):
    """
    set_mode
    """
    if mode == "ge":
        context.set_context(mode=context.GRAPH_MODE, jit_config={'jit_level': 'O2'})
    elif mode == "kbk":
        context.set_context(mode=context.GRAPH_MODE, jit_config={'jit_level': 'O0'})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


def generate_input_tensors():
    """
    generate_input_tensors
    """
    val = Tensor(np.random.rand(), ms.float32)
    input_grad = Tensor(np.random.rand(2, 5).astype(np.float32))
    sfda = Tensor(np.random.rand(3).astype(np.float32))
    step = Tensor(np.random.randint(1, 10, size=[1]), ms.int64)
    return val, input_grad, sfda, step


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("mode", ["ge", "kbk", "pyboost"])
def test_silent_check_v2_static_shape(mode):
    """
    Feature: SilentCheckV2.
    Description: test op SilentCheckV2.
    Expectation: expect correct result.
    """
    set_mode(mode)
    val, input_grad, sfda, step = generate_input_tensors()
    outs = silent_check_v2_forward_func(val, input_grad, sfda, step)
    print(f"before input_grad:\n{input_grad}\nsfda:\n{sfda}\nstep:\n{step}.")
    print(f"after silent check, input_grad:\n{outs[0]}\nsfda:\n{outs[1]}"
          f"\nstep:\n{outs[2]}\nresult:\n{outs[3]}.")


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
def test_silent_check_v2_dyn_shape():
    """
    Feature: SilentCheckV2.
    Description: test op SilentCheckV2.
    Expectation: expect correct result.
    """
    context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.
    val, input_grad, sfda, step = generate_input_tensors()
    input_case1 = [val, input_grad, sfda, step, 7, 1000000., 100000., 10000., 5000., 1]
    val, input_grad, sfda, step = generate_input_tensors()
    input_case2 = [val, input_grad, sfda, step, 3, 100000., 50000., 7000., 2500., 3]
    TEST_OP(
        silent_check_v2_forward_func,
        [
            input_case1,
            input_case2,
        ],
        "silent_check_v2",
        disable_input_check=True,
        disable_grad=True,
        inplace_update=True
    )
