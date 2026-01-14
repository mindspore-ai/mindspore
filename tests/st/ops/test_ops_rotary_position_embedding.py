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
import numpy as np
import pytest
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import rotary_position_embedding


def generate_random_input(bound_value, shape):
    np.random.seed(1)
    return np.random.uniform(-bound_value, bound_value, size=shape).astype(np.float16)


def generate_expect_ouput():
    return np.asarray([[[[0.20144, 0.24207], [0.20079, -1.22971]],
                        [[-0.21777, 1.20316], [0.637207, 0.740154]]],
                       [[[0.094316, -0.113503], [0.176663, 0.183972]],
                        [[2.69531, -0.130479], [2.572266, 0.477868]]]]).astype(np.float32)


def generate_expect_grad_ouput():
    dx = np.asarray([[[[0.274658, 0.606689], [0.274658, 0.606689]],
                      [[-1.39526, 0.60474], [-1.39526, 0.60474]]],
                     [[[0.274658, 0.606689], [0.274658, 0.606689]],
                      [[-1.39526, 0.60474], [-1.39526, 0.60474]]]]).astype(np.float32)

    dcos = np.asarray([[[[-3.06812, 0.986814]], [[-5.7412, -0.054199]]]]).astype(np.float32)
    dsin = np.asarray([[[[-0.986814, -3.06812]], [[0.054199, -5.7412]]]]).astype(np.float32)
    return (dx, dcos, dsin)


def rotary_position_embedding_forward_func(input_x, input_cos, input_sin, mode):
    return rotary_position_embedding(input_x, input_cos, input_sin, mode)


def rotary_position_embedding_backward_func(input_x, input_cos, input_sin, mode):
    return ms.grad(rotary_position_embedding_forward_func, (0, 1, 2))(input_x, input_cos, input_sin, mode)


@pytest.mark.parametrize('ms_type', [mstype.float32, mstype.float16, mstype.bfloat16])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('mode_value', [0, 1])
def test_ops_rotary_position_embedding(context_mode, ms_type, mode_value):
    """
    Feature: pyboost function.
    Description: test function cross forward.
    Expectation: expect correct result.
    """
    np_x = generate_random_input(2, (2, 2, 2, 2))
    np_cos = generate_random_input(1, (1, 2, 1, 2))
    np_sin = generate_random_input(1, (1, 2, 1, 2))
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    ms_x = Tensor(np_x, dtype=ms_type)
    ms_cos = Tensor(np_cos, dtype=ms_type)
    ms_sin = Tensor(np_sin, dtype=ms_type)
    output = rotary_position_embedding_forward_func(ms_x, ms_cos, ms_sin, mode_value)
    output_grad = rotary_position_embedding_backward_func(ms_x, ms_cos, ms_sin, mode_value)
    expect_out = generate_expect_ouput()
    expect_grad_out = generate_expect_grad_ouput()
    rotl_value = 1e-3
    if ms_type == mstype.bfloat16:
        rotl_value = 2e-2
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=rotl_value)
    np.testing.assert_allclose(output_grad[0].asnumpy(), expect_grad_out[0], rtol=rotl_value)


@arg_mark(plat_marks=['platform_ascend910b'],
        level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=20.0)
def test_ops_rotary_position_embedding_dynamic():
    """
    Feature: pyboost function.
    Description: test function cross forward with dynamic shape.
    Expectation: expect correct result.
    """
    test_cell = test_utils.to_cell_obj(rotary_position_embedding_forward_func)
    np_x1 = generate_random_input(2, (2, 2, 2, 32))
    np_cos1 = generate_random_input(1, (1, 2, 1, 32))
    np_sin1 = generate_random_input(1, (1, 2, 1, 32))
    x1 = Tensor(np_x1, dtype=mstype.float32)
    cos1 = Tensor(np_cos1, dtype=mstype.float32)
    sin1 = Tensor(np_sin1, dtype=mstype.float32)

    np_x2 = generate_random_input(2, (4, 4, 4, 32))
    np_cos2 = generate_random_input(1, (1, 4, 1, 32))
    np_sin2 = generate_random_input(1, (1, 4, 1, 32))
    x2 = Tensor(np_x2, dtype=mstype.float32)
    cos2 = Tensor(np_cos2, dtype=mstype.float32)
    sin2 = Tensor(np_sin2, dtype=mstype.float32)

    TEST_OP(test_cell, [[x1, cos1, sin1, 0], [x2, cos2, sin2, 0]],
            disable_mode=["GRAPH_MODE_GE"],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True})
