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
import mindspore as ms
from mindspore import mint, ops
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def histc_forward_func(x, bins, min_val, max_val):
    return mint.histc(x, bins, min_val, max_val)


@test_utils.run_with_cell
def histc_backward_func(x):
    # pylint: disable=E1102
    return ops.grad(histc_forward_func, (0))(x, 4, 0, 3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_histc_ext_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function histc forward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor([1, 2, 1], ms.int32)
    bins, min_val, max_val = 4, 0.0, 3.0
    output = histc_forward_func(x, bins, min_val, max_val)
    expected_output = np.array([0, 2, 1, 0])
    assert np.array_equal(output.asnumpy(), expected_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=20.0)
def test_histc_ext_dynamic_shape():
    """
    Feature: Test operator Histc by TEST_OP
    Description:  Test operator Histc with dynamic input
    Expectation: the result of Histc is correct.
    """
    x1 = generate_random_input((2, 3, 4, 5), np.int32)
    x2 = generate_random_input((3, 4, 5, 6, 7), np.int32)
    TEST_OP(histc_forward_func, [[ms.Tensor(x1), 4, 0.0, 3.0], [ms.Tensor(x2), 5, 1.0, 8.0]],
            case_config={'disable_grad': True}, disable_mode=["GRAPH_MODE_GE"])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_ops_histc_backward(mode):
    """
    Feature: pyboost function.
    Description: test histc backward.
    Expectation: success.
    """
    ms.set_context(mode=mode)
    test_cell = test_utils.to_cell_obj(histc_backward_func)
    x = ms.Tensor([2., 3., 4., 5.], ms.float32)
    expect_output = ms.Tensor([0., 0., 0., 0.], ms.float32)
    output = test_cell(x)
    error = np.array([1., 1., 1., 1.], np.float32) * 1.0e-6
    diff = output[0].asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)
