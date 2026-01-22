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
from mindspore import ops, context, Tensor
from mindspore.graph.api import _pynative_executor

from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase


@test_utils.run_with_cell
def gcd_forward_func(x1, x2):
    return ops.gcd(x1, x2)


@test_utils.run_with_cell
def gcd_backward_func(x1, x2):
    return ms.grad(gcd_forward_func, (0, 1))(x1, x2)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (
        loss_count / total_count
    ) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater]
    )


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


def ops_gcd_binary_compare(input_binary_data, output_binary_data, loss, mode):
    if mode == "pynative":
        context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    elif mode == "GRAPH":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O2")
    input_x = Tensor(input_binary_data[0])
    other = Tensor(input_binary_data[1])

    output = gcd_forward_func(input_x, other)
    allclose_nparray(output.asnumpy(), output_binary_data[0], loss, loss)

    with pytest.raises(RuntimeError):
        gcd_backward_func(input_x, other)
        _pynative_executor.sync()


@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((1000, 1000), np.int32),
            ((1000, 1000), np.int32),
        ],
        output_info=[((1000, 1000), np.int32)],
        extra_info="auto_drive",
    )
)
def ops_gcd_binary_case1(input_binary_data=None, output_binary_data=None, loss=0, mode="pynative"):
    ops_gcd_binary_compare(input_binary_data, output_binary_data, loss, mode)


@arg_mark(
    plat_marks=["platform_ascend", "platform_gpu"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative", "KBK", "GRAPH"])
def test_ops_gcd_binary_cases(mode):
    """
    Feature: ops.gcd
    Description: Verify the result of gcd
    Expectation: success
    """
    ops_gcd_binary_case1(loss=0, mode=mode)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_gcd_normal_cpu(mode):
    """
    Feature: Ops.
    Description: test op gcd.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x1 = ms.Tensor(np.array([7, 8, 9]), ms.int32)
    x2 = ms.Tensor(np.array([14, 6, 12]), ms.int32)
    expect_out = np.array([7, 2, 3])
    out = gcd_forward_func(x1, x2)
    assert np.allclose(out.asnumpy(), expect_out)
    with pytest.raises(RuntimeError):
        gcd_backward_func(x1, x2)
        _pynative_executor.sync()


@arg_mark(
    plat_marks=["platform_ascend", "platform_gpu", "cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_gcd_vmap(mode):
    """
    Feature: test vmap function.
    Description: test gcd op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = -1
    x1 = ms.Tensor(np.array([7, 8, 9]), ms.int32)
    x2 = ms.Tensor(np.array([14, 6, 12]), ms.int32)
    nest_vmap = ops.vmap(gcd_forward_func, in_axes=in_axes, out_axes=0)
    nest_vmap(x1, x2)


@arg_mark(
    plat_marks=["platform_ascend", "platform_gpu", "cpu_linux"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_gcd_dynamic():
    """
    Feature: Test gcd with dynamic shape in graph mode using TEST_OP.
    Description: call ops.gcd with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((3, 3), np.int32)
    x2 = generate_random_input((3, 3), np.int32)
    y1 = generate_random_input((2, 2, 2), np.int32)
    y2 = generate_random_input((2, 2, 2), np.int32)
    TEST_OP(
        ops.gcd,
        [[ms.Tensor(x1), ms.Tensor(x2)], [ms.Tensor(y1), ms.Tensor(y2)]],
        case_config={'disable_grad': True,
                     'all_dim_zero': True},
    )
