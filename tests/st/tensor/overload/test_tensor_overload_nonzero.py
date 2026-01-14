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
import mindspore as ms
from mindspore import ops
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    x = np.random.randn(*shape).astype(dtype)
    return np.where(x > 0.4, x, 0)


def generate_expect_forward_output(x, as_tuple=False):
    if as_tuple:
        return np.nonzero(x)
    return np.transpose(np.nonzero(x))


@test_utils.run_with_cell
def nonzero_forward_func_1(x):
    return x.nonzero()


@test_utils.run_with_cell
def nonzero_forward_func_2(x, as_tuple_=False):
    return x.nonzero(as_tuple=as_tuple_)


@test_utils.run_with_cell
def nonzero_astuple_false_forward_func(x):
    return x.nonzero()


@test_utils.run_with_cell
def nonzero_astuple_false_forward_2_func(x):
    return x.nonzero(as_tuple=False)


@test_utils.run_with_cell
def nonzero_astuple_true_forward_func(x):
    return x.nonzero(as_tuple=True)


@test_utils.run_with_cell
def nonzero_backward_func(x, as_tuple=False):
    if as_tuple:
        grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
        return grad_op(nonzero_forward_func_2)(x, as_tuple)
    return ms.grad(nonzero_forward_func_1, (0))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('as_tuple', [True, False])
@test_utils.run_test_with_On
def test_ops_nonzero_normal(context_mode, as_tuple):
    """
    Feature: pyboost function.
    Description: test function nonzero forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.context.set_context(jit_config={"jit_level": "O0"})
    x = generate_random_input((2, 3, 4, 5), np.float32)
    if not as_tuple:
        output = nonzero_forward_func_2(ms.Tensor(x), as_tuple)
        output_1 = nonzero_forward_func_1(ms.Tensor(x))
        expect = generate_expect_forward_output(x, as_tuple)
        np.testing.assert_array_equal(output.asnumpy(), expect)
        np.testing.assert_array_equal(output_1.asnumpy(), expect)

    if as_tuple and ms.get_context(attr_key='device_target') == 'Ascend':
        output_tuple = nonzero_forward_func_2(ms.Tensor(x), as_tuple)
        expect_tuple = generate_expect_forward_output(x, as_tuple)
        for expect, output in zip(expect_tuple, output_tuple):
            np.testing.assert_array_equal(output.asnumpy(), expect)

    x = np.array([1, 2, 2, 4, 3]).astype(np.float32)
    if not as_tuple:
        output = nonzero_backward_func(ms.Tensor(x), as_tuple)
        expect = np.array([0., 0., 0., 0., 0.]).astype(np.float32)
        np.testing.assert_array_equal(output.asnumpy(), expect)

    if as_tuple and ms.get_context(attr_key='device_target') == 'Ascend':
        output_tuple = nonzero_backward_func(ms.Tensor(x), as_tuple)
        expect_tuple = np.array([[0, 0, 0, 0, 0]])
        for expect, output in zip(expect_tuple, output_tuple):
            np.testing.assert_array_equal(output.asnumpy(), expect)

    with pytest.raises(TypeError):
        ms.Tensor(x).nonzero(False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('as_tuple', [True, False])
@test_utils.run_test_with_On
def test_ops_nonzero_bf16(context_mode, as_tuple):
    """
    Feature: pyboost function.
    Description: test function nonzero forward(bf16).
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_tensor = ms.Tensor([1.58, 2.64, 9.34, 0.00], dtype=ms.bfloat16)
    if ms.get_context(attr_key='device_target') != 'Ascend':
        pytest.skip("cpu and gpu not support bfloat16!")

    if not as_tuple:
        output = nonzero_forward_func_2(x_tensor, as_tuple)
        output_1 = nonzero_forward_func_1(x_tensor)
        expect = np.array([[0], [1], [2]])
        np.testing.assert_array_equal(output.asnumpy(), expect)
        np.testing.assert_array_equal(output_1.asnumpy(), expect)
    else:
        output_tuple = nonzero_forward_func_2(x_tensor, as_tuple)
        expect_tuple = np.array([[0, 1, 2]])
        for expect, output in zip(expect_tuple, output_tuple):
            np.testing.assert_array_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
def test_nonzero_astuple_false_dy_shape():
    """
    Feature: Test dynamic shape.
    Description: test function nonzero as_tuple=false dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 4, 5), np.float32)
    ms_data2 = generate_random_input((3, 4, 5, 6, 7), np.float32)
    TEST_OP(nonzero_astuple_false_forward_func,
            [[ms.Tensor(ms_data1)], [ms.Tensor(ms_data2)]],
            disable_case=['EmptyTensor', 'ScalarTensor'])

    TEST_OP(nonzero_astuple_false_forward_2_func,
            [[ms.Tensor(ms_data1)], [ms.Tensor(ms_data2)]],
            disable_case=['EmptyTensor', 'ScalarTensor'])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.0)
def test_nonzero_astuple_true_dy_shape():
    """
    Feature: Test dynamic shape.
    Description: test function nonzero as_tuple = true dynamic feature,
                 only ascend PYNATIVE_MODE.
    Expectation: expect correct result.
    """
    if ms.get_context(attr_key='device_target') != 'Ascend':
        pytest.skip("cpu and gpu not support as_tuple=True")
    ms_data1 = generate_random_input((2, 3, 4, 5), np.float32)
    ms_data2 = generate_random_input((3, 4, 5, 6, 7), np.float32)
    TEST_OP(nonzero_astuple_true_forward_func, [[ms.Tensor(ms_data1)], [ms.Tensor(ms_data2)]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'])
