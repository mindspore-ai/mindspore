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
from mindspore.common import mutable
from mindspore.ops.auto_generate.gen_ops_def import index
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def index_forward_func(x, indices):
    return index(x, indices)


@test_utils.run_with_cell
def index_backward_func(x, indices):
    return ms.grad(index_forward_func, (0,))(x, indices)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_index_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function index forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    x = generate_random_input((3, 4, 5, 6, 7), np.float64)

    # shape(0,) and shape(0,0,0,0,0,0,0,0,0)
    indices1 = ms.Tensor(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32))
    indices2 = ms.Tensor(np.array([0, 1, 2], dtype=np.int32))
    indices3 = ms.Tensor(np.array(([1], [1]), dtype=np.int32))
    indices4 = ms.Tensor(ms.numpy.empty([0]*9, dtype=ms.numpy.int64))
    indices5 = ms.Tensor(ms.numpy.empty((0,), dtype=ms.numpy.int64))

    output_1 = index_forward_func(ms.Tensor(x), [indices1, indices2])
    output_2 = index_forward_func(ms.Tensor(x), [indices4, indices3, indices5, indices4, indices4])
    output_3 = index_forward_func(ms.Tensor(x), [indices5, indices4, indices3, indices4])
    output_4 = index_forward_func(ms.Tensor(x), [indices4, indices4, indices5, indices4, indices3])
    output_5 = index_forward_func(ms.Tensor(x), [indices1, indices4, indices2, indices4, indices2])

    indices1_2 = ms.Tensor(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32))
    indices2_2 = ms.Tensor(np.array([0, 1, 2], dtype=np.int32))
    expect_1 = ms.Tensor(x)[indices1_2, indices2_2]
    expect_2 = ms.Tensor(x)[:, indices3, indices5, :, :]
    expect_3 = ms.Tensor(x)[indices5, :, indices3, :]
    expect_4 = ms.Tensor(x)[:, :, indices5, :, indices3]
    expect_5 = ms.Tensor(x)[indices1_2, :, indices2_2, :, indices2_2]

    np.testing.assert_allclose(output_1.asnumpy(), expect_1.asnumpy(), rtol=1e-3)
    np.testing.assert_allclose(output_2.asnumpy(), expect_2.asnumpy(), rtol=1e-3)
    np.testing.assert_allclose(output_3.asnumpy(), expect_3.asnumpy(), rtol=1e-3)
    np.testing.assert_allclose(output_4.asnumpy(), expect_4.asnumpy(), rtol=1e-3)
    np.testing.assert_allclose(output_5.asnumpy(), expect_5.asnumpy(), rtol=1e-3)

    # int and (bool or uint8)
    indices6 = np.array([1, 0], dtype=np.int32)
    indices7 = np.array([1, 0, 0, 1], dtype=np.bool_)
    indices8 = np.array([1, 0, 0, 1], dtype=np.uint8)

    output_6 = index_forward_func(ms.Tensor(x), [ms.Tensor(indices6), ms.Tensor(indices7)])
    output_7 = index_forward_func(ms.Tensor(x), [ms.Tensor(indices6), ms.Tensor(indices8)])
    ## the bool and uint8 is same for pta
    expect_6_7 = x[indices6, indices7]
    np.testing.assert_allclose(output_6.asnumpy(), expect_6_7, rtol=1e-3)
    np.testing.assert_allclose(output_7.asnumpy(), expect_6_7, rtol=1e-3)

    ## ValueError:size (indices_list) > rank input
    with pytest.raises(ValueError):
        output_index_error = index_forward_func(ms.Tensor(x), [ms.Tensor(indices6)] * (len(x.shape) + 1))
        print(output_index_error)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential',
        mem_peak=2048.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_index_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function index backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    x = generate_random_input((4, 3, 2, 4), np.float64)
    indices1 = np.array([[0, 1], [1, 2], [2, 1]], dtype=np.int32)
    indices2 = np.array([[1, 2]], dtype=np.int32)
    ## ms
    indices = [ms.Tensor(indices1), ms.Tensor(indices2)]
    test_cell = test_utils.to_cell_obj(index_backward_func)
    test_cell.set_inputs(ms.Tensor(x), indices)
    output_b = test_cell(ms.Tensor(x), indices)
    ## numpy
    output_f_int_1 = index_forward_func(ms.Tensor(x), [ms.Tensor(indices1), ms.Tensor(indices2)])
    grads = np.ones(output_f_int_1.shape, np.float64)
    tmp = np.zeros(x.shape)
    expect_b = np.zeros(x.shape)
    ### The original value does not exist.When the value of grads is 1,It can be avoided.
    indices2 = np.broadcast_to(indices2, indices1.shape)
    tmp[indices1, indices2] = grads
    for i in np.nditer([indices1, indices2]):
        expect_b[i] += tmp[i]
    np.testing.assert_allclose(output_b.asnumpy(), expect_b, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_index_bf16(context_mode):
    """
    Feature: pyboost function.
    Description: test function index forward(bf16).
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    x1 = x_np = generate_random_input((5, 6, 4, 3, 2, 4), np.float64)
    indices1 = indices1_np = np.array([[0, 1], [1, 3], [2, 1]], dtype=np.int32)
    indices2 = indices2_np = np.array([[0, 4]], dtype=np.int32)
    output = index_forward_func(ms.Tensor(x1, dtype=ms.bfloat16), [ms.Tensor(indices1), ms.Tensor(indices2)])
    expect = np.squeeze(x_np[indices1_np, indices2_np[:, np.newaxis]], axis=0)
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=4e-3, atol=4e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b']
          , level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_index_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function index  dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 5, 2, 2), np.float64)
    ms_data2 = generate_random_input((3, 4, 5, 6), np.float64)
    indices1 = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int32)
    indices2 = np.array([[0, 1]], dtype=np.int32)
    indices3 = np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int32)
    indices4 = np.array([[1, 1, 1], [1, 0, 1]], dtype=np.bool_)
    indices5 = np.array([1], dtype=np.int32)

    TEST_OP(index_forward_func,
            [[ms.Tensor(ms_data1), mutable([ms.Tensor(indices1), ms.Tensor(indices2)])],
             [ms.Tensor(ms_data2), mutable([ms.Tensor(indices3)])]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor', 'GradByRequirement']
            )

    TEST_OP(index_forward_func,
            [[ms.Tensor(ms_data1), mutable([ms.Tensor(indices4)])],
             [ms.Tensor(ms_data2), mutable([ms.Tensor(indices3), ms.Tensor(indices5)])]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor', 'GradByRequirement'],
            case_config={'disable_tensor_dynamic_type': "DYNAMIC_RANK"}
            )
