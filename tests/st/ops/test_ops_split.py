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

import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops
from mindspore.ops.function.array_func import split_ext

from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

@test_utils.run_with_cell
def split_forward_func(x, axis, output_num):
    return split_ext(x, axis, output_num)

@test_utils.run_with_cell
def split_backward_func(x, axis, output_num):
    return ops.grad(split_forward_func, (0,))(x, axis, output_num)

def split_dyn_shape_func(x, axis=0, output_num=2):
    return ops.Split(axis, output_num)(x)

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_split_ext_int_forward(mode):
    """
    Feature: Split
    Description: test op Split
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_x = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(np_x, dtype=ms.float32)
    out = split_forward_func(x, 5, 0)
    expect = [np.array(np.arange(10).reshape((5, 2)), dtype=np.float32),
              np.array(np.arange(10, 20).reshape((5, 2)), dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_split_ext_int_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op Split.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.arange(20).reshape(10, 2), (ms.float32))
    split_size_or_sections = 5
    grads = split_backward_func(x, split_size_or_sections, 0)
    expect_shape = x.shape
    assert grads.asnumpy().shape == expect_shape

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_split_ext_list_forward(mode):
    """
    Feature: Split.
    Description: test auto grad of op Split.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    logits = np.array(np.arange(20).reshape((2, 10)), dtype=np.float32)
    x = ms.Tensor(logits, dtype=ms.float32)
    split_size_or_sections = [1, 1]
    out = split_forward_func(x, split_size_or_sections, 0)
    expect = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32),
              np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_split_ext_list_backward(mode):
    """
    Feature: Split.
    Description: test auto grad of op Split.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)

    logits = np.array(np.arange(20).reshape((2, 10)), dtype=np.float32)
    x = ms.Tensor(logits, dtype=ms.float32)
    split_size_or_sections = [1, 1]
    grads = split_backward_func(x, split_size_or_sections, 0)
    expect_shape = logits.shape
    assert grads.asnumpy().shape == expect_shape

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_f_split_ext_dynamic(mode):
    """
    Feature: test dynamic split.
    Description: test auto grad of op Split.
    Expectation: expect correct result.
    """
    np_x1 = np.arange(4 * 4).reshape(4, 4)
    x1 = ms.Tensor(np_x1, ms.float32)
    np_x2 = np.arange(4 * 6).reshape(4, 6)
    x2 = ms.Tensor(np_x2, ms.float32)
    TEST_OP(split_forward_func, [[x1, 2, 1], [x2, 2, 1]], mode=mode, grad=True)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_split_dynamic(mode):
    """
    Feature: test dynamic tensor of split.
    Description: test dynamic tensor of split.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(split_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    np_x1 = np.arange(2 * 2).reshape(2, 2)
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect1 = (np.array([[0, 1]]), np.array([[2, 3]]))
    for res, exp in zip(output1, expect1):
        assert np.allclose(res.asnumpy(), exp)
    np_x2 = np.arange(2 * 3).reshape(2, 3)
    x2 = ms.Tensor(np_x2, ms.float32)
    output2 = test_cell(x2)
    expect2 = (np.array([[0, 1, 2]]), np.array([[3, 4, 5]]))
    for res, exp in zip(output2, expect2):
        assert np.allclose(res.asnumpy(), exp)
