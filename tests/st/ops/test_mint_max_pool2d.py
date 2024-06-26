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
from mindspore import ops
from mindspore.mint.nn.functional import max_pool2d
from mindspore import dtype as mstype
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def max_pool2d_forward_func(x, kernel_size, stride, padding, dilation, ceil_mode=False, return_indices=False):
    return max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode=ceil_mode, return_indices=return_indices)

@test_utils.run_with_cell
def max_pool2d_backward_func(x, kernel_size, stride, padding, dilation, ceil_mode, return_indices):
    return ops.grad(max_pool2d_forward_func, (0,))(x, kernel_size, stride, padding, dilation,
                                                   ceil_mode, return_indices)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_max_pool2d_forward_return_indices(mode):
    """
    Feature: Pyboost function.
    Description: Test function max_pool2d forward with return indices.
    Expectation: Correct result.
    """
    ms.context.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x = np.array([[[[1, 2, 3], [1, 2, 3]]]]).astype(np.float32)
    kernel_size = 2
    stride = None
    padding = 0
    dilation = (1, 1)
    return_indices = True
    ceil_mode = False
    output, indices = max_pool2d_forward_func(ms.Tensor(x), kernel_size, stride, padding,
                                              dilation, ceil_mode, return_indices)
    expect_out1 = np.array([[[[2.]]]])
    expect_out2 = np.array([[[[1]]]])
    np.testing.assert_allclose(output.asnumpy(), expect_out1, rtol=1e-6)
    np.testing.assert_allclose(indices.asnumpy(), expect_out2, rtol=1e-6)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_max_pool2d_forward_without_return_indices(mode):
    """
    Feature: Pyboost function.
    Description: Test function max_pool2d forward without return indices.
    Expectation: Correct result.
    """
    ms.context.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x = np.array([[[[1, 2, 3], [1, 2, 3]]]]).astype(np.float32)
    kernel_size = 2
    stride = None
    padding = 0
    dilation = (1, 1)
    return_indices = False
    ceil_mode = False
    output = max_pool2d_forward_func(ms.Tensor(x), kernel_size, stride, padding,
                                     dilation, ceil_mode, return_indices)
    expect_out = np.array([[[[2.]]]])
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=1e-6)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_max_pool2d_backward_return_indices(mode):
    """
    Feature: Pyboost function.
    Description: Test function max_pool2d backward with return indices.
    Expectation: Correct result.
    """
    ms.context.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x = np.array([[[[1, 2, 3], [1, 2, 3]]]]).astype(np.float32)
    kernel_size = 2
    stride = None
    padding = 0
    dilation = (1, 1)
    return_indices = True
    ceil_mode = False
    output = max_pool2d_backward_func(ms.Tensor(x), kernel_size, stride, padding, dilation,
                                      ceil_mode, return_indices)
    expect = np.array([[[[0., 1., 0.], [0., 0., 0.]]]])
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-6)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_max_pool2d_backward_without_return_indices(mode):
    """
    Feature: Pyboost function.
    Description: Test function max_pool2d backward without return indices.
    Expectation: Correct result.
    """
    ms.context.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x = np.array([[[[1, 2, 3], [1, 2, 3]]]]).astype(np.float32)
    kernel_size = 2
    stride = None
    padding = 0
    dilation = (1, 1)
    return_indices = False
    ceil_mode = False
    output = max_pool2d_backward_func(ms.Tensor(x), kernel_size, stride, padding, dilation,
                                      ceil_mode, return_indices)
    expect = np.array([[[[0., 1., 0.], [0., 0., 0.]]]])
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-6)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_ops_max_pool2d_dynamic():
    """
    Feature: Pyboost function.
    Description: Test function max_pool2d forward and backward with dynamic shape and rank.
    Expectation: Correct result.
    """
    x1 = ms.Tensor(np.arange(2 * 3 * 10 * 20).reshape((2, 3, 10, 20)), mstype.float32)
    kernel_size1 = 2
    stride1 = 2
    padding1 = 0
    dilation1 = 1
    ceil_mode1 = True

    x2 = ms.Tensor(np.arange(10 * 1 * 20 * 10).reshape((10, 1, 20, 10)), mstype.float32)
    kernel_size2 = 4
    stride2 = 2
    padding2 = 2
    dilation2 = 1
    ceil_mode2 = True

    TEST_OP(max_pool2d_forward_func,
            [[x1, kernel_size1, stride1, padding1, dilation1, ceil_mode1],
             [x2, kernel_size2, stride2, padding2, dilation2, ceil_mode2]], '', disable_input_check=True,
            disable_yaml_check=True, disable_mode=['GRAPH_MODE'])
