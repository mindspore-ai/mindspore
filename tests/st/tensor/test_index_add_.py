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
from mindspore import nn
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def genetate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


class Net(nn.Cell):

    def construct(self, x, dim, index, source, alpha):
        output = x.index_add_(dim, index, source, alpha=alpha)
        return output


def index_add__forward_func(x, dim, index, source, alpha):
    return Net()(x, dim, index, source, alpha)


def index_add__forward_func_grad(x, dim, index, source, alpha):
    return Net()(x * 1, dim, index, source, alpha)


@test_utils.run_with_cell
def index_add__backward_func(x, dim, index, source, alpha):
    return ms.grad(index_add__forward_func_grad, (3))(x, dim, index, source,
                                                      alpha)


def set_mode(mode):
    if mode == "GRAPH_MODE":
        ms.context.set_context(mode=ms.GRAPH_MODE,
                               jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_tensor_index_add_(mode):
    """
    Feature: tensor.index_add_
    Description: Verify the result of tensor.index_add_
    Expectation: success
    """
    set_mode(mode)
    x = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                           dtype=np.float32))
    axis = 1
    index = ms.Tensor([0, 2], ms.int32)
    source = ms.Tensor(
        np.array([[1., 3.], [4., 6.], [7., 9.]], dtype=np.float32))
    expect_output = ms.Tensor(
        np.array([[2., 2., 6.], [8., 5, 12.], [14., 8., 18.]],
                 dtype=np.float32))
    expect_backward_output = ms.Tensor(
        np.array([[1., 1.], [1., 1.], [1., 1.]], dtype=np.float32))
    output = index_add__forward_func(x, axis, index, source, 1)
    backward_output = index_add__backward_func(x, axis, index, source, 1)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
    assert np.allclose(backward_output.asnumpy(),
                       expect_backward_output.asnumpy())
    assert np.allclose(x.asnumpy(), expect_output.asnumpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_index_add__dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: Verify the result of tensor.index_add_
    Expectation: success
    """
    tensor_x1 = ms.Tensor(genetate_random_input((3, 3), np.float32))
    tensor_x2 = ms.Tensor(genetate_random_input((3, 3, 3), np.float32))

    axis1 = 0
    axis2 = 1

    index1 = ms.Tensor([0, 2], ms.int32)
    index2 = ms.Tensor([0, 1, 2], ms.int32)

    tensor_source1 = ms.Tensor(genetate_random_input((2, 3), np.float32))
    tensor_source2 = ms.Tensor(genetate_random_input((3, 3, 3), np.float32))

    alpha1 = 1
    alpha2 = 2

    TEST_OP(index_add__forward_func_grad,
            [[tensor_x1, axis1, index1, tensor_source1, alpha1],
             [tensor_x2, axis2, index2, tensor_source2, alpha2]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True},
            inplace_update=True)
