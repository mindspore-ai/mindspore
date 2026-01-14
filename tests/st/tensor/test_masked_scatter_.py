# Copyright 2025 Huawei Technologies Co., Ltd
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
from mindspore import nn, Tensor
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def genetate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


class Net(nn.Cell):

    def construct(self, x, y, z):
        x.masked_scatter_(y, z)
        return x


def masked_scatter__forward_func(x, y, z):
    return Net()(x.clone(), y, z)


@test_utils.run_with_cell
def masked_scatter__backward_func(x, y, z):
    return ms.grad(masked_scatter__forward_func, (0, 2))(x, y, z)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_tensor_masked_scatter_(mode):
    """
    Feature: tensor.masked_scatter_
    Description: Verify the result of tensor.masked_scatter_
    Expectation: success
    """
    if mode == ms.GRAPH_MODE:
        ms.context.set_context(mode=mode, jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=mode)
    net = Net()
    x = Tensor(np.array([1., 2., 3., 4.]), ms.float32)
    mask = Tensor(np.array([True, True, False, True]), ms.bool_)
    source = Tensor(np.array([5., 6., 7.]), ms.float32)
    output = net(x, mask, source)
    expect_output = Tensor(np.asarray([5., 6., 3., 7.]), ms.float32)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
    assert np.allclose(x.asnumpy(), expect_output.asnumpy())
    expect_x_grad = Tensor(np.array([0., 0., 1., 0.], dtype=np.float32))
    expect_source_grad = Tensor(np.array([1., 1., 1.], dtype=np.float32))
    grad_output = masked_scatter__backward_func(x, mask, source)
    assert np.allclose(grad_output[0].asnumpy(), expect_x_grad.asnumpy())
    assert np.allclose(grad_output[1].asnumpy(), expect_source_grad.asnumpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_tensor_masked_scatter__dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: Verify the result of tensor.masked_scatter_
    Expectation: success
    """
    x1 = Tensor(genetate_random_input((3, 3), np.float32))
    mask1 = Tensor(genetate_random_input((3, 3), np.bool_))
    source1 = Tensor(genetate_random_input((3, 3), np.float32))

    x2 = Tensor(genetate_random_input((3, 3, 3), np.float32))
    mask2 = Tensor(genetate_random_input((3, 3, 3), np.bool_))
    source2 = Tensor(genetate_random_input((3, 3, 3), np.float32))

    TEST_OP(masked_scatter__forward_func,
            [[x1, mask1, source1], [x2, mask2, source2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor'],
            inplace_update=True)
