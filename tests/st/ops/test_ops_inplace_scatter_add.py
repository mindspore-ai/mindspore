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
from mindspore import Tensor, ops
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


class Net(ms.nn.Cell):
    def construct(self, input_x, dim, index, src):
        input_x.scatter_add_(dim, index, src)
        return input_x


def inplace_scatter_add_forward_func(input_x, dim, index, src):
    return Net()(input_x, dim, index, src)


def inplace_scatter_add_backward_func(input_x, dim, index, src):
    grad = ops.GradOperation(get_all=True)
    return grad(Net(), (0, 3))(input_x, dim, index, src)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_inplace_scatter_add_normal(mode):
    """
    Feature: Ops.
    Description: test op inplace_scatter_add.
    Expectation: expect correct result.
    """
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode, device_target="Ascend")
    ## forward
    x = Tensor(np.array([[1.341, 2.435, 3.21, -4.144, 5.098]]), dtype=ms.float32)
    src = Tensor(np.array([[8.131, -2.348]]), dtype=ms.float32)
    dim = 1
    index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
    out = inplace_scatter_add_forward_func(x, dim, index, src)
    expect = np.array([[1.341, 2.435, 11.341, -4.144, 2.75]])
    assert np.allclose(out.asnumpy(), expect)

    x = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    src = Tensor(np.array([[1.23, -2.34, 3.45], [-4.56, 5.67, 6.78], [-7.89, -8.91, 9.123]]), dtype=ms.float32)
    index = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), dtype=ms.int64)
    dim = 0
    out1 = inplace_scatter_add_forward_func(x, dim, index, src)
    expect1 = np.array([[1.23, -2.34, 3.45, 0., 0.,],
                        [0., 0., 0., 0., 0.],
                        [-4.56, 5.67, 6.78, 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [-7.89, -8.91, 9.123, 0., 0.]])
    assert np.allclose(out1.asnumpy(), expect1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=34.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_inplace_scatter_add_bfloat16(mode):
    """
    Feature: Ops.
    Description: test op inplace_scatter_add.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode, device_target="Ascend")
    ## forward
    x = Tensor(np.array([[1.412, 2.124, 3.1412, 4.68752, 5.14135]]), dtype=ms.bfloat16)
    src = Tensor(np.array([[8.134, 8.351897]]), dtype=ms.bfloat16)
    dim = 1
    index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
    out = inplace_scatter_add_forward_func(x, dim, index, src)
    expect = np.array([[1.412, 2.124, 11.2752, 4.68752, 13.493247]])
    assert np.allclose(out.float().asnumpy(), expect, rtol=4e-3, atol=4e-3)

    x = Tensor(np.zeros((5, 5)), dtype=ms.bfloat16)
    src = Tensor(np.array([[1.123, 2, 3], [4, -4.55415, 6], [7, 8, -9131.1349]]), dtype=ms.bfloat16)
    index = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), dtype=ms.int64)
    dim = 0
    out1 = inplace_scatter_add_forward_func(x, dim, index, src)
    expect1 = np.array([[1.123, 2., 3., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [4, -4.55415, 6., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [7., 8., -9131.1349, 0., 0.]])
    assert np.allclose(out1.float().asnumpy(), expect1, rtol=4e-3, atol=4e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_inplace_scatter_add_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test ops.inplace_scatter_add dynamic shape feature.
    Expectation: expect correct result.
    """
    x1 = Tensor(np.array([[1.123, -2.311, 3.1238, 4.8614, -5.9714]]), dtype=ms.float32)
    dim1 = 1
    index1 = Tensor(np.array([[2, 4]]), dtype=ms.int64)
    src1 = Tensor(np.array([[8.341, 9.38]]), dtype=ms.float32)

    x2 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    dim2 = 0
    index2 = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), dtype=ms.int64)
    src2 = Tensor(np.array([[1.123, 2, 3], [4.35, 5.131, -6.513], [7.24, -1.348, 9.314]]), dtype=ms.float32)
    TEST_OP(inplace_scatter_add_forward_func, [[x1, dim1, index1, src1], [x2, dim2, index2, src2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True, 'disable_grad': True})
