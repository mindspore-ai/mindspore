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
from mindspore.mint import scatter_add
from mindspore import Tensor, ops
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def scatter_add_ext_forward_func(x, dim, index, src):
    return scatter_add(input=x, dim=dim, index=index, src=src)


@test_utils.run_with_cell
def scatter_add_ext_backward_func(x, dim, index, src):
    return ops.grad(scatter_add_ext_forward_func, (0, 3))(x, dim, index, src)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scatter_add_ext_normal(mode):
    """
    Feature: Ops.
    Description: test op scatter_add_ext.
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
    out = scatter_add_ext_forward_func(x, dim, index, src)
    expect = np.array([[1.341, 2.435, 11.341, -4.144, 2.75]])
    assert np.allclose(out.asnumpy(), expect)

    x = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    src = Tensor(np.array([[1.23, -2.34, 3.45], [-4.56, 5.67, 6.78], [-7.89, -8.91, 9.123]]), dtype=ms.float32)
    index = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), dtype=ms.int64)
    dim = 0
    out1 = scatter_add_ext_forward_func(x, dim, index, src)
    expect1 = np.array([[1.23, -2.34, 3.45, 0., 0.,],
                        [0., 0., 0., 0., 0.,],
                        [-4.56, 5.67, 6.78, 0., 0.,],
                        [0., 0., 0., 0., 0.,],
                        [-7.89, -8.91, 9.123, 0., 0.,]])
    assert np.allclose(out1.asnumpy(), expect1)

    ## backward
    x = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    src = Tensor(np.array([[1.23, 2.232, 3.4343], [4.122, -5.1212, 6.131], [-7.1313, 0.8, -0.089]]), dtype=ms.float32)
    index = Tensor(np.array([[0, 2, 4], [0, 2, 4], [0, 2, 4]]), dtype=ms.int64)
    dim = 1
    out = scatter_add_ext_backward_func(x, dim, index, src)
    expect_dx = np.ones((5, 5))
    expect_dsrc = np.ones((3, 3))
    assert np.allclose(out[0].asnumpy(), expect_dx)
    assert np.allclose(out[1].asnumpy(), expect_dsrc)

    x = Tensor(np.array([[-0.11, 2.141, 3.1314, 4.645, -5.14]]), dtype=ms.float32)
    src = Tensor(np.array([[8.131, -131.8]]), dtype=ms.float32)
    dim = 0
    index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
    out1 = scatter_add_ext_backward_func(x, dim, index, src)
    expect_dx = np.ones((5, 5))
    expect_dsrc = np.ones((1, 2))
    assert np.allclose(out1[0].asnumpy(), expect_dx)
    assert np.allclose(out1[1].asnumpy(), expect_dsrc)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scatter_add_ext_vmap(mode):
    """
    Feature: test vmap function.
    Description: test scatter_add_ext op vmap.
    Expectation: expect correct result.
    """
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[[4.123, 5.1232, -12321.233, -4.33367, 5.1237]]]), dtype=ms.float32)
    src = Tensor(np.array([[8.1313, -1.2228]]), dtype=ms.float32)
    dim = 1
    index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
    expect = np.array([[4.123, 5.1232, -12313.1017, -4.33367, 3.9009]])
    nest_vmap = ops.vmap(scatter_add_ext_forward_func, in_axes=(0, None, None, None), out_axes=(0,))
    out = nest_vmap(x, dim, index, src)
    expect = scatter_add_ext_forward_func(x[0], dim, index, src)
    assert np.allclose(out.asnumpy(), expect.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scatter_add_ext_bfloat16(mode):
    """
    Feature: Ops.
    Description: test op scatter_add_ext.
    Expectation: expect correct result.
    """
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode, device_target="Ascend")
    ## forward
    x = Tensor(np.array([[1.412, 2.124, 3.1412, 4.68752, 5.14135]]), dtype=ms.bfloat16)
    src = Tensor(np.array([[8.134, 8.351897]]), dtype=ms.bfloat16)
    dim = 1
    index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
    out = scatter_add_ext_forward_func(x, dim, index, src)
    expect = np.array([[1.412, 2.124, 11.2752, 4.68752, 13.493247]])
    assert np.allclose(out.float().asnumpy(), expect, rtol=4e-3, atol=4e-3)

    x = Tensor(np.zeros((5, 5)), dtype=ms.bfloat16)
    src = Tensor(np.array([[1.123, 2, 3], [4, -4.55415, 6], [7, 8, -9131.1349]]), dtype=ms.bfloat16)
    index = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), dtype=ms.int64)
    dim = 0
    out1 = scatter_add_ext_forward_func(x, dim, index, src)
    expect1 = np.array([[1.123, 2., 3., 0., 0.,],
                        [0., 0., 0., 0., 0.,],
                        [4, -4.55415, 6., 0., 0.,],
                        [0., 0., 0., 0., 0.,],
                        [7., 8., -9131.1349, 0., 0.,]])
    assert np.allclose(out1.float().asnumpy(), expect1, rtol=4e-3, atol=4e-3)

    ## backward
    x = Tensor(np.zeros((5, 5)), dtype=ms.bfloat16)
    src = Tensor(np.array([[1.123, 2, -131.3], [4, 314.5, 6.1421454], [7.356478, 8, -341.45559]]), dtype=ms.bfloat16)
    index = Tensor(np.array([[0, 2, 4], [0, 2, 4], [0, 2, 4]]), dtype=ms.int64)
    dim = 1
    out = scatter_add_ext_backward_func(x, dim, index, src)
    expect_dx = np.ones((5, 5))
    expect_dsrc = np.ones((3, 3))
    assert np.allclose(out[0].float().asnumpy(), expect_dx, rtol=4e-3, atol=4e-3)
    assert np.allclose(out[1].float().asnumpy(), expect_dsrc, rtol=4e-3, atol=4e-3)

    x = Tensor(np.array([[1.3135, 2.314135, 3.515, -15135.4, -1.222335]]), dtype=ms.bfloat16)
    src = Tensor(np.array([[8, 8]]), dtype=ms.bfloat16)
    dim = 0
    index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
    out1 = scatter_add_ext_backward_func(x, dim, index, src)
    expect_dx = np.ones((5, 5))
    expect_dsrc = np.ones((1, 2))
    assert np.allclose(out1[0].float().asnumpy(), expect_dx, rtol=4e-3, atol=4e-3)
    assert np.allclose(out1[1].float().asnumpy(), expect_dsrc, rtol=4e-3, atol=4e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scatter_add_ext_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test ops.scatter_add_ext dynamic shape feature.
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
    TEST_OP(scatter_add_ext_forward_func, [[x1, dim1, index1, src1], [x2, dim2, index2, src2]], 'scatter_add_ext',
            disable_input_check=True, disable_mode=['GRAPH_MODE'])
