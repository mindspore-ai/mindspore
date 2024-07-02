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
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops


def softmax_(x, axis):
    op = ops.Softmax(axis)
    return op(x)


@test_utils.run_with_cell
def softmax_forward_func(x, axis=-1):
    return softmax_(x, axis)


@test_utils.run_with_cell
def softmax_backward_func(x, axis=-1):
    return ops.grad(softmax_forward_func, (0,))(x, axis)


@test_utils.run_with_cell
def softmax_backward_forward_func(dout, out, dim=-1):
    return ops.auto_generate.SoftmaxBackward()(dout, out, dim)


@test_utils.run_with_cell
def softmax_double_backward_func(dout, out, dim=-1):
    return ops.grad(softmax_backward_forward_func, (0, 1))(dout, out, dim)


@test_utils.run_with_cell
def tensor_softmax_forward_func(x, axis=-1):
    return x.softmax(axis)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_softmax_static_shape(mode):
    """
    Feature: Ops
    Description: test op softmax
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    logits = Tensor(np.array([1, 2, 3, 4, 5]).astype(np.float32))
    output = softmax_forward_func(logits, 0)
    expect_out = np.array(
        [0.01165623, 0.03168492, 0.08612854, 0.23412167, 0.6364086]
    ).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_out, 1e-04, 1e-04)

    x = Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    grads = softmax_backward_func(x, 0)
    expect_shape = (10, 36, 12, 12)
    assert grads.asnumpy().shape == expect_shape


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [context.GRAPH_MODE])
def test_softmax_double_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op SoftmaxBackward.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    context.set_context(jit_config={"jit_level": "O2"})
    dout = Tensor(np.random.rand(10, 10).astype(np.float32))
    out = Tensor(np.random.rand(10, 10).astype(np.float32))
    dim = -1
    grads = softmax_double_backward_func(dout, out, dim)
    print(grads)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_sofmax_vmap(mode):
    """
    Feature: test vmap function.
    Description: test softmax op vmap.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    in_axes = [-1, None]
    x = Tensor(np.random.randn(1, 6, 6, 3, 6).astype(np.float32))
    nest_vmap = ops.vmap(
        ops.vmap(softmax_forward_func, in_axes=in_axes, out_axes=0),
        in_axes=in_axes,
        out_axes=0,
    )
    out = nest_vmap(x, 0)
    expect_shape = (6, 3, 1, 6, 6)
    assert out.asnumpy().shape == expect_shape


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_tensor_softmax(mode):
    """
    Feature: Tensor
    Description: test Tensor.softmax
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    logits = Tensor(np.array([1, 2, 3, 4, 5]).astype(np.float32))
    output = tensor_softmax_forward_func(logits)
    expect_out = np.array(
        [0.01165623, 0.03168492, 0.08612854, 0.23412167, 0.6364086]
    ).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_softmax_dynamic():
    """
    Feature: Ops
    Description: test op softmax dynamic shape
    Expectation: expect correct result.
    """
    context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(8, 30, 3), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(9, 4), dtype=ms.float32)
    TEST_OP(
        softmax_forward_func,
        [
            [input_case1, (-1,)],
            [input_case2, (0,)],
        ],
        "softmax", disable_input_check=True
    )


def run_softmax_api(ms_type, nptype):
    """
    Feature: test softmax tensor api.
    Description: test inputs using given dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.array([1, 2, 3, 4, 5]), ms_type)
    output = softmax_forward_func(input_x, -1)
    excepted = np.array(
        [0.01165623, 0.03168492, 0.08612854, 0.23412165, 0.6364086]
    ).astype(nptype)
    if ms_type == ms.bfloat16:
        np.testing.assert_array_almost_equal(
            output.float().asnumpy(), excepted, decimal=3
        )
    else:
        np.testing.assert_array_almost_equal(output.asnumpy(), excepted, decimal=6)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_softmax_bfloat16_tensor_api(mode):
    """
    Feature: test softmax tensor api.
    Description: test bfloat16 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    run_softmax_api(ms.bfloat16, np.float32)
