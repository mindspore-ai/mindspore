"""ST tests for searchsorted operators."""

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

import numpy as np
import pytest
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
from mindspore import ops, nn
import mindspore as ms


class SearchSortedNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.searchsorted = ops.searchsorted

    def construct(self, x1, x2, out_int32=False, right=False, side="left", sorter=None):
        return self.searchsorted(x1, x2, out_int32=out_int32, right=right, side=side, sorter=sorter)


class TensorSearchSortedNet(nn.Cell):
    def __init__(self, side="left"):
        super().__init__()
        self.side = side

    def construct(self, x, y, sorter):
        return x.searchsorted(y, side=self.side, sorter=sorter)


def generate_random_input(shape, dtype):
    return np.sort(np.random.uniform(0.9, 1.0, size=shape).astype(dtype)), \
        np.random.uniform(0.9, 1.0, size=shape).astype(dtype)


def searchsorted_forward_func(x, y, out_int32=False, right=False, sorter=None):
    return SearchSortedNet()(x, y, out_int32=out_int32, right=right, side="right", sorter=sorter)


def searchsorted_forward_func_static_dtype(x, y, right=False, sorter=None):
    return SearchSortedNet()(x, y, right=right, side="right", sorter=sorter)


def generate_expect_forward_output(x, y, side="right", sorter=None):
    return np.searchsorted(x, y, side=side, sorter=sorter)


@test_utils.run_with_cell
def searchsorted_vmap_func(x, y, out_int32=False, right=False, sorter=None):
    return ops.vmap(searchsorted_forward_func, in_axes=(0, 0, None, None, None),
                    out_axes=0)(x, y, out_int32, right, sorter)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_searchsorted_forward(context_mode):
    """
    Feature: Ops.
    Description: test op searchsorted forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_np = np.array([-1, 0.0, 1.0, 2.0], dtype=np.float32)
    y_np = np.array([0.0, 1.0, 2.0, -2], dtype=np.float32)
    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    out = searchsorted_forward_func(x, y)
    expect_out = generate_expect_forward_output(x_np, y_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', mem_peak=1024.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_searchsorted_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function searchsorted vmap.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x_np = np.array([[-1, 0.0, 1.0, 2.0], [-1, 0.0, 1.0, 2.0]], dtype=np.float32)
    y_np = np.array([[0.0, 1.0, 2.0, -2], [0.0, 1.0, 2.0, -2]], dtype=np.float32)
    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    out = searchsorted_vmap_func(x, y)
    expect_out = []
    for s_x_np, s_y_np in zip(x_np, y_np):
        np_out = generate_expect_forward_output(s_x_np, s_y_np)
        expect_out.append(np_out)

    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'],
        level_mark='level1', card_mark='onecard', mem_peak=2048.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_searchsorted_dynamic(context_mode):
    """
    Feature: pyboost function.
    Description: test function searchsorted dynamic.
    Expectation: expect correct result.
    """

    x1, _ = generate_random_input((20), np.float32)
    _, other1 = generate_random_input((2, 3, 4, 5), np.float32)
    right_1 = False
    sorter_1 = ms.Tensor(np.argsort(x1, axis=-1))

    x2, _ = generate_random_input((2, 3, 4), np.float32)
    _, other2 = generate_random_input((2, 3, 4), np.float32)
    right_2 = True
    sorter_2 = ms.Tensor(np.argsort(x2, axis=-1))

    TEST_OP(searchsorted_forward_func_static_dtype,
            [[ms.Tensor(x1), ms.Tensor(other1), right_1, sorter_1],
             [ms.Tensor(x2), ms.Tensor(other2), right_2, sorter_2]],
            disable_generalize_test=True,
            disable_case=['ScalarTensor'])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', mem_peak=2048.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_searchsorted_invalid_side_raises_before_device_launch(context_mode):
    """
    Feature: Ops.
    Description: Invalid side for searchsorted should be rejected before device execution.
    Expectation: Raise ValueError and leave follow-up execution healthy.
    """
    ms.context.set_context(mode=context_mode)
    sorted_sequence_np = np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float32)
    values_np = np.array([2.0], dtype=np.float32)
    sorted_sequence = ms.Tensor(sorted_sequence_np)
    values = ms.Tensor(values_np)
    sorter = ms.Tensor(np.argsort(sorted_sequence_np), ms.int64)

    with pytest.raises(ValueError, match="argument 'side'"):
        ops.searchsorted(sorted_sequence, values, side="11313", sorter=sorter).asnumpy()

    output = ops.searchsorted(sorted_sequence, values, side="right")
    expect = np.searchsorted(sorted_sequence_np, values_np, side="right")
    np.testing.assert_array_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', mem_peak=2048.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_searchsorted_invalid_sorter_shape_raises_before_device_launch(context_mode):
    """
    Feature: Ops.
    Description: Invalid sorter for searchsorted should be rejected before ACLNN execution.
    Expectation: Raise user-facing ValueError and keep follow-up execution healthy.
    """
    ms.context.set_context(mode=context_mode)
    sorted_sequence_np = np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float32)
    values_np = np.array([2.0], dtype=np.float32)
    sorted_sequence = ms.Tensor(sorted_sequence_np)
    values = ms.Tensor(values_np)
    sorter = ms.Tensor(np.array([0, 1, 2], dtype=np.int64))

    with pytest.raises(ValueError, match="shape of 'sorter' must be the same as 'sorted_sequence'"):
        ops.searchsorted(sorted_sequence, values, sorter=sorter).asnumpy()

    output = ops.searchsorted(sorted_sequence, values, sorter=ms.Tensor(np.argsort(sorted_sequence_np), ms.int64))
    expect = np.searchsorted(sorted_sequence_np, values_np, sorter=np.argsort(sorted_sequence_np))
    np.testing.assert_array_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', mem_peak=2048.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_searchsorted_invalid_side_raises_before_device_launch(context_mode):
    """
    Feature: Tensor.searchsorted.
    Description: Invalid side for Tensor.searchsorted should be rejected before device execution.
    Expectation: Raise ValueError and keep follow-up execution healthy.
    """
    ms.context.set_context(mode=context_mode)
    sorted_sequence_np = np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float32)
    values_np = np.array([2.0], dtype=np.float32)
    sorted_sequence = ms.Tensor(sorted_sequence_np)
    values = ms.Tensor(values_np)
    sorter = ms.Tensor(np.argsort(sorted_sequence_np), ms.int64)

    with pytest.raises(ValueError, match="argument 'side'"):
        sorted_sequence.searchsorted(values, side="11313", sorter=sorter).asnumpy()

    output = sorted_sequence.searchsorted(values, side="right")
    expect = np.searchsorted(sorted_sequence_np, values_np, side="right")
    np.testing.assert_array_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', mem_peak=2048.0)
def test_tensor_searchsorted_invalid_side_in_graph_construct_raises_before_device_launch():
    """
    Feature: Tensor.searchsorted graph syntax.
    Description: Invalid side in Tensor.searchsorted inside Cell.construct should be rejected before device execution.
    Expectation: Raise ValueError and keep follow-up graph execution healthy.
    """
    ms.context.set_context(mode=ms.GRAPH_MODE)
    sorted_sequence_np = np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float32)
    values_np = np.array([2.0], dtype=np.float32)
    sorted_sequence = ms.Tensor(sorted_sequence_np)
    values = ms.Tensor(values_np)
    sorter = ms.Tensor(np.argsort(sorted_sequence_np), ms.int64)

    with pytest.raises(ValueError, match="argument 'side'"):
        TensorSearchSortedNet(side="11313")(sorted_sequence, values, sorter)

    output = TensorSearchSortedNet(side="right")(sorted_sequence, values, sorter)
    expect = np.searchsorted(sorted_sequence_np, values_np, side="right", sorter=np.argsort(sorted_sequence_np))
    np.testing.assert_array_equal(output.asnumpy(), expect)
