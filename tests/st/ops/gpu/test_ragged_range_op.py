# Copyright 2022 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore.ops.operations.math_ops import RaggedRange
from mindspore.nn import Cell
from mindspore import Tensor

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class RaggedRangeNet(Cell):
    def __init__(self, tsplits=mstype.int32):
        super(RaggedRangeNet, self).__init__()
        self.ragged_range = RaggedRange(Tsplits=tsplits)

    def construct(self, starts, limits, deltas):
        return self.ragged_range(starts, limits, deltas)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_int():
    """
    Feature: test int32.
    Description: test RaggedRange when input is int32.
    Expectation: result is right.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    raggedrange = RaggedRange(Tsplits=mstype.int32)
    starts = Tensor(np.array([9, 10, 11, 12, 13]).astype(np.int32))
    limits = Tensor(np.array([20, 21, 22, 23, 24]).astype(np.int32))
    deltas = Tensor(np.array([6, 90, 8, 9, 10]).astype(np.int32))
    (rt_nested_splits, rt_dense_values) = raggedrange(starts, limits, deltas)
    assert rt_nested_splits.asnumpy().dtype == 'int32'
    assert rt_dense_values.asnumpy().dtype == 'int32'
    rt_nested_splits_expected = np.array([0, 2, 3, 5, 7, 9], np.int32)
    np.testing.assert_array_equal(rt_nested_splits.asnumpy(), rt_nested_splits_expected)
    rt_dense_values_expected = np.array([9, 15, 10, 11, 19, 12, 21, 13, 23], np.int32)
    np.testing.assert_array_equal(rt_dense_values.asnumpy(), rt_dense_values_expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_float():
    """
    Feature: test float32.
    Description: test RaggedRange when input is float32.
    Expectation: result is right.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    raggedrange = RaggedRangeNet(mstype.int32)
    starts = Tensor([2.3, 1.22], mstype.float32)
    limits = Tensor([5.5, 1.30], mstype.float32)
    deltas = Tensor([1.2, 0.02], mstype.float32)
    (rt_nested_splits, rt_dense_values) = raggedrange(starts, limits, deltas)
    assert rt_nested_splits.asnumpy().dtype == 'int32'
    assert rt_dense_values.asnumpy().dtype == 'float32'
    rt_nested_splits_expected = np.array([0, 3, 7], np.int32)
    np.testing.assert_array_equal(rt_nested_splits.asnumpy(), rt_nested_splits_expected)
    rt_dense_values_expected = np.array([2.3, 3.5, 4.7, 1.22, 1.24, 1.26, 1.28], np.float32)
    np.testing.assert_array_equal(rt_dense_values.asnumpy(), rt_dense_values_expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_int64():
    """
    Feature: test int64.
    Description: test RaggedRange when input is int64.
    Expectation: result is right.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    raggedrange = RaggedRange(Tsplits=mstype.int64)
    starts = Tensor(np.array([4, 2, 1, 6, 2147483648]).astype(np.int64))
    limits = Tensor(np.array([8, 9, 10, 11, 2147483650]).astype(np.int64))
    deltas = Tensor(np.array([2, 4, 6, 8, 1]).astype(np.int64))
    (rt_nested_splits, rt_dense_values) = raggedrange(starts, limits, deltas)
    assert rt_nested_splits.asnumpy().dtype == 'int64'
    assert rt_dense_values.asnumpy().dtype == 'int64'
    rt_nested_splits_expected = np.array([0, 2, 4, 6, 7, 9], np.int64)
    np.testing.assert_array_equal(rt_nested_splits.asnumpy(), rt_nested_splits_expected)
    rt_dense_values_expected = np.array([4, 6, 2, 6, 1, 7, 6, 2147483648, 2147483649], np.int64)
    np.testing.assert_array_equal(rt_dense_values.asnumpy(), rt_dense_values_expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_float64():
    """
    Feature: test float64.
    Description: test RaggedRange when input is float64.
    Expectation: result is right.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    raggedrange = RaggedRange(Tsplits=mstype.int32)
    starts = Tensor([2.3, -4, 8.0, 1.5], mstype.float64)
    limits = Tensor([5.5, -1, 1.0, -1], mstype.float64)
    deltas = Tensor([1.2, 1.5, -1.0, -18.9], mstype.float64)
    (rt_nested_splits, rt_dense_values) = raggedrange(starts, limits, deltas)
    assert rt_nested_splits.asnumpy().dtype == 'int32'
    assert rt_dense_values.asnumpy().dtype == 'float64'
    rt_nested_splits_expected = np.array([0, 3, 5, 12, 13], np.int32)
    np.testing.assert_array_equal(rt_nested_splits.asnumpy(), rt_nested_splits_expected)
    rt_dense_values_expected = np.array([2.3, 3.5, 4.7, -4.0, -2.5, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5], np.float64)
    np.allclose(rt_dense_values.asnumpy(), rt_dense_values_expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_negative_delta():
    """
    Feature: test negative deltas.
    Description: test RaggedRange when deltas < 0.
    Expectation: result is right.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    raggedrange = RaggedRange(Tsplits=mstype.int32)
    starts = Tensor([5, 6], mstype.int32)
    limits = Tensor([0, 0], mstype.int32)
    deltas = Tensor([-2, -3], mstype.int32)
    (rt_nested_splits, rt_dense_values) = raggedrange(starts, limits, deltas)
    assert rt_nested_splits.asnumpy().dtype == 'int32'
    assert rt_dense_values.asnumpy().dtype == 'int32'
    rt_nested_splits_expected = np.array([0, 3, 5], np.int32)
    np.testing.assert_array_equal(rt_nested_splits.asnumpy(), rt_nested_splits_expected)
    rt_dense_values_expected = np.array([5, 3, 1, 6, 3], np.int32)
    np.allclose(rt_dense_values.asnumpy(), rt_dense_values_expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_empty_result():
    """
    Feature: test RaggedRange empty result.
    Description: test RaggedRange when result is empty.
    Expectation: result is empty.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    raggedrange = RaggedRange(Tsplits=mstype.int32)
    starts = Tensor([5, 6], mstype.int32)
    limits = Tensor([1, 1], mstype.int32)
    deltas = Tensor([2, 2], mstype.int32)
    (rt_nested_splits, rt_dense_values) = raggedrange(starts, limits, deltas)
    assert rt_nested_splits.asnumpy().dtype == 'int32'
    assert rt_dense_values.asnumpy().dtype == 'int32'
    rt_nested_splits_expected = np.array([0, 0, 0], np.int32)
    np.testing.assert_array_equal(rt_nested_splits.asnumpy(), rt_nested_splits_expected)
    rt_dense_values_expected = np.array([], np.int32)
    np.allclose(rt_dense_values.asnumpy(), rt_dense_values_expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_empty_input():
    """
    Feature: test RaggedRange empty input.
    Description: test RaggedRange when input is empty.
    Expectation: result is empty.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    raggedrange = RaggedRange(Tsplits=mstype.int64)
    starts = Tensor(())
    limits = Tensor(())
    deltas = Tensor(())
    (rt_nested_splits, rt_dense_values) = raggedrange(starts, limits, deltas)
    rt_nested_splits_expected = np.array([0], np.int32)
    rt_dense_values_expected = np.array([], np.int32)
    np.testing.assert_array_equal(rt_nested_splits.asnumpy(), rt_nested_splits_expected)
    np.allclose(rt_dense_values.asnumpy(), rt_dense_values_expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_neither_0d_nor_1d():
    """
    Feature: test RaggedRange with tensor is neither 0d nor 1d.
    Description: test RaggedRange when input tensor is neither 0d nor 1d.
    Expectation: should raise ValueError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    raggedrange = RaggedRange(Tsplits=mstype.int32)
    with pytest.raises(ValueError):
        _ = raggedrange(Tensor([[1], [2]], mstype.int32), Tensor([1], mstype.int32),
                        Tensor([8], mstype.int32))
        _ = raggedrange(Tensor([1], mstype.int32), Tensor([[1], [2]], mstype.int32),
                        Tensor([8], mstype.int32))
        _ = raggedrange(Tensor([1], mstype.int32), Tensor([1], mstype.int32),
                        Tensor([[1], [2]], mstype.int32))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_invalid_arg_zero_delta():
    """
    Feature: test RaggedRange with zero delta.
    Description: test RaggedRange when delta is zero.
    Expectation: should raise ValueError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    raggedrange = RaggedRange(Tsplits=mstype.int32)
    with pytest.raises(ValueError):
        _ = raggedrange(Tensor(1, mstype.int32), Tensor(10, mstype.int32),
                        Tensor(0, mstype.int32))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_shapes_mismatch():
    """
    Feature: test RaggedRange with mismatch shapes.
    Description: test RaggedRange when shapes are mismatch.
    Expectation: should raise ValueError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    raggedrange = RaggedRange(Tsplits=mstype.int32)
    with pytest.raises(ValueError):
        _ = raggedrange(Tensor(1, mstype.int32), Tensor(3, mstype.int32),
                        Tensor([2, 5], mstype.int32))
        _ = raggedrange(Tensor(1, mstype.int32), Tensor([1, 2], mstype.int32),
                        Tensor([2, 5], mstype.int32))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_not_tensor():
    """
    Feature: test RaggedRange with non tensor input.
    Description: test RaggedRange when input is not a tensor.
    Expectation: should raise TypeError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    raggedrange = RaggedRange(Tsplits=mstype.int32)
    with pytest.raises(TypeError):
        _ = raggedrange(1, Tensor(1, mstype.int32),
                        Tensor(2, mstype.int32))
        _ = raggedrange(Tensor(1, mstype.int32), 1,
                        Tensor(2, mstype.int32))
        _ = raggedrange(Tensor(1, mstype.int32), Tensor(5, mstype.int32), 2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_not_the_same_type():
    """
    Feature: test RaggedRange with different input types.
    Description: test RaggedRange when input tensors are with different types.
    Expectation: should raise TypeError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    raggedrange = RaggedRange(Tsplits=mstype.int32)
    with pytest.raises(TypeError):
        _ = raggedrange(Tensor(1, mstype.int32), Tensor(3, mstype.float32),
                        Tensor(10, mstype.int32))
        _ = raggedrange(Tensor(1, mstype.float64), Tensor(3, mstype.int32),
                        Tensor(10, mstype.int64))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ragged_range_tsplits_invalid_type():
    """
    Feature: test RaggedRange with invalid type.
    Description: test RaggedRange when ts_splits is with invalid type.
    Expectation: should raise TypeError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    with pytest.raises(TypeError):
        _ = RaggedRange(Tsplits=mstype.float32)
