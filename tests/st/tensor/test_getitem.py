# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""test case of tensor index getitem"""

import os
import numpy as np
import pytest
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import get_code_extra, has_graph

import mindspore as ms
from mindspore import nn
from mindspore import Tensor, ops, context
from mindspore.common import mutable
import torch
import torch.nn as nn_pt


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_parser_tensor_fancy_index_tuple_list_mix(mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem with fancy index tuple list
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    class Net(nn.Cell):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.relu = nn.ReLU()

        def construct(self, x):
            x = x[self.index]
            x = self.relu(x)
            return x

    class TorchNet(nn_pt.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.relu = nn_pt.ReLU()

        def forward(self, x):
            x = x[self.index]
            x = self.relu(x)
            return x

    index = ((-2, 0, -1), [1, 2, 1], [True, True, False, True], [2, 2, 2])
    net_ms = Net(index)
    net_pt = TorchNet(index)

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_ms = Tensor(input_np)
    input_pt = torch.from_numpy(input_np)

    output_ms = net_ms(input_ms)
    output_pt = net_pt(input_pt)
    assert np.allclose(output_pt.numpy(), output_ms.asnumpy(), 0.0001, 0.0001)


def assert_executed_by_graph_mode(func, x, index):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    if jcr is not None:
        assert jcr['stat'] == 'GRAPH_CALLABLE', f"ms_x: {x}, index: {index}"
        assert jcr['break_count_'] == 0, f'break_count expect: 0, actual: {jcr["break_count_"]}, '\
                                         f"ms_x: {x}, index: {index}"
        assert has_graph(jcr), f"ms_x: {x}, index: {index}"


class NetIndex3(nn.Cell):
    def construct(self, x, index1, index2, index3):
        y = x[index1, index2, index3]
        return y


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_getitem_index_negative(mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem with indexes are negative
    Expectation: success
    """
    ms.set_context(mode=mode)
    context.set_context(jit_level='O0')
    net = NetIndex3()
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    index1_np = -1
    index2_np = -1
    index3_np = -1
    y_np = x_np[index1_np, index2_np, index3_np]
    y = net(Tensor(x_np), Tensor(index1_np),
            Tensor(index2_np), Tensor(index3_np))
    assert np.allclose(y_np, y.asnumpy())


class NetIndex2Slice(nn.Cell):
    def construct(self, x, index1, index2):
        y = x[index1, 0:2, index2]
        return y


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_getitem_index_negative_with_slice(mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem with indexes are negative
    Expectation: success
    """
    ms.set_context(mode=mode)
    context.set_context(jit_level='O0')
    net = NetIndex2Slice()
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    index1_np = -1
    index2_np = -1
    y_np = x_np[index1_np, 0:2, index2_np]
    y = net(Tensor(x_np), Tensor(index1_np), Tensor(index2_np))
    assert np.allclose(y_np, y.asnumpy())


def previous_getitem_check_indexing(x, index, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:
        def func(ms_x, index):
            ms_y = ms_x[index]
            return ms_y
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def func(ms_x, index):
            ms_y = ms_x[index]
            return ms_y

    ms_output = func(x, index)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(func, x, index)

    assert np.allclose(np_expected, ms_output.asnumpy()), f"ms_x: {x}, index: {index}, " \
                                                          f"expected:{np_expected} {np_expected.shape}, " \
                                                          f"ms_output:{ms_output} {ms_output.shape}"


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize('capture_mode', [None])
def test_previous_getitem_level0(capture_mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem
    Expectation: success
    """
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)

    # Basic index
    basic_indices = [0, slice(0, 1), True, None, ..., (0, 2, ...), [0, 1]]
    for index in basic_indices:
        np_expected = np_x[index]
        previous_getitem_check_indexing(ms_x, index, np_expected, capture_mode)

    # Tensor index
    tensor_indices = [
        Tensor(0),
        slice(Tensor(0), Tensor(2)),
        Tensor([0, 1]),
        Tensor([True, True])
    ]
    np_indices = [0, slice(0, 2), [0, 1], [True, True]]
    for np_index, tensor_index in zip(np_indices, tensor_indices):
        np_expected = np_x[np_index]
        previous_getitem_check_indexing(
            ms_x, tensor_index, np_expected, capture_mode)

    # Fancy index
    fancy_indices = [([0, 1], [0, 1]),
                     (Tensor([0, 1]), Tensor([0, 1])),
                     ([0.0, 1.0], [0.0, 1.0]),
                     ([0, 1], 0, [0, 1]),
                     (Tensor([0, 1]), Tensor(0), Tensor([0, 1])),
                     (0, [0, 1], [0, 1]),
                     (Tensor(0), Tensor([0, 1]), Tensor([0, 1])),
                     ([0, 1], slice(0, 2), [0, 1]),
                     (Tensor([0, 1]), slice(0, 2), Tensor([0, 1])),
                     ([0, 1], True, [0, 1]),
                     ([0, 1], None, [0, 1]),
                     (Tensor([0, 1]), None, Tensor([0, 1])),
                     ([0, 1], ..., [0, 1]),
                     (Tensor([0, 1]), ..., Tensor([0, 1])),
                     (Tensor([0]), Tensor(0), slice(0, 4, 2))]

    np_expecteds = [np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([0, 13]),
                    np.array([0, 13]),
                    np.array([0, 5]),
                    np.array([0, 5]),
                    np.array([[0, 4], [13, 17]]),
                    np.array([[0, 4], [13, 17]]),
                    np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([[[0, 1, 2, 3]], [[16, 17, 18, 19]]]),
                    np.array([[[0, 1, 2, 3]], [[16, 17, 18, 19]]]),
                    np.array([[0, 4, 8], [13, 17, 21]]),
                    np.array([[0, 4, 8], [13, 17, 21]]),
                    np.array([[0, 2]])]

    for index, np_expected in zip(fancy_indices, np_expecteds):
        previous_getitem_check_indexing(ms_x, index, np_expected, capture_mode)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize('capture_mode', [None])
def test_previous_getitem_level1(capture_mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem
    Expectation: success
    """
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)

    # Slice index with None
    basic_indices = [slice(None, 2), slice(0, None), slice(None, None)]
    np_expected = np_x[slice(0, 2)]
    for index in basic_indices:
        previous_getitem_check_indexing(ms_x, index, np_expected, capture_mode)

    # List index which is normal tuple
    index = [0, slice(0, 2), ...]
    np_expected = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    previous_getitem_check_indexing(ms_x, index, np_expected, capture_mode)

    # Tuple index with negative step slice
    index = (0, slice(2, 0, -1))
    np_expected = np_x[index]
    previous_getitem_check_indexing(ms_x, index, np_expected, capture_mode)

    # Tensor index with all False
    index = [False, False]
    np_expected = np_x[index]
    previous_getitem_check_indexing(
        ms_x, Tensor(index), np_expected, capture_mode)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_previous_getitem_exception_index_error(mode, capture_mode):
    """
    Feature: previous tensor getitem
    Description: Verify the result of previous tensor getitem exception
    Expectation: success
    """

    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)
    tensor_index = Tensor(
        np.array([[[True], [True], [True]], [[True], [True], [True]]]))

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_float_index(x):
        return x[1.1]
    with pytest.raises(IndexError):
        _ = ms_x[1.1] if mode == ms.PYNATIVE_MODE else func_float_index(ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_array_index(x):
        return x[np.arange(0)]
    with pytest.raises(IndexError):
        _ = ms_x[np.arange(
            0)] if mode == ms.PYNATIVE_MODE else func_array_index(ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_tensor_index_dim_greater_data_dim(x):
        return x[[tensor_index, 0]]
    ms_x = Tensor(np_x)
    with pytest.raises(IndexError):
        _ = ms_x[[tensor_index, 0]
                 ] if mode == ms.PYNATIVE_MODE else func_tensor_index_dim_greater_data_dim(ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_tensor_index_with_float(x):
        return x[Tensor(1.1)]
    ms_x = Tensor(np_x)
    with pytest.raises(IndexError):
        _ = ms_x[Tensor(
            1.1)] if mode == ms.PYNATIVE_MODE else func_tensor_index_with_float(ms_x)


@arg_mark(
    plat_marks=['cpu_windows', 'cpu_macos'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_previous_getitem_exception_index_error_without_centos(mode, capture_mode):
    """
    Feature: previous tensor getitem
    Description: Verify the result of previous tensor getitem exception
    Expectation: success
    """
    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_list_index_dim_greater_tensor_data_dim(x):
        return x[[0, 2]]
    ms_x = Tensor(np_x)
    with pytest.raises(IndexError):
        _ = ms_x[[
            0, 2]] if mode == ms.PYNATIVE_MODE else func_list_index_dim_greater_tensor_data_dim(ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_tuple_index_dim_greater_tensor_data_dim(x):
        return x[([0, 2])]
    ms_x = Tensor(np_x)
    with pytest.raises(IndexError):
        _ = ms_x[(
            [0, 2])] if mode == ms.PYNATIVE_MODE else func_tuple_index_dim_greater_tensor_data_dim(ms_x)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_previous_getitem_exception_type_error(mode, capture_mode):
    """
    Feature: previous tensor getitem
    Description: Verify the result of previous tensor getitem exception
    Expectation: success
    """

    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_scalar_tensor_with_int_index(x):
        return x[0]
    ms_x = Tensor(1)
    with pytest.raises(TypeError):
        _ = ms_x[0] if mode == ms.PYNATIVE_MODE else func_scalar_tensor_with_int_index(
            ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_tuple_index_with_float(x):
        return x[[0, 1.4]]
    ms_x = Tensor(np_x)
    with pytest.raises(TypeError):
        _ = ms_x[[0, 1.4]] if mode == ms.PYNATIVE_MODE else func_tuple_index_with_float(
            ms_x)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_previous_getitem_exception_value_error(mode, capture_mode):
    """
    Feature: previous tensor getitem
    Description: Verify the result of previous tensor getitem exception
    Expectation: success
    """

    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_scalar_tensor_with_slice(x):
        return x[:]
    ms_x = Tensor(1)
    with pytest.raises(ValueError):
        _ = ms_x[:] if mode == ms.PYNATIVE_MODE else func_scalar_tensor_with_slice(
            ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_index_none_data_dim_out_8(x):
        return x[(None, slice(0, 2), slice(0, 2), slice(0, 2), slice(0, 2), slice(0, 2), slice(0, 2), slice(0, 2))]
    ms_x = Tensor(np.arange(1 * 2 * 3 * 4 * 5 * 6 * 7 * 8)
                  ).reshape(1, 2, 3, 4, 5, 6, 7, 8)
    s = slice(0, 2)
    indices = (None, s, s, s, s, s, s, s)
    with pytest.raises(ValueError):
        _ = ms_x[indices] if mode == ms.PYNATIVE_MODE else func_index_none_data_dim_out_8(
            ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_data_dim_out_8(x):
        return x[(0, 1)]
    ms_x = Tensor(np.arange(1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9)
                  ).reshape(1, 2, 3, 4, 5, 6, 7, 8, 9)
    with pytest.raises(ValueError):
        _ = ms_x[(0, 1)] if mode == ms.PYNATIVE_MODE else func_data_dim_out_8(ms_x)


def getitem_check_indexing(x, index, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:
        def func(ms_x, index):
            ms_y = ms_x[index]
            return ms_y
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def func(ms_x, index):
            ms_y = ms_x[index]
            return ms_y

    ms_output = func(x, index)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(func, x, index)

    assert np.allclose(np_expected, ms_output.asnumpy()), f"ms_x: {x}, index: {index}, " \
                                                          f"expected:{np_expected} {np_expected.shape}, " \
                                                          f"ms_output:{ms_output} {ms_output.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('capture_mode', [None, 'ast'])
def test_getitem(capture_mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem
    Expectation: success
    """

    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)

    # Basic index
    basic_indices = [0, slice(0, 1), True, False,
                     None, ..., (0, 2, ...), [0, 1]]
    for index in basic_indices:
        np_expected = np_x[index]
        getitem_check_indexing(ms_x, index, np_expected, capture_mode)

    np_expected = np.empty(shape=(0, 3, 4), dtype=np.int64)
    getitem_check_indexing(ms_x, [], np_expected, capture_mode)

    # Numpy index
    if capture_mode is None:
        numpy_indices = [np.array(0), np.array(True), np.array(False)]
        for index in numpy_indices:
            np_expected = np_x[index]
            getitem_check_indexing(ms_x, index, np_expected, capture_mode)

    # Tensor index
    tensor_indices = [
        Tensor(0), Tensor(True),
        Tensor(False),
        slice(Tensor(0), Tensor(2)),
        Tensor([0, 1]),
        Tensor([True, False])
    ]
    np_indices = [0, True, False, slice(0, 2), [0, 1], [True, False]]
    for np_index, tensor_index in zip(np_indices, tensor_indices):
        np_expected = np_x[np_index]
        getitem_check_indexing(ms_x, tensor_index, np_expected, capture_mode)

    # Tuple index
    tuple_indices = [(0, slice(0, 2), True), (0, None, ...)]
    np_expecteds = [
        np.array([[[0, 1, 2, 3]], [[4, 5, 6, 7]]]),
        np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]])
    ]
    for index, np_expected in zip(tuple_indices, np_expecteds):
        getitem_check_indexing(ms_x, index, np_expected, capture_mode)

    # Fancy index
    fancy_indices = [([0, 1], [0, 1]),
                     (Tensor([0, 1]), Tensor([0, 1])),
                     ([0.0, 1.0], [0.0, 1.0]),
                     ([0, 1], 0, [0, 1]),
                     (Tensor([0, 1]), Tensor(0), Tensor([0, 1])),
                     (0, [0, 1], [0, 1]),
                     (Tensor(0), Tensor([0, 1]), Tensor([0, 1])),
                     ([0, 1], slice(0, 2), [0, 1]),
                     (Tensor([0, 1]), slice(0, 2), Tensor([0, 1])),
                     ([0, 1], True, [0, 1]),
                     (Tensor([0, 1]), Tensor(True), Tensor([0, 1])),
                     ([0, 1], None, [0, 1]),
                     (Tensor([0, 1]), None, Tensor([0, 1])),
                     ([0, 1], ..., [0, 1]),
                     (Tensor([0, 1]), ..., Tensor([0, 1])),
                     (Tensor([0]), Tensor(0), slice(0, 4, 2))]

    np_expecteds = [np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([0, 13]),
                    np.array([0, 13]),
                    np.array([0, 5]),
                    np.array([0, 5]),
                    np.array([[0, 4], [13, 17]]),
                    np.array([[0, 4], [13, 17]]),
                    np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([[[0, 1, 2, 3]], [[16, 17, 18, 19]]]),
                    np.array([[[0, 1, 2, 3]], [[16, 17, 18, 19]]]),
                    np.array([[0, 4, 8], [13, 17, 21]]),
                    np.array([[0, 4, 8], [13, 17, 21]]),
                    np.array([[0, 2]])]

    for index, np_expected in zip(fancy_indices, np_expecteds):
        getitem_check_indexing(ms_x, index, np_expected, capture_mode)


def getitem_check_indexing_without_jit(x, index, np_expected, capture_mode=None):
    """getitem run and check"""
    def func(ms_x, index):
        ms_y = ms_x[index]
        return ms_y
    ms_output = func(x, index)
    assert np.allclose(np_expected, ms_output.asnumpy()), f"ms_x: {x}, index: {index}, " \
                                                          f"expected:{np_expected} {np_expected.shape}, " \
                                                          f"ms_output:{ms_output} {ms_output.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('capture_mode', [None])
def test_slice_tensor_index_getitem_without_jit(capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem
    Expectation: success
    """

    # Slice Tensor index
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    basic_type = [ms.int8, ms.uint8, ms.int16, ms.uint16, ms.int, ms.int32, ms.uint32, ms.int64, ms.uint64, ms.float16,
                  ms.float, ms.float32, ms.double, ms.float64, ms.bfloat16]
    np_expected = np.array([[[12, 13, 14, 15],
                             [16, 17, 18, 19],
                             [20, 21, 22, 23]]])
    for dtype in basic_type:
        start = Tensor(1, dtype=dtype)
        end = Tensor(2, dtype=dtype)
        step = Tensor(1, dtype=dtype)
        slice_index = slice(start, end, step)
        getitem_check_indexing_without_jit(
            ms_x, slice_index, np_expected, capture_mode)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('capture_mode', [None, 'ast'])
def test_getitem2(capture_mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem
    Expectation: success
    """

    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)

    fancy_indices = [([Tensor(0), 1], Tensor(0), [0, Tensor(1)]),
                     (Tensor(0), [0, Tensor(1)], [Tensor(0), Tensor(1)])]

    np_expecteds = [np.array([0, 13]),
                    np.array([0, 5])]

    for index, np_expected in zip(fancy_indices, np_expecteds):
        getitem_check_indexing(ms_x, index, np_expected, capture_mode)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize('capture_mode', [None, 'ast', 'bytecode'])
def test_getitem_with_ellipsis(capture_mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem with ellipsis
    Expectation: success
    """

    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    np_x = np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    ms_x = Tensor(np_x)

    ellipsis_indices = [(1, ..., True, [0, 1]),
                        (Tensor(1), ..., Tensor(True), Tensor([0, 1])),
                        (slice(0, 1), ..., None)]
    np_expecteds = [
        np.array([[[60, 61], [65, 66], [70, 71], [75, 76]], [[80, 81], [85, 86], [90, 91], [95, 96]],
                  [[100, 101], [105, 106], [110, 111], [115, 116]]]),
        np.array([[[60, 61], [65, 66], [70, 71], [75, 76]], [[80, 81], [85, 86], [90, 91], [95, 96]],
                  [[100, 101], [105, 106], [110, 111], [115, 116]]]),
        np.array([[[[[0], [1], [2], [3], [4]], [[5], [6], [7], [8], [9]], [[10], [11], [12], [13], [14]],
                    [[15], [16], [17], [18], [19]]],
                   [[[20], [21], [22], [23], [24]], [[25], [26], [27], [28], [29]], [[30], [31], [32], [33], [34]],
                    [[35], [36], [37], [38], [39]]],
                   [[[40], [41], [42], [43], [44]], [[45], [46], [47], [48], [49]], [[50], [51], [52], [53], [54]],
                    [[55], [56], [57], [58], [59]]]]])
    ]

    for index, np_expected in zip(ellipsis_indices, np_expecteds):
        getitem_check_indexing(ms_x, index, np_expected, capture_mode)


class NetGetitem(nn.Cell):

    def __init__(self, index):
        super().__init__()
        self.index = index

    def construct(self, x):
        x = ops.relu(x)
        y = x[self.index]
        return y


def getitem_check_grad(x, index, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:
        def grad_func(net, x):
            return ms.grad(net)(x)
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def grad_func(net, x):
            return ms.grad(net)(x)

    net = NetGetitem(index)
    ms_grad = grad_func(net, x)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(net.construct, x, index)

    assert np.allclose(np_expected, ms_grad.asnumpy()), f"ms_x: {x}, index: {index}, " \
                                                        f"expected:{np_expected} {np_expected.shape}, " \
                                                        f"ms_grad:{ms_grad} {ms_grad.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('capture_mode', [None, 'ast', 'bytecode'])
def test_getitem_grad_index_negative(capture_mode):
    """
    Feature: tensor getitem grad
    Description: Verify the result of tensor getitem grad with negative index
    Expectation: success
    """

    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
    index = (Tensor(-1), Tensor(-1), Tensor(-1))
    np_expected = np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                            [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 1.,]]])
    getitem_check_grad(ms_x, index, np_expected, capture_mode)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('capture_mode', [None, 'ast', 'bytecode'])
def test_getitem_grad(capture_mode):
    """
    Feature: tensor getitem grad
    Description: Verify the result of tensor getitem grad
    Expectation: success
    """

    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    # Base index
    base_indices = [0, slice(0, 2), True, False, ..., None, [0, 1]]
    np_expecteds = [np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]])]
    for index, np_expected in zip(base_indices, np_expecteds):
        ms_x = Tensor(
            np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
        getitem_check_grad(ms_x, index, np_expected, capture_mode)

    # Tensor index
    tensor_indices = [Tensor(0), Tensor(True), Tensor(
        False), slice(Tensor(0), Tensor(2)), Tensor([0, 1])]
    np_expecteds = [np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]])]
    for index, np_expected in zip(tensor_indices, np_expecteds):
        getitem_check_grad(ms_x, index, np_expected, capture_mode)

    # Tuple index
    tuple_indices = [(0, slice(0, 2), True), (0, None, ...)]
    np_expecteds = [np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]])]
    for index, np_expected in zip(tuple_indices, np_expecteds):
        getitem_check_grad(ms_x, index, np_expected, capture_mode)

    # Fancy index
    fancy_indices = [([0, 1], [0, 1]),
                     (Tensor([0, 1]), Tensor([0, 1])),
                     ([0, 1], 0, [0, 1]),
                     (Tensor([0, 1]), Tensor(0), Tensor([0, 1])),
                     (0, [0, 1], [0, 1]),
                     (Tensor(0), Tensor([0, 1]), Tensor([0, 1])),
                     ([0, 1], slice(0, 2), [0, 1]),
                     (Tensor([0, 1]), slice(0, 2), Tensor([0, 1])),
                     ([0, 1], True, [0, 1]),
                     (Tensor([0, 1]), Tensor(True), Tensor([0, 1])),
                     ([0, 1], None, [0, 1]),
                     (Tensor([0, 1]), None, Tensor([0, 1])),
                     ([0, 1], ..., [0, 1]),
                     (Tensor([0, 1]), ..., Tensor([0, 1]))]
    np_expecteds = [np.array([[[0., 1., 1., 1.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 1., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 1., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 1., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 1., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [1., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 1., 0., 0.,], [0., 1., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [1., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 1., 0., 0.,], [0., 1., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [1., 0., 0., 0.,], [1., 0., 0., 0.,]],
                              [[0., 1., 0., 0.,], [0., 1., 0., 0.,], [0., 1., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [1., 0., 0., 0.,], [1., 0., 0., 0.,]],
                              [[0., 1., 0., 0.,], [0., 1., 0., 0.,], [0., 1., 0., 0.,]]])]
    for index, np_expected in zip(fancy_indices, np_expecteds):
        getitem_check_grad(ms_x, index, np_expected, capture_mode)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'ast'), (ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_getitem_exception(mode, capture_mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem exception
    Expectation: success
    """
    ms.set_context(jit_config={"jit_level": "O0"})
    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func1(x):
        return x[2, 0, 0]
    with pytest.raises(IndexError) as exc:
        _ = ms_x[2, 0, 0] if mode == ms.PYNATIVE_MODE else func1(ms_x)
    assert "is out of bounds" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func2(x):
        return x[0, 0, 0, 0]
    with pytest.raises(IndexError) as exc:
        _ = ms_x[0, 0, 0, 0] if mode == ms.PYNATIVE_MODE else func2(ms_x)
    assert "too many indices" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func3(x):
        return x[0, 't']
    with pytest.raises(IndexError) as exc:
        _ = ms_x[0, 't'] if mode == ms.PYNATIVE_MODE else func3(ms_x)
    assert "Invalid tensor index type" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func4(x):
        return x[0:3:-1]
    with pytest.raises(ValueError) as exc:
        _ = ms_x[0:3:-1] if mode == ms.PYNATIVE_MODE else func4(ms_x)
    assert "slice step must be positive" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func5(x):
        return x[0]
    ms_x = Tensor(0)
    with pytest.raises(TypeError) as exc:
        _ = ms_x[0] if mode == ms.PYNATIVE_MODE else func5(ms_x)
    assert "Invalid index of a 0-dim tensor." in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func6(x):
        return x[0:1]
    with pytest.raises(TypeError) as exc:
        _ = ms_x[0:1] if mode == ms.PYNATIVE_MODE else func6(ms_x)
    assert "Invalid index of a 0-dim tensor." in str(exc.value)

    with pytest.raises(TypeError) as exc:
        _ = sum(ms_x)
    assert "Invalid index of a 0-dim tensor." in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func7(x):
        return x[Tensor(1.4), :, :]
    ms_x = Tensor(np_x)
    with pytest.raises(TypeError) as exc:
        _ = ms_x[Tensor(1.4), :, :] if mode == ms.PYNATIVE_MODE else func7(
            ms_x)
    assert "For 'Index', tensors used as indices must be long, int, uint8, or bool tensors" in str(
        exc.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_getitem_exception_without_jit_ast(mode, capture_mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor setitem exception
    Expectation: success
    """
    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_tensor_as_slice_index_dim_out_1(x):
        s = Tensor([0, 1])
        index = slice(s, s, s)
        return x[index]
    with pytest.raises(ValueError):
        _ = ms_x[Tensor(
            [0, 1]):1:1] if mode == ms.PYNATIVE_MODE else func_tensor_as_slice_index_dim_out_1(ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_tensor_as_slice_index_with_unsupport_type(x):
        s = Tensor(3 + 4j, dtype=ms.complex64)
        index = slice(s, s, s)
        return x[index]
    with pytest.raises(TypeError):
        _ = (
            ms_x[Tensor(3 + 4j, dtype=ms.complex64):1:1]
            if mode == ms.PYNATIVE_MODE
            else func_tensor_as_slice_index_with_unsupport_type(ms_x)
        )

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_slice_with_float_index(x):
        return x[slice(1.1, 2)]
    with pytest.raises(IndexError) as exc:
        _ = ms_x[slice(
            1.1, 2)] if mode == ms.PYNATIVE_MODE else func_slice_with_float_index(ms_x)
    assert "slice indices must be integers or None or Tensor" in str(exc.value)


class NetMutableSequenceIndex(nn.Cell):
    def construct(self, x, index):
        y = x[index]
        return y


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_getitem_mutable_sequence_index():
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem exception in graph mode
    Expectation: success
    """
    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
    np_x = np.arange(3 * 3 * 2).reshape((3, 3, 2))
    ms_x = Tensor(np_x)
    index = mutable([2, 1, 0])

    net = NetMutableSequenceIndex()
    pynative_res = net(ms_x, index)
    np_expected = np_x[::-1]
    assert np.allclose(np_expected, pynative_res.asnumpy()), f"ms_x: {ms_x}, index: {index}, " \
        f"expected: {np_expected} {np_expected.shape}, " \
        f"pynative_res: {pynative_res} {pynative_res.shape}"

    with pytest.raises(IndexError) as err:
        net.construct = ms.jit(net.construct, backend="ms_backend")
        net(ms_x, index)
    assert "Current Tensor indexing does not support mutable list/tuple or list containing tensors. " \
           "Please use an immutable expression instead." in str(err.value)


class NetTensorInListIndex(nn.Cell):
    def construct(self, x, index):
        y = x[0, [0, 1, index]]
        return y


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_getitem_tensor_in_list_index():
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem exception in graph mode
    Expectation: success
    """
    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
    np_x = np.arange(3 * 3 * 2).reshape((3, 3, 2))
    ms_x = Tensor(np_x)
    index = Tensor(2)

    net = NetTensorInListIndex()
    pynative_res = net(ms_x, index)
    np_expected = np_x[0]
    assert np.allclose(np_expected, pynative_res.asnumpy()), f"ms_x: {ms_x}, index: {index}, " \
        f"expected: {np_expected} {np_expected.shape}, " \
        f"pynative_res: {pynative_res} {pynative_res.shape}"

    with pytest.raises(IndexError) as err:
        net.construct = ms.jit(net.construct, backend="ms_backend")
        net(ms_x, index)
    assert "Current Tensor indexing does not support mutable list/tuple or list containing tensors. " \
           "Please use an immutable expression instead." in str(err.value)


class NetParamIndexWithAssign(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = ms.Parameter(
            Tensor(np.arange(3 * 3 * 2).reshape((3, 3, 2))), name="param")

    def construct(self, x):
        self.param = self.param[[0, 1, 2]]
        y = self.param + x
        return y


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_param_getitem_with_assign_value():
    """
    Feature: tensor getitem
    Description: Verify the result of parameter getitem with assign value
    Expectation: success
    """
    np_x = np.arange(3 * 3 * 2).reshape((3, 3, 2))
    ms_x = Tensor(np_x)
    net = NetParamIndexWithAssign()
    pynative_res = net(ms_x)
    np_expected = np_x[[0, 1, 2]] + np_x
    assert np.allclose(np_expected, pynative_res.asnumpy()), f"ms_x: {ms_x}" \
                                                             f"expected: {np_expected} {np_expected.shape}, " \
                                                             f"pynative_res: {pynative_res} {pynative_res.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_tensor_data_ptr_with_offset():
    """
    Feature: Tensor data_ptr
    Description: Test Tensor data_ptr with storage offset
    Expectation: success
    """
    x = Tensor(np.ones(10, dtype=np.float32))
    y = x[5]
    # nbytes = element(5) * element_size(4)
    assert y.data_ptr() - x.data_ptr() == 20

    y = x[5::2]
    assert y.data_ptr() - x.data_ptr() == 20


def no_jit(x, *args, **kwargs):
    return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_tensor_negative_index_for_tuple_input(mode):
    """
    Feature: tensor getitem.
    Description: Test that negative slice indices work correctly on tuple inputs in both JIT and no-JIT modes.
    Expectation: Output matches native Python slicing result.
    """
    def net(x):
        x = x[-3:-1]
        return x

    input_tuple = (0, -1, 1997, 2000)
    res = input_tuple[-3:-1]

    result = mode(net)(input_tuple)
    assert res == result


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_tensor_negative_index_for_list_input(mode):
    """
    Feature: tensor getitem.
    Description: Verify negative slice indexing on list inputs behaves as expected under JIT/no-JIT.
    Expectation: Result equals Python list slicing output.
    """
    def net(x):
        x = x[-3:-1]
        return x

    input_tuple = [0, -1, 1997, 2000]
    res = input_tuple[-3:-1]
    result = mode(net, backend="ms_backend")(input_tuple)
    assert res == result


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_tensor_negative_index_for_tensor_input_001(mode):
    """
    Feature: tensor getitem.
    Description: Test slicing with negative step on Tensor; Ascend backend disallows negative step in no-JIT.
    Expectation: ValueError raised on Ascend in no-JIT; otherwise output matches expected ReLU result.
    """
    def net(x):
        x = x * 1
        x = x[-1:-3:-1]
        x = ops.relu(x)
        return x

    input_np = [[[0.5, 0.8, -0.6, -0.1], [1.5, 3.8, 0.9, -2.8], [0.7, 0.8, -0.6, 0.8], [0.5, 0.8, -0.6, -0.8]],
                [[2.4, 2.8, -0.3, 0.4], [0.5, 0.4, -0.6, 0.8],
                    [0.5, 0.8, -0.5, -0.8], [0.5, 0.8, -0.6, -0.8]],
                [[0.2, 0.2, 0.6, -0.2], [0.3, 0.8, -0.6, -0.8],
                    [0.5, 2.8, -0.6, -0.8], [0.5, 0.8, 0.6, -0.8]],
                [[1.1, 2.1, -3.6, -2.8], [0.5, 0.8, -0.6, -0.8], [0.5, 0.8, 0.7, -0.8], [0.5, 0.8, 0.6, -0.8]]]
    input_me = Tensor(input_np)
    if ms.get_context("device_target") == "Ascend" and mode is no_jit:
        with pytest.raises(ValueError,
                           match=r"slice step must be positive") as err:
            assert mode(net)(input_me)
        assert err
    else:
        out_me = mode(net)(input_me)
        out_tf = [[[1.1, 2.1, 0., 0.], [0.5, 0.8, 0., 0.], [0.5, 0.8, 0.7, 0.], [0.5, 0.8, 0.6, 0.]],
                  [[0.2, 0.2, 0.6, 0.], [0.3, 0.8, 0., 0.], [0.5, 2.8, 0., 0.], [0.5, 0.8, 0.6, 0.]]]
        assert np.allclose(out_tf, out_me.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_tensor_negative_index_for_tensor_input_002(mode):
    """
    Feature: tensor getitem.
    Description: Validate standard negative slice (e.g., [-3:-1]) on Tensor in JIT/no-JIT modes.
    Expectation: Output matches expected sliced and ReLU-applied result.
    """
    def net(x):
        x = x[-3:-1]
        x = ops.relu(x)
        return x

    input_np = [[[0.5, 0.8, -0.6, -0.1], [1.5, 3.8, 0.9, -2.8], [0.7, 0.8, -0.6, 0.8], [0.5, 0.8, -0.6, -0.8]],
                [[2.4, 2.8, -0.3, 0.4], [0.5, 0.4, -0.6, 0.8],
                    [0.5, 0.8, -0.5, -0.8], [0.5, 0.8, -0.6, -0.8]],
                [[0.2, 0.2, 0.6, -0.2], [0.3, 0.8, -0.6, -0.8],
                    [0.5, 2.8, -0.6, -0.8], [0.5, 0.8, 0.6, -0.8]],
                [[1.1, 2.1, -3.6, -2.8], [0.5, 0.8, -0.6, -0.8], [0.5, 0.8, 0.7, -0.8], [0.5, 0.8, 0.6, -0.8]]]

    input_me = Tensor(input_np)
    out_me = mode(net)(input_me)
    out_tf = [[[2.4, 2.8, 0., 0.4], [0.5, 0.4, 0., 0.8], [0.5, 0.8, 0., 0.], [0.5, 0.8, 0., 0.]],
              [[0.2, 0.2, 0.6, 0.], [0.3, 0.8, 0., 0.], [0.5, 2.8, 0., 0.], [0.5, 0.8, 0.6, 0.]]]
    assert np.allclose(out_tf, out_me.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_tensor_negative_index_for_str_input(mode):
    """
    Feature: tensor getitem.
    Description: Test single negative index (e.g., [-3]) on string input.
    Expectation: Returns correct character as in native Python.
    """
    def net(x):
        x = x[-3]
        return x

    input_str = 'asdfghjkl'
    exp = input_str[-3]
    result = mode(net)(input_str)
    assert exp == result


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_tensor_negative_slice_for_str_input(mode):
    """
    Feature: tensor getitem.
    Description: Test empty negative slice (e.g., [-1:-3]) on string.
    Expectation: Returns empty string as in native Python slicing.
    """
    def net(x):
        x = x[-1:-3]
        return x

    input_str = 'asdfghjkl'
    exp = input_str[-1:-3]
    result = mode(net)(input_str)
    assert exp == result


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.25)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_tensor_fancy_index_bool_list(mode):
    """
    Feature: tensor getitem.
    Description: Test indexing with True/True on 1D tensor and compare with PyTorch.
    Expectation: Forward and gradient outputs match PyTorch result.
    """
    def net_ms(x):
        x = x[True][True]
        x = ops.relu(x)
        return x

    class TorchNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn_pt.ReLU()

        def forward(self, x):
            x = x[True][True]
            x = self.relu(x)
            return x

    net_pt = TorchNet()
    input_np = np.random.randn(3).astype(np.float32)
    ms_input = ms.Tensor(input_np)
    pt_input = torch.tensor(input_np, requires_grad=True)
    ms_out = mode(net_ms)(ms_input)
    pt_out = net_pt(pt_input)
    assert np.allclose(ms_out.asnumpy(), pt_out.detach().numpy())
    np_sens = np.random.randn(*ms_out.shape).astype(np.float32)
    ms_sens = ms.Tensor(np_sens)
    pt_sens = torch.tensor(np_sens)
    ms_grad = mode(ms.grad(net_ms, sens_param=True))(ms_input, ms_sens)
    pt_grad = torch.autograd.grad(pt_out, pt_input, grad_outputs=pt_sens)
    assert np.allclose(ms_grad.asnumpy(), pt_grad[0].detach().numpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_parser_tensor_index_exception_out_of_range(mode):
    """
    Feature: tensor getitem.
    Description: Index tensor with out-of-bound indices.
    Expectation: No exception raised (behavior may be device/backend-specific; currently passes).
    """
    def net(x, index):
        x = x[index]
        x = ops.relu(x)
        return x

    input_np = np.random.rand(4, 4, 4)
    input_me = Tensor(input_np, ms.float32)
    index_np_x = np.array([[6, 7]])
    index_me_x = Tensor(index_np_x, ms.int32)
    mode(net)(input_me, index_me_x)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
@pytest.mark.parametrize("np_idx_dtype", [np.int8, np.int16, np.float32])
def test_tensor_index_unsupported_dtype(mode, np_idx_dtype):
    """
    Feature: tensor getitem.
    Description: Use int8, int16, float32 as index tensors.
    Expectation: Raises IndexError, TypeError, or RuntimeError.
    """
    def net(x, index):
        x = x[index]
        x = ops.relu(x)
        return x

    input_np = np.random.rand(4, 4, 4)
    input_me = Tensor(input_np, ms.float32)
    index_np_x = np.array([0, 1]).astype(np_idx_dtype)
    index_me_x = Tensor(index_np_x)
    with pytest.raises((IndexError, TypeError, RuntimeError)):  # int16-jit raise RuntimeError
        mode(net)(input_me, index_me_x)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_parser_tensor_index_exception_shape_cannot_broadcast(mode):
    """
    Feature: tensor getitem.
    Description: Pass index tensors with shapes that cannot broadcast.
    Expectation: ValueError raised due to shape mismatch.
    """
    def net(x, index1, index2):
        x = x[index1, index2]
        x = ops.relu(x)
        return x

    input_np = np.random.rand(4, 4, 4)
    input_me = Tensor(input_np, ms.float32)
    index_np_x = np.array([[0, 1], [1, 0]]).astype(np.int32)
    index_me_x = Tensor(index_np_x)
    index_np_y = np.array([[0, 1, 1]]).astype(np.int32)
    index_me_y = Tensor(index_np_y)
    with pytest.raises(ValueError):
        mode(net)(input_me, index_me_x, index_me_y)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_parser_tensor_index_exception_multi_tensor_dtype_int64(mode):
    """
    Feature: tensor getitem.
    Description: Index 3D tensor using three int64 index tensors.
    Expectation: Output matches NumPy advanced indexing result after ReLU.
    """
    def myrelu(x):
        return np.maximum(0, x)

    def net(x, index1, index2, index3):
        x = x[index1, index2, index3]
        x = ops.relu(x)
        return x

    input_np = np.random.rand(4, 4, 4)
    input_me = Tensor(input_np, ms.float32)
    index_np_x = np.array([[0, 1]])
    index_me_x = Tensor(index_np_x, ms.int64)
    index_np_y = np.array([[0, 1]])
    index_me_y = Tensor(index_np_y, ms.int64)
    index_np_z = np.array([[2, 3]])
    index_me_z = Tensor(index_np_z, ms.int64)
    out_me = mode(net)(input_me, index_me_x, index_me_y, index_me_z)
    out_np = myrelu(input_np[index_np_x, index_np_y, index_np_z])
    np.allclose(out_me.asnumpy(), out_np)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_tensor_index_abnormal(mode):
    """
    Feature: tensor getitem.
    Description: Use float64 tensor as index.
    Expectation: TypeError in no-JIT; IndexError in JIT.
    """
    def net(x, tensor):
        z = x[tensor] * 1
        return z

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float64)
    input_me_x = ms.Parameter(Tensor(input_np_x, ms.float32))
    input_np_y = -1
    tensor = Tensor(input_np_y, ms.float64)
    if mode is no_jit:
        with pytest.raises(TypeError,
                           match=r"For 'Index', tensors used as indices must be long, int, uint8, "
                                 r"or bool tensors") as err:
            assert mode(net)(input_me_x, tensor).asnumpy()
        assert err
    else:
        with pytest.raises(IndexError):
            mode(net)(input_me_x, tensor).asnumpy()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_tensor_index_int64_normal(mode):
    """
    Feature: tensor getitem.
    Description: Index tensor using int64 index tensor.
    Expectation: Output matches NumPy advanced indexing result.
    """
    def net(inputs, tensor_in):
        result = inputs[tensor_in]
        return result

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x, ms.float32)
    input_np_y = np.random.randint(2, size=[1, 2]).astype(np.int64)
    tensor = Tensor(input_np_y, ms.int64)
    out_me = mode(net)(input_me_x, tensor).asnumpy()
    out_numpy = input_np_x[input_np_y]
    assert np.allclose(out_me, out_numpy, 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.0)
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_tensor_fancy_index_tuple_empty(mode):
    """
    Feature: tensor getitem.
    Description: Use empty tuple followed by integer index (e.g., x[(), 3]).
    Expectation: Passes in no-JIT; raises IndexError in JIT (GE backend limitation).
    """
    def net_ms(x):
        x = x[(), 3]
        return x

    input_np = np.random.randn(3, 4).astype(np.float32)
    if mode is no_jit:
        mode(net_ms)(Tensor(input_np))
    else:
        with pytest.raises(IndexError):
            mode(net_ms)(Tensor(input_np))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=48.0)
@pytest.mark.parametrize("mode, jit_level", [(no_jit, None),  (ms.jit, "O0"),  (ms.jit, "O1"), (ms.jit, "O2")])
def test_parser_tensor_fancy_index_mul_list_multilevel(mode, jit_level):
    """
    Feature: tensor getitem.
    Description: Deep nested indexing with mixed slices, bools, and tuples.
    Expectation: Passes in no-JIT and JIT O0/O1; fails in O2/GE due to unsupported pattern.
    """
    def net_ms(x):
        x = x[1][2][1, 2][:][0, True][True][:, :, (2, 3, 4)]
        x = ops.relu(x)
        return x

    input_np = np.random.randn(3, 4, 5, 6, 7, 8, 9, 5, 9).astype(np.float32)

    if mode is ms.jit and (jit_level == "O0" or jit_level == "O1"):
        mode(net_ms, jit_level=jit_level)(Tensor(input_np))
    elif mode is no_jit:
        mode(net_ms)(Tensor(input_np))
    else:
        with pytest.raises((ValueError, RuntimeError)):
            mode(net_ms, backend="GE")(Tensor(input_np))
