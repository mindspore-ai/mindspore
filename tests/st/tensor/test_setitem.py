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
"""test case of tensor index setitem"""

import os
import numpy as np
import pytest
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import get_code_extra, has_graph

import mindspore as ms
from mindspore import nn
from mindspore import Tensor, Parameter, ops
from mindspore import dtype as mstype
import torch
import torch.nn as nn_pt


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_tensor_fancy_index_set_item_032(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem with fancy index tuple list
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    class Net(nn.Cell):
        """test setitem use net"""

        def __init__(self, index, value):
            super().__init__()
            self.index = index
            self.value = value
            self.relu = nn.ReLU()

        def construct(self, input_x):
            input_x[self.index] = self.value
            out = self.relu(input_x)
            return out

    class NetPytorch1(nn_pt.Module):
        """test setitem use net"""

        def __init__(self, index, value):
            super().__init__()
            self.index = index
            self.value = value
            self.relu = nn_pt.ReLU()

        def forward(self, input_x):
            input_x[self.index] = self.value
            out = self.relu(input_x)
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    value_np = np.ones((2, 3, 4, 5), np.float32)

    index = (slice(None, None), ..., [True, True, True, True], slice(0, 5))
    value_mx = Tensor(value_np)
    value_pt = torch.tensor(value_np)
    net_ms = Net(index, value_mx)
    net_pt = NetPytorch1(index, value_pt)

    input_ms = Tensor(input_np)
    input_pt = torch.from_numpy(input_np)

    output_ms = net_ms(input_ms)
    output_pt = net_pt(input_pt)

    assert np.allclose(output_pt.numpy(), output_ms.asnumpy(), 0.001, 0.001)


def assert_executed_by_graph_mode(func, x, index, value):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    if jcr is not None:
        assert jcr['stat'] == 'GRAPH_CALLABLE', f"ms_x: {x}, index: {index}, value: {value}"
        assert jcr['break_count_'] == 0, f'break_count expect: 0, actual: {jcr["break_count_"]}, '\
                                         f"ms_x: {x}, index: {index}, value: {value}"
        assert has_graph(jcr), f"ms_x: {x}, index: {index}, value: {value}"


def is_index_need_skip(index, skip_list):
    """check if index need skip, used for debug"""
    def check_index_same(index, to_skip):
        if type(to_skip) != type(index):  # pylint: disable=unidiomatic-typecheck
            return False
        if isinstance(to_skip, Tensor):
            result = (index == to_skip).all()
        elif isinstance(to_skip, (tuple, list)):
            result = all(check_index_same(
                index[i], to_skip[i]) for i in range(len(to_skip)))
        else:
            result = index == to_skip
        return bool(result)

    for to_skip in skip_list:
        if check_index_same(index, to_skip):
            return True

    return False


def previous_setitem_check_indexing(x, index, value, np_expected, capture_mode=None):
    """previous setitem run and check"""
    if capture_mode is None:
        def func(ms_x, index, value):
            ms_x[index] = value
            return ms_x
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def func(ms_x, index, value):
            ms_x[index] = value
            return ms_x

    ms_output = func(x, index, value)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(func, x, index, value)

    assert np.allclose(np_expected, ms_output.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                          f"expected:{np_expected} {np_expected.shape}, " \
                                                          f"ms_output:{ms_output} {ms_output.shape}"


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize('capture_mode', [None])
def test_previous_setitem_level0(capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of previous tensor setitem
    Expectation: success
    """

    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)

    # Basic index
    basic_indices = [0, slice(0, 1), True, None, ..., (0, 2, ...), [0, 1]]
    for index in basic_indices:
        np_x = np.arange(2*3*4).reshape(2, 3, 4)
        ms_x = Tensor(np_x)
        value = -1
        np_x[index] = value
        previous_setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Basic index with bool value
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    index = slice(0, 1)
    value = False
    np_x[index] = value
    previous_setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Basic index with bool value
    np_x = np.array([False, True])
    ms_x = Tensor(np_x)
    index = 0
    value = True
    np_x[index] = value
    previous_setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Basic index with float value
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    index = [0, 1]
    value = -1.0
    np_x[index] = value
    previous_setitem_check_indexing(ms_x, index, value, np_x, capture_mode)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize('capture_mode', [None])
def test_previous_setitem_level1(capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of previous tensor setitem
    Expectation: success
    """
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)

    # Tensor index which shape.size() == 1
    index = Tensor([1])
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    np_expected = np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                            [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]])
    value = -1
    previous_setitem_check_indexing(
        ms_x, index, value, np_expected, capture_mode)

    # Tensor index with bool
    index = Tensor([[True], [True]])
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    np_expected = np.array([[[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                            [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]])
    value = -1
    previous_setitem_check_indexing(
        ms_x, index, value, np_expected, capture_mode)

    # Slice index with value is list
    index = slice(0, 2)
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    np_expected = np.array([[[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                            [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]])
    value = [-1]
    previous_setitem_check_indexing(
        ms_x, index, value, np_expected, capture_mode)

    # Slice index with invalid start stop and step
    index = slice(0, 2, -1)
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    value = -1
    previous_setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Int index which source tensor.shape = (1.) and value type is tensor
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor([3])
    np_expected = np.array([4])
    value = Tensor([4])
    previous_setitem_check_indexing(ms_x, 0, value, np_expected, capture_mode)

    # Tuple index can use tensor inner
    index = [0, slice(None, None, 1), slice(None, None, 1)]
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    np_expected = np.array([[[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    value = -1
    previous_setitem_check_indexing(
        ms_x, index, value, np_expected, capture_mode)


class NetSetitemCpu(nn.Cell):
    """test setitem use net"""

    def __init__(self, index, value):
        super().__init__()
        self.index = index
        self.value = value

    def construct(self, x):
        x = ops.abs(x)
        x[self.index] = self.value
        return x


def previous_setitem_check_grad(x, index, value, np_expected, capture_mode=None):
    """setitem run and check"""
    if capture_mode is None:
        def grad_func(net, x):
            return ms.grad(net)(x)
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def grad_func(net, x):
            return ms.grad(net)(x)

    net = NetSetitemCpu(index, value)
    ms_grad = grad_func(net, x)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(net.construct, x, index, value)

    assert np.allclose(np_expected, ms_grad.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                        f"expected:{np_expected} {np_expected.shape}, " \
                                                        f"ms_grad:{ms_grad} {ms_grad.shape}"


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize('capture_mode', [None])
def test_previous_setitem_with_grad_on_cpu(capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of previous tensor setitem grad
    Expectation: success
    """
    index = 0
    np_expected = np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                            [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
    value = -1
    previous_setitem_check_grad(ms_x, index, value, np_expected)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_previous_setitem_exception_index_error(mode, capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of previous tensor setitem exception
    Expectation: success
    """

    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_tensor_index_with_float(x):
        x[(Tensor(1.4), 1)] = -1
        return x
    ms_x = Tensor(np_x)
    with pytest.raises(IndexError):
        if mode == ms.PYNATIVE_MODE:
            ms_x[(Tensor(1.4), 1)] = -1
        else:
            _ = func_tensor_index_with_float(ms_x)


@arg_mark(
    plat_marks=['cpu_windows', 'cpu_macos'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_previous_setitem_exception_index_error_without_centos(mode, capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of previous tensor setitem exception
    Expectation: success
    """

    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_list_index_dim_out_data_dim(x):
        x[[0, 2]] = -1
        return x
    ms_x = Tensor(np_x)
    with pytest.raises(IndexError):
        if mode == ms.PYNATIVE_MODE:
            ms_x[[0, 2]] = -1
        else:
            _ = func_list_index_dim_out_data_dim(ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_tuple_index_dim_out_data_dim(x):
        x[([0, 2])] = -1
        return x
    ms_x = Tensor(np_x)
    with pytest.raises(IndexError):
        if mode == ms.PYNATIVE_MODE:
            ms_x[([0, 2])] = -1
        else:
            _ = func_tuple_index_dim_out_data_dim(ms_x)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_previous_setitem_exception_type_error(mode, capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of previous tensor setitem exception
    Expectation: success
    """

    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_not_support_type_as_value(x):
        x[0] = slice(0, 1)
        return x
    with pytest.raises(TypeError):
        if mode == ms.PYNATIVE_MODE:
            ms_x[0] = slice(0, 1)
        else:
            _ = func_not_support_type_as_value(ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_scalar_tensor(x):
        x[[0]] = -1
        return x
    ms_x = Tensor(1)
    with pytest.raises(TypeError):
        if mode == ms.PYNATIVE_MODE:
            ms_x[[0]] = -1
        else:
            _ = func_scalar_tensor(ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_scalar_tensor_with_slice_index(x):
        x[slice(0, 1, 1)] = -1
        return x
    ms_x = Tensor(1)
    with pytest.raises(TypeError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[slice(0, 1, 1)] = -1
        else:
            _ = func_scalar_tensor_with_slice_index(ms_x)
    assert "Cannot iterate over a scalar tensor" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_scalar_tensor_with_Tensor_index(x):
        x[Tensor(1)] = -1
        return x
    ms_x = Tensor(1)
    with pytest.raises(TypeError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[Tensor(1)] = -1
        else:
            _ = func_scalar_tensor_with_Tensor_index(ms_x)
    assert "Cannot iterate over a scalar tensor" in str(exc.value)


def setitem_check_indexing(x, index, value, np_expected, capture_mode=None):
    """setitem run and check"""
    if capture_mode is None:
        def func(ms_x, index, value):
            ms_x[index] = value
            return ms_x
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def func(ms_x, index, value):
            ms_x[index] = value
            return ms_x

    ms_output = func(x, index, value)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(func, x, index, value)

    assert np.allclose(np_expected, ms_output.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                          f"expected:{np_expected} {np_expected.shape}, " \
                                                          f"ms_output:{ms_output} {ms_output.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast'])
def test_setitem(capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem
    Expectation: success
    """
    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    # Basic index
    basic_indices = [0, slice(0, 1), True, False,
                     None, ..., (0, 2, ...), [0, 1]]
    for index in basic_indices:
        np_x = np.arange(2*3*4).reshape(2, 3, 4)
        ms_x = Tensor(np_x)
        value = -1
        np_x[index] = value
        setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Basic index with bool value
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    index = slice(0, 1)
    value = False
    np_x[index] = value
    setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Basic index with bool value
    np_x = np.array([False, True])
    ms_x = Tensor(np_x)
    index = 0
    value = True
    np_x[index] = value
    setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Basic index with float value
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    index = [0, 1]
    value = -1.0
    np_x[index] = value
    setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Basic index with tensor value
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    index = 0
    value = Tensor([[[-1, -1, -1, -1]]])
    np_expected = np.array([[[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    setitem_check_indexing(ms_x, index, value, np_expected, capture_mode)

    # Numpy index
    if capture_mode is None:
        numpy_indices = [np.array(0), np.array(True), np.array(
            False), (np.array(0), np.array(1)), np.array([0, 1])]
        for index in numpy_indices:
            np_x = np.arange(2*3*4).reshape(2, 3, 4)
            ms_x = Tensor(np_x)
            value = -1
            np_x[index] = value
            setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Tensor index
    tensor_indices = [
        Tensor(0),
        Tensor(True),
        Tensor(False),
        slice(Tensor(0), Tensor(2)),
        Tensor([0, 1]),
        Tensor([True, False])
    ]
    np_indices = [0, True, False, slice(0, 2), [0, 1], [True, False]]
    for index, np_index in zip(tensor_indices, np_indices):
        np_x = np.arange(2*3*4).reshape(2, 3, 4)
        ms_x = Tensor(np_x)
        value = -1
        np_x[np_index] = value
        setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Tuple index
    tuple_index = (0, None, ...)
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    value = -1
    np_expected = np.array([[[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    setitem_check_indexing(ms_x, tuple_index, value, np_expected, capture_mode)

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
                     (Tensor([0, 1]), ..., Tensor([0, 1])),
                     (Tensor([0]), Tensor(0), slice(0, 4, 2))]
    np_expecteds = [np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]]),
                    np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, -1, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, -1, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, -1, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, -1, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [-1, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, -1, 14, 15], [16, -1, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [-1, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, -1, 14, 15], [16, -1, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]]),
                    np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]]),
                    np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]]),
                    np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [-1, 5, 6, 7], [-1, 9, 10, 11]],
                              [[12, -1, 14, 15], [16, -1, 18, 19], [20, -1, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [-1, 5, 6, 7], [-1, 9, 10, 11]],
                              [[12, -1, 14, 15], [16, -1, 18, 19], [20, -1, 22, 23]]]),
                    np.array([[[-1, 1, -1, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])]
    for index, np_expected in zip(fancy_indices, np_expecteds):
        ms_x = Tensor(np.arange(2*3*4).reshape(2, 3, 4))
        value = -1
        setitem_check_indexing(ms_x, index, value, np_expected, capture_mode)


def setitem_check_indexing_without_jit(x, index, value, np_expected, capture_mode=None):
    """setitem run and check"""
    def func(ms_x, index, value):
        ms_x[index] = value
        return ms_x
    ms_output = func(x, index, value)
    assert np.allclose(np_expected, ms_output.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                          f"expected:{np_expected} {np_expected.shape}, " \
                                                          f"ms_output:{ms_output} {ms_output.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None])
def test_slice_tensor_index_setitem_without_jit(capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem
    Expectation: success
    """
    # Slice Tensor index with value
    basic_type = [ms.int8, ms.uint8, ms.int16, ms.uint16, ms.int, ms.int32, ms.uint32, ms.int64, ms.uint64, ms.float16,
                  ms.float, ms.float32, ms.double, ms.float64, ms.bfloat16]
    np_expected = np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                            [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]])
    for dtype in basic_type:
        np_x = np.arange(2*3*4).reshape(2, 3, 4)
        ms_x = Tensor(np_x)
        start = Tensor(1, dtype=dtype)
        end = Tensor(2, dtype=dtype)
        step = Tensor(1, dtype=dtype)
        slice_index = slice(start, end, step)
        value = -1
        setitem_check_indexing_without_jit(
            ms_x, slice_index, value, np_expected, capture_mode)


def setitem_check_iadd_indexing(x, index, value, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:
        def func(ms_x, index, value):
            ms_x[index] += value
            return ms_x
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def func(ms_x, index, value):
            ms_x[index] += value
            return ms_x

    ms_output = func(x, index, value)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(func, x, index, value)

    assert np.allclose(np_expected, ms_output.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                          f"expected:{np_expected} {np_expected.shape}, " \
                                                          f"ms_output:{ms_output} {ms_output.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast'])
def test_setitem_with_iadd(capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem with iadd
    Expectation: success
    """

    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    # Basic index
    basic_indices = [0, slice(0, 1), True, False, None, ..., (0, slice(
        0, 2), True), (slice(0, 2), None, ...), [0, 1]]
    for index in basic_indices:
        np_x = np.arange(2*3*4).reshape(2, 3, 4)
        ms_x = Tensor(np_x)
        value = -1
        np_x[index] += value
        setitem_check_iadd_indexing(ms_x, index, value, np_x, capture_mode)

    # Tuple index
    tuple_index = (0, None, ...)
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    value = -1
    np_expected = np.array([[[-1, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]],
                            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    setitem_check_iadd_indexing(
        ms_x, tuple_index, value, np_expected, capture_mode)

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
    np_expecteds = [np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]]),
                    np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 12, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 12, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, 4, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, 4, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [3, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 12, 14, 15], [16, 16, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [3, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 12, 14, 15], [16, 16, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]]),
                    np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]]),
                    np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]]),
                    np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [3, 5, 6, 7], [7, 9, 10, 11]],
                              [[12, 12, 14, 15], [16, 16, 18, 19], [20, 20, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [3, 5, 6, 7], [7, 9, 10, 11]],
                              [[12, 12, 14, 15], [16, 16, 18, 19], [20, 20, 22, 23]]])]
    for index, np_expected in zip(fancy_indices, np_expecteds):
        ms_x = Tensor(np.arange(2*3*4).reshape(2, 3, 4))
        value = -1
        setitem_check_iadd_indexing(
            ms_x, index, value, np_expected, capture_mode)


class NetSetitem(nn.Cell):
    """test setitem use net"""

    def __init__(self, index, value):
        super().__init__()
        self.index = index
        self.value = value

    def construct(self, x):
        x = ops.abs(x)
        x[self.index] = self.value
        return x


def setitem_check_grad(x, index, value, np_expected, capture_mode=None):
    """setitem run and check"""
    if capture_mode is None:
        def grad_func(net, x):
            return ms.grad(net)(x)
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def grad_func(net, x):
            return ms.grad(net)(x)

    net = NetSetitem(index, value)
    ms_grad = grad_func(net, x)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(net.construct, x, index, value)

    assert np.allclose(np_expected, ms_grad.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                        f"expected:{np_expected} {np_expected.shape}, " \
                                                        f"ms_grad:{ms_grad} {ms_grad.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast', 'bytecode'])
def test_setitem_grad(capture_mode):
    """
    Feature: tensor setitem grad
    Description: Verify the result of tensor setitem grad1
    Expectation: success
    """

    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    # Base index
    base_indices = [0, slice(0, 2), True, False, ..., None, [0, 1]]
    np_expecteds = [np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]])]
    for index, np_expected in zip(base_indices, np_expecteds):
        ms_x = Tensor(
            np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
        value = -1
        setitem_check_grad(ms_x, index, value, np_expected, capture_mode)

    # Basic index with bool value
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
    index = slice(0, 1)
    value = False
    np_expected = np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                            [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]])
    setitem_check_grad(ms_x, index, value, np_expected, capture_mode)

    # Basic index with float value
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
    index = [0, 1]
    value = -1.0
    np_expected = np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                            [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]])
    setitem_check_grad(ms_x, index, value, np_expected, capture_mode)

    # Basic index with tensor value
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
    index = 0
    value = Tensor([[[-1, -1, -1, -1]]])
    np_expected = np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                            [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]])
    setitem_check_grad(ms_x, index, value, np_expected, capture_mode)

    # Tensor index
    tensor_indices = [Tensor(0), Tensor(True), Tensor(
        False), slice(Tensor(0), Tensor(2)), Tensor([0, 1])]
    value = -1
    np_expecteds = [np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]])]
    for index, np_expected in zip(tensor_indices, np_expecteds):
        setitem_check_grad(ms_x, index, value, np_expected, capture_mode)

    # Tuple index
    tuple_indices = [(0, slice(0, 2), True), (0, None, ...)]
    value = -1
    np_expecteds = [np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]])]
    for index, np_expected in zip(tuple_indices, np_expecteds):
        setitem_check_grad(ms_x, index, value, np_expected, capture_mode)

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
                     (Tensor([0, 1]), ..., Tensor([0, 1])),
                     (Tensor([0]), Tensor(0), slice(0, 4, 2))]
    value = -1
    np_expecteds = [
        np.array([[[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]]]),
        np.array([[[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 0., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 0., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [1., 0., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [1., 0., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [0., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 0., 1., 1.], [1., 0., 1., 1.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [0., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 0., 1., 1.], [1., 0., 1., 1.], [1., 1., 1., 1.]]]),
        np.array([[[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]]]),
        np.array([[[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]]]),
        np.array([[[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]]]),
        np.array([[[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [0., 1., 1., 1.], [0., 1., 1., 1.]],
                  [[1., 0., 1., 1.], [1., 0., 1., 1.], [1., 0., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [0., 1., 1., 1.], [0., 1., 1., 1.]],
                  [[1., 0., 1., 1.], [1., 0., 1., 1.], [1., 0., 1., 1.]]]),
        np.array([[[0., 1., 0., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]])]
    for index, np_expected in zip(fancy_indices, np_expecteds):
        setitem_check_grad(ms_x, index, value, np_expected, capture_mode)


class NetSetitemIadd(nn.Cell):
    """test setitem use self-add net"""

    def __init__(self, index, value):
        super().__init__()
        self.index = index
        self.value = value

    def construct(self, x):
        x = ops.abs(x)
        x[self.index] += self.value
        return x


def setitem_check_iadd_grad(x, index, value, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:
        def grad_func(net, x):
            return ms.grad(net)(x)
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def grad_func(net, x):
            return ms.grad(net)(x)

    net = NetSetitemIadd(index, value)
    ms_grad = grad_func(net, x)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(net.construct, x, index, value)

    assert np.allclose(np_expected, ms_grad.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                        f"expected:{np_expected} {np_expected.shape}, " \
                                                        f"ms_grad:{ms_grad} {ms_grad.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast',
                                          pytest.param('bytecode', marks=pytest.mark.skip(reason="Unsupported now"))])
@pytest.mark.parametrize(
    'index',
    [
        # 0-6: base index
        0,
        True,
        False,
        None,
        slice(0, 2),
        ...,
        [0, 1],
        # 7-11: tensor index
        Tensor(0),
        Tensor(True),
        Tensor(False),
        slice(Tensor(0), Tensor(2)),
        Tensor([0, 1]),
        # 12-13: tuple index
        (0, None, ...),
        (0, slice(0, 2), True),
        # 14-27: fancy index
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
        ([0, 1], [0, 1]),
        (Tensor([0, 1]), Tensor([0, 1])),
        ([0, 1], ..., [0, 1]),
        (Tensor([0, 1]), ..., Tensor([0, 1])),
    ],
)
def test_setitem_grad_with_iadd(capture_mode, index):
    """
    Feature: tensor setitem grad
    Description: Verify the result of tensor setitem grad with iadd
    Expectation: success
    """

    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    np_expected = np.array([[[0., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                            [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
    value = -1
    setitem_check_iadd_grad(ms_x, index, value, np_expected, capture_mode)


class NetSetitemImul(nn.Cell):
    """test setitem use self-mul net"""

    def __init__(self, index, value):
        super().__init__()
        self.index = index
        self.value = value

    def construct(self, x):
        x = ops.abs(x)
        x[self.index] *= self.value
        return x


def setitem_check_imul_grad(x, index, value, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:
        def grad_func(net, x):
            return ms.grad(net)(x)
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def grad_func(net, x):
            return ms.grad(net)(x)

    net = NetSetitemImul(index, value)
    ms_grad = grad_func(net, x)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(net.construct, x, index, value)

    assert np.allclose(np_expected, ms_grad.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                        f"expected:{np_expected} {np_expected.shape}, " \
                                                        f"ms_grad:{ms_grad} {ms_grad.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast',
                                          pytest.param('bytecode', marks=pytest.mark.skip(reason="Unsupported now"))])
def test_setitem_grad_with_imul(capture_mode):
    """
    Feature: tensor setitem grad
    Description: Verify the result of tensor setitem grad with imul
    Expectation: success
    """

    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
    iadd_indices = [True, None]
    np_expected = np.array([[[0., 3., 3., 3.], [3., 3., 3., 3.], [3., 3., 3., 3.]],
                            [[3., 3., 3., 3.], [3., 3., 3., 3.], [3., 3., 3., 3.]]])
    for index in iadd_indices:
        ms_x = Tensor(
            np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
        value = 3
        setitem_check_imul_grad(ms_x, index, value, np_expected, capture_mode)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'ast'), (ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_setitem_exception(mode, capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem exception
    Expectation: success
    """
    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func1(x):
        x[2, 0, 0] = -1
        return x
    with pytest.raises(IndexError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[2, 0, 0] = -1
        else:
            _ = func1(ms_x)
    assert "is out of bounds" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func2(x):
        x[0, 0, 0, 0] = -1
        return x
    with pytest.raises(IndexError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0, 0, 0, 0] = -1
        else:
            _ = func2(ms_x)
    assert "too many indices" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func3(x):
        x[0, 't'] = -1
        return x
    with pytest.raises(IndexError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0, 't'] = -1
        else:
            _ = func3(ms_x)
    assert "Invalid tensor index type" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func4(x):
        x[0:3:-1] = -1
        return x
    with pytest.raises(ValueError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0:3:-1] = -1
        else:
            _ = func4(ms_x)
    assert "slice step must be positive" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func5(x):
        x[0] = (1, 2, 3)
        return x
    with pytest.raises(TypeError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0] = (1, 2, 3)
        else:
            _ = func5(ms_x)
    assert "the type of value can only be bool, int, float or Tensor." in str(
        exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func6(x):
        x[0] = [1, 2, 3]
        return x
    with pytest.raises(TypeError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0] = [1, 2, 3]
        else:
            _ = func6(ms_x)
    assert "the type of value can only be bool, int, float or Tensor." in str(
        exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func7(x):
        x[0] = -1
        return x
    ms_x = Tensor(0)
    with pytest.raises(TypeError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0] = -1
        else:
            _ = func7(ms_x)
    assert "Invalid index of a 0-dim tensor." in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func8(x):
        x[0:1] = -1
        return x
    with pytest.raises(TypeError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0:1] = -1
        else:
            _ = func8(ms_x)
    assert "Invalid index of a 0-dim tensor." in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func9(x):
        x[Tensor(1.4), :, :] = -1
        return x
    ms_x = Tensor(np_x)
    with pytest.raises(TypeError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[Tensor(1.4), :, :] = -1
        else:
            _ = func9(ms_x)
    assert "For 'InplaceIndexPut', tensors used as indices must be long, int, uint8, or bool tensors" in str(
        exc.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_setitem_exception_without_jit_ast(mode, capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem exception
    Expectation: success
    """
    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_tensor_as_slice_index_dim_out_1(x):
        s = Tensor([0, 1])
        index = slice(s, s, s)
        x[index] = -1
        return x
    ms_x = Tensor(np_x)
    with pytest.raises(ValueError):
        if mode == ms.PYNATIVE_MODE:
            ms_x[Tensor([0, 1]):1:1] = -1
        else:
            _ = func_tensor_as_slice_index_dim_out_1(ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_tensor_as_slice_index_with_unsupport_type(x):
        s = Tensor(3 + 4j, dtype=ms.complex64)
        index = slice(s, s, s)
        x[index] = -1
        return x
    ms_x = Tensor(np_x)
    with pytest.raises(TypeError):
        if mode == ms.PYNATIVE_MODE:
            ms_x[Tensor(3 + 4j, dtype=ms.complex64):1:1] = -1
        else:
            _ = func_tensor_as_slice_index_with_unsupport_type(ms_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func_slice_with_float_index(x):
        x[slice(1.1, 2)] = -1
        return x
    ms_x = Tensor(np_x)
    with pytest.raises(IndexError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[slice(1.1, 2)] = -1
        else:
            _ = func_slice_with_float_index(ms_x)
    assert "slice indices must be integers or None or Tensor" in str(exc.value)


class IndexDynamicShapeNet(nn.Cell):
    def construct(self, x, index, value):
        x = ops.abs(x)
        x[0:2, index] = value
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', ['ast', 'bytecode'])
@pytest.mark.parametrize('x_shape', [(3, 3, None), (3, 3, 3)])
@pytest.mark.parametrize('index_shape', [(2, None), (2, 2)])
@pytest.mark.parametrize('value_shape', [(None,), (1,)])
def test_setitem_index_dynamic_shape_test(capture_mode, x_shape, index_shape, value_shape):
    """
    Feature: tensor setitem with index dynamic shape
    Description: Verify the result of tensor setitem with index dynamic shape
    Expectation: success
    """
    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func(net, x, index, value):
        return net(x, index, value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def grad_func(net, x, index, value):
        return ms.grad(net)(x, index, value)

    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    pt_result = np.array([[[-1., -1., -1.], [-1., -1., -1.], [6., 7., 8.]],
                          [[-1., -1., -1.], [-1., -1., -1.], [15., 16., 17.]],
                          [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]]])
    pt_grad = np.array([[[0., 0., 0.], [0., 0., 0.], [1., 1., 1.]],
                        [[0., 0., 0.], [0., 0., 0.], [1., 1., 1.]],
                        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]])

    net = IndexDynamicShapeNet()
    ms_x = Tensor(np.arange(3 * 3 * 3).reshape((3, 3, 3)).astype(np.float32))
    index = Tensor([[0, 1], [0, 1]])
    value = Tensor(-1, ms.float32)
    x_dyn = Tensor(
        shape=x_shape, dtype=ms.float32) if None in x_shape else ms_x
    index_dyn = Tensor(shape=index_shape,
                       dtype=ms.int64) if None in index_shape else index
    value_dyn = Tensor(shape=value_shape,
                       dtype=ms.float32) if None in value_shape else value
    net.set_inputs(x_dyn, index_dyn, value_dyn)

    ms_result = func(net, ms_x, index, value)
    assert np.allclose(pt_result, ms_result.asnumpy()), f"pt_result: {pt_result}, " \
                                                        f"ms_result: {ms_result.asnumpy()}"

    ms_grad = grad_func(net, ms_x, index, value)
    assert np.allclose(pt_grad, ms_grad.asnumpy()), f"pt_grad: {pt_grad}, " \
                                                    f"ms_grad: {ms_grad.asnumpy()}"


class IndexDynamicRankNet(nn.Cell):
    def construct(self, x, index1, index2, cond, value):
        x = ops.abs(x)
        if cond:
            index = index1
        else:
            index = index2
        x[0:2, index] = value
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', ['ast', 'bytecode'])
def test_setitem_index_dynamic_rank_test(capture_mode):
    """
    Feature: tensor setitem with index dynamic rank
    Description: Verify the result of tensor setitem with index dynamic rank
    Expectation: success
    """
    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func(net, x, index1, index2, cond, value):
        return net(x, index1, index2, cond, value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def grad_func(net, x, index1, index2, cond, value):
        return ms.grad(net)(x, index1, index2, cond, value)

    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    pt_result = np.array([[[0., 1., 2.], [-1., -1., -1.], [6., 7., 8.]],
                          [[9., 10., 11.], [-1., -1., -1.], [15., 16., 17.]],
                          [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]]])
    pt_grad = np.array([[[0., 1., 1.], [0., 0., 0.], [1., 1., 1.]],
                        [[1., 1., 1.], [0., 0., 0.], [1., 1., 1.]],
                        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]])

    net = IndexDynamicRankNet()
    ms_x = Tensor(np.arange(3 * 3 * 3).reshape((3, 3, 3)).astype(np.float32))
    index1 = Tensor([1])
    index2 = Tensor([[0, 1], [0, 1]])
    cond = Tensor(True)
    value = Tensor(-1, ms.float32)
    ms_result = func(net, ms_x, index1, index2, cond, value)
    assert np.allclose(pt_result, ms_result.asnumpy()), f"pt_result: {pt_result}, " \
                                                        f"ms_result: {ms_result.asnumpy()}"

    ms_grad = grad_func(net, ms_x, index1, index2, cond, value)
    assert np.allclose(pt_grad, ms_grad.asnumpy()), f"pt_grad: {pt_grad}, " \
                                                    f"ms_grad: {ms_grad.asnumpy()}"


class IndexDynamicRank2Net(nn.Cell):
    def construct(self, x, index1, index2, value):
        x = ops.abs(x)
        mask = index1 == index2
        index = index1[mask]
        value = value[mask]
        x[0:1, index] = value
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', ['ast'])
def test_setitem_index_dynamic_rank_test2(capture_mode):
    """
    Feature: tensor setitem with index dynamic rank
    Description: Verify the result of tensor setitem with index dynamic rank
    Expectation: success
    """
    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func(net, x, index1, index2, value):
        return net(x, index1, index2, value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def grad_func(net, x, index1, index2, value):
        return ms.grad(net)(x, index1, index2, value)

    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    pt_result = np.array([[[0., 1., 2.], [3., 4., 5.], [-3., -3., -3.]],
                          [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
                          [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]]])
    pt_grad = np.array([[[0., 1., 1.], [1., 1., 1.], [0., 0., 0.]],
                        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]])

    net = IndexDynamicRank2Net()
    ms_x = Tensor(np.arange(3 * 3 * 3).reshape((3, 3, 3)).astype(np.float32))
    index1 = Tensor([0, 1, 2])
    index2 = Tensor([1, 2, 2])
    value = Tensor([-1, -2, -3], ms.float32)
    ms_result = func(net, ms_x, index1, index2, value)
    assert np.allclose(pt_result, ms_result.asnumpy()), f"pt_result: {pt_result}, " \
                                                        f"ms_result: {ms_result.asnumpy()}"

    ms_grad = grad_func(net, ms_x, index1, index2, value)
    assert np.allclose(pt_grad, ms_grad.asnumpy()), f"pt_grad: {pt_grad}, " \
                                                    f"ms_grad: {ms_grad.asnumpy()}"


class NetWithIndexAndMul(nn.Cell):
    def construct(self, x):
        a = ms.mint.zeros_like(x)
        a[:, 0:1] = 1
        b = x * a
        return b


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_setitem_with_mul(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of network with mul after setitem.
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    ms_x = Tensor([[0, 0], [2, 0]])
    net = NetWithIndexAndMul()
    ms_y = net(ms_x)
    np_expect = np.array([[0, 0], [2, 0]])
    assert np.allclose(np_expect, ms_y.asnumpy()
                       ), f"np_expect:{np_expect}, ms_y:{ms_y}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_setitem_graph_mode(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem in graph mode
    Expectation: success
    """
    ms.set_context(mode=mode)
    np_x = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    ms_x = Tensor(np_x)
    ms_x[0] = -1
    np_expect = np.array([[-1, -1, -1], [3, 4, 5]]).astype(np.float32)
    assert np.allclose(np_expect, ms_x.asnumpy()
                       ), f"np_expect:{np_expect}, ms_x:{ms_x}"


class NetIndexBool(nn.Cell):
    def construct(self, x):
        x[True] *= 3
        return x


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_setitem_index_bool(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem with indexes are bool
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = NetIndexBool()
    x = Tensor(np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32))
    x = net(Tensor(x))
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    x_np[True] *= 3
    assert np.allclose(x_np, x.asnumpy())


class NetIndexNone(nn.Cell):
    def construct(self, x):
        x[None] *= 3
        return x


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_setitem_index_none(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem with indexes are None
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = NetIndexBool()
    x = Tensor(np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32))
    x = net(Tensor(x))
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    x_np[None] *= 3
    assert np.allclose(x_np, x.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_setitem_repeatly():
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem repeatedly
    Expectation: success
    """
    x = Tensor([1, 2, 3])
    x[0] = -1
    assert np.allclose(np.array([-1, 2, 3]), x.asnumpy())
    x[0] = -2
    assert np.allclose(np.array([-2, 2, 3]), x.asnumpy())
    x = Tensor([False, True])
    x[0] = True
    assert np.allclose(np.array([True, True]), x.asnumpy())
    x[1] = False
    assert np.allclose(np.array([True, False]), x.asnumpy())


def no_jit(x, *args, **kwargs):
    return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_setitem_nested_list_as_value_false(mode):
    """
    Feature: tensor setitem.
    Description: Assigning a nested list as value in tensor setitem should be rejected.
    Expectation: TypeError in no-JIT; ValueError in JIT.
    """
    def get_ms_net():
        input_x = Parameter(
            Tensor(np.arange(120).reshape(2, 3, 4, 5).astype(np.float32)))
        b = Tensor(np.array([1]))

        def net():
            input_x[1, b, [False, True, False, False], (2, ), True] = [[2.0]]
            out = ops.relu(input_x)
            return out
        return net

    net = get_ms_net()
    if mode is ms.jit:
        with pytest.raises(ValueError):
            mode(net)()
    else:
        with pytest.raises(TypeError, match="the type of value can only be bool, int, float or Tensor"):
            mode(net)()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [no_jit, ms.jit])
def test_setitem_nested_tuple_as_value_false(mode):
    """
    Feature: tensor setitem.
    Description: Assigning a nested tuple containing mixed scalars as value is unsupported.
    Expectation: TypeError in no-JIT; ValueError in JIT.
    """
    def get_ms_net():
        b = Tensor(np.array([1]))

        def net(input_x):
            input_x[1, b, [False, True, False, False],
                    (2, ), True] = ((2.0, ), 3.0)
            out = ops.ReLU()(input_x)
            return out
        return net

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = get_ms_net()
    if mode is ms.jit:
        with pytest.raises(ValueError):
            mode(net)(input_me)
    else:
        with pytest.raises(TypeError, match="the type of value can only be bool, int, float or Tensor"):
            mode(net)(input_me)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_setitem_nested_list_as_value_ok():
    """
    Feature: tensor setitem.
    Description: Assigning a correctly shaped list as value works and matches PyTorch.
    Expectation: Forward and gradient outputs match PyTorch result.
    """
    def get_ms_net():
        input_x = Parameter(
            Tensor(np.arange(120).reshape(2, 3, 4, 5).astype(np.float32)))
        b = Tensor(np.array([1, 0]))

        def net(input_y):
            input_x[None, True, b, (False, True, True), [2], 4] = [[2.0, 3.0]]
            input_z = input_x + input_y
            out = ops.relu(input_z)
            return out

        return net

    class NetPytorch1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_x = torch.from_numpy(
                np.arange(120).reshape(2, 3, 4, 5).astype(np.float32))
            self.b = torch.from_numpy(np.array([1, 0]))
            self.relu = torch.nn.ReLU()

        def forward(self, input_y):
            self.input_x[None, True, self.b, (False, True, True), [2],
                         4] = torch.tensor([[2.0, 3.0]])
            input_z = self.input_x + input_y
            out = self.relu(input_z)
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    net_ms = get_ms_net()
    net_pt = NetPytorch1()
    ms_input = ms.Tensor(input_np)
    pt_input = torch.tensor(input_np, requires_grad=True)
    ms_out = ms.jit(net_ms, backend="ms_backend")(ms_input)
    pt_out = net_pt(pt_input)
    assert np.allclose(ms_out.asnumpy(), pt_out.detach().numpy())
    np_sens = np.random.randn(*ms_out.shape).astype(np.float32)
    ms_sens = ms.Tensor(np_sens)
    pt_sens = torch.tensor(np_sens)
    ms_grad = ms.jit(ms.grad(net_ms, sens_param=True), backend="ms_backend")(ms_input, ms_sens)
    pt_grad = torch.autograd.grad(pt_out, pt_input, grad_outputs=pt_sens)
    assert np.allclose(ms_grad.asnumpy(), pt_grad[0].detach().numpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_setitem_nested_tuple_as_value_in_jit_ok():
    """
    Feature: tensor setitem.
    Description: Tuple value with matching shape is accepted in JIT mode.
    Expectation: Output matches PyTorch; gradients correct.
    """
    b = Tensor(np.array([1, 0]))

    def get_ms_net():
        def net(input_x):
            input_x = input_x * 1
            input_x[None, True, b, (False, True, True), [
                2], 4] = ((2.0, 3.0), )
            out = ops.relu(input_x)
            return out
        return net

    class NetPytorch1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, input_x):
            input_x.requires_grad = False
            b = torch.from_numpy(np.array([1, 0]))
            input_x[None, True, b, (False, True, True), [
                2], 4] = torch.tensor([[2.0, 3.0]])
            out = self.relu(input_x)
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    net_ms = get_ms_net()
    net_pt = NetPytorch1()
    ms_input = ms.Tensor(input_np)
    pt_input = torch.tensor(input_np, requires_grad=True)
    ms_out = ms.jit(net_ms)(ms_input)
    pt_out = net_pt(pt_input)
    assert np.allclose(ms_out.asnumpy(), pt_out.detach().numpy())
    np_sens = np.random.randn(*ms_out.shape).astype(np.float32)
    ms_sens = ms.Tensor(np_sens)
    grad_ms = ms.jit(ms.grad(net_ms, sens_param=True))(ms_input, ms_sens)
    assert not np.any(
        grad_ms[None, True, b, (False, True, True), [2], 4].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_setitem_nested_tuple_as_value_in_jit_false():
    """
    Feature: tensor setitem.
    Description: Tuple length does not match target slice shape.
    Expectation: ValueError raised in JIT.
    """
    def get_ms_net():
        b = Tensor(np.array([1, 0]))

        def net(input_x):
            input_x[None, True, b, (False, True, True), [
                2], 4] = ((2.0, 3.0, 4.0), )
            out = ops.relu(input_x)
            return out
        return net

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = get_ms_net()
    with pytest.raises(ValueError):
        ms.jit(net)(input_me)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize(
    "value1,value2",
    [(Tensor([[2, 3]], mstype.int8), torch.tensor([[2, 3]], dtype=torch.float32)),
     (Tensor([[2, 3]], mstype.int16), torch.tensor(
         [[2, 3]], dtype=torch.float32)),
     (Tensor([[2, 3]], mstype.int64), torch.tensor(
         [[2, 3]], dtype=torch.float32)),
     (Tensor([[2, 3]], mstype.float32), torch.tensor([[2, 3]], dtype=torch.float32))])
def test_setitem_of_different_value_types_001(value1, value2):
    """
    Feature: tensor setitem.
    Description: Test assignment using different value tensor types.
    Expectation: Output matches PyTorch after ReLU.
    """
    def get_ms_net():
        b = Tensor(np.array([1, 0]))

        def net(input_x):
            input_x[None, True, b, (False, True, True), [2], 4] = value1
            out = ops.relu(input_x)
            return out
        return net

    class NetPytorch1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, input_x):
            b = torch.from_numpy(np.array([1, 0]))
            input_x[None, True, b, (False, True, True), [2], 4] = value2
            out = self.relu(input_x)
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float16)
    net_me = get_ms_net()
    net_torch = NetPytorch1()
    input_me = Tensor(input_np)
    input_torch = torch.from_numpy(input_np.astype(np.float32))
    out_me = ms.jit(net_me)(input_me)
    out_torch = net_torch(input_torch)
    np.allclose(out_torch.numpy(), out_me.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize(
    "value1,value2",
    [(Tensor([[2, 3]], mstype.float16), torch.tensor([[2, 3]], dtype=torch.float32)),
     (Tensor([[2, 3]], mstype.float64),
      torch.tensor([[2, 3]], dtype=torch.float32)),
     (Tensor([[2, 3]], mstype.int64), torch.tensor([[2, 3]], dtype=torch.float32))])
def test_setitem_of_different_value_types_002(value1, value2):
    """
    Feature: tensor setitem.
    Description: Validate setitem behavior with non-default value dtypes.
    Expectation: Output matches PyTorch; gradients correct.
    """
    b = Tensor(np.array([1, 0]))

    def get_ms_net():
        def net(input_x):
            input_x[None, True, b, (False, True, True), [2], 4] = value1
            out = ops.relu(input_x)
            return out
        return net

    class NetPytorch1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, input_x):
            input_x.requires_grad = False
            b = torch.from_numpy(np.array([1, 0]))
            input_x[None, True, b, (False, True, True), [2], 4] = value2
            out = self.relu(input_x)
            return out

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    net_ms = get_ms_net()
    net_pt = NetPytorch1()
    ms_input = ms.Tensor(input_np)
    pt_input = torch.tensor(input_np, requires_grad=True)
    ms_out = ms.jit(net_ms)(ms_input)
    pt_out = net_pt(pt_input)
    assert np.allclose(ms_out.asnumpy(), pt_out.detach().numpy())
    np_sens = np.random.randn(*ms_out.shape).astype(np.float32)
    ms_sens = ms.Tensor(np_sens)
    grad_ms = ms.jit(ms.grad(net_ms, sens_param=True))(ms_input, ms_sens)
    assert not np.any(
        grad_ms[None, True, b, (False, True, True), [2], 4].asnumpy())
