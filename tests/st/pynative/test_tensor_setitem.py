# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test_tensor_setitem """
import numpy as np
import pytest

from mindspore import Tensor, context, ops
from mindspore.nn import Cell
from mindspore import dtype as mstype
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark


def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE)


def setup_testcase(input_np, case_fn):
    input_ms = Tensor(input_np)

    class TensorSetItem(Cell):
        def construct(self, x):
            return case_fn(x)

    class NumpySetItem():
        def __call__(self, x):
            return case_fn(x)

    out_ms = TensorSetItem()(input_ms)
    out_np = NumpySetItem()(input_np)
    assert np.all(out_ms.asnumpy() == out_np)


class TensorSetItemByList(Cell):
    def construct(self, x):
        x[[0, 1], [1, 2], [1, 3]] = 3
        x[([0, 1], [0, 2], [1, 1])] = 10
        x[[0, 1], ..., [0, 1]] = 4
        return x


class NumpySetItemByList():
    def __call__(self, x):
        x[[0, 1], [1, 2], [1, 3]] = 3
        x[([0, 1], [0, 2], [1, 1])] = 10
        x[[0, 1], ..., [0, 1]] = 4
        return x


class SetitemNet(Cell):
    def construct(self, x, y):
        z = x * 2
        z[:, 1:] += y[:, 1:]
        return z


class SetitemGradNet(Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True)

    def construct(self, x, y):
        gradient_func = self.grad_op(self.net)
        return gradient_func(x, y)


def test_setitem_grad():
    """
    Feature: Test setitem grad
    Description: setitem should return correct grad
    Expectation: success
    """
    net = SetitemNet()
    a = Tensor(np.random.randn(2, 2, 2, 2), mstype.float32)
    b = Tensor(np.random.randn(2, 2, 2, 2), mstype.float32)
    b = ops.zeros_like(b)
    output = SetitemGradNet(net)(a, b)
    x_grad = np.ones((2, 2, 2, 2), np.float32) * 2
    y_grad = np.array([[[[0, 0], [0, 0]], [[1, 1], [1, 1]]], [[[0, 0], [0, 0]], [[1, 1], [1, 1]]]], np.float32)
    assert np.array_equal(output[0].asnumpy(), x_grad)
    assert np.array_equal(output[1].asnumpy(), y_grad)


def test_setitem_by_list():
    """
    Feature: Test setitem by list
    Description: setitem with index of type list
    Expectation: success
    """
    x = np.ones((2, 3, 4), dtype=np.float32)

    def cases(x):
        x[[0, 1], [1, 2], [1, 3]] = 3
        x[([0, 1], [0, 2], [1, 1])] = 10
        x[[0, 1], ..., [0, 1]] = 4
        return x
    setup_testcase(x, cases)


def test_setitem_dtype():
    """
    Feature: Test setitem with different dtype
    Description: setitem with different dtype
    Expectation: Success
    """
    x = np.ones((2, 3, 4), dtype=np.float32)

    def cases(x):
        x[...] = 3
        x[..., 1] = 3.0
        x[0] = True
        x[1:2] = -1
        return x
    setup_testcase(x, cases)


def test_setitem_by_tuple_with_int():
    x = np.arange(24).reshape(2, 3, 4).astype(np.float32)

    def cases(x):
        x[..., 2, False, 1] = -1
        x[0, True, 0, None, True] = -2
        x[0, ..., None] = -3
        x[..., 0, None, 1, True, True, None] = -4
        x[Tensor(-1), 0:1] = -5
        return x
    setup_testcase(x, cases)


def test_setitem_by_tuple_with_list():
    """
    Feature: Test setitem by tuple with list
    Description: setitem with index of type tuple with list
    Expectation: Success
    """
    x = np.arange(24).reshape(2, 3, 4).astype(np.float32)

    def cases(x):
        x[..., 2, False, 1] = -1
        x[0, True, 0, None, True] = -2
        x[0, ..., None] = -3
        x[..., 0, None, 1, True, True, None] = -4
        x[None, True, [1, 0], (False, True, True), [2]] = -5
        return x
    setup_testcase(x, cases)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_setitem_with_bool_tensor_index():
    """
    Feature: Test setitem with mixed integer and bool tensor indices.
    Description: Verify bool tensor indices are expanded to integer coordinates
        before device execution.
    Expectation: The result is equal to the NumPy result.
    """
    input_np = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(np.float32)
    integer_index_np = np.array([0, 1], dtype=np.int64)
    bool_mask_np = np.array([[True, False, False, False, False],
                             [False, True, False, False, False],
                             [False, False, False, False, False],
                             [False, False, False, False, False]])
    expected_np = input_np.copy()
    expected_np[integer_index_np, :, bool_mask_np] = -1

    output_ms = Tensor(input_np)
    output_ms[Tensor(integer_index_np), :, Tensor(bool_mask_np)] = -1
    _pynative_executor.sync()
    np.testing.assert_array_equal(output_ms.asnumpy(), expected_np)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_setitem_with_1d_bool_tensor_index():
    """
    Feature: Test setitem with a 1-D bool tensor index.
    Description: Verify mixed integer and bool tensor indices with a trailing slice.
    Expectation: The result is equal to the NumPy result.
    """
    input_np = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    integer_index_np = np.array([0, 1], dtype=np.int64)
    bool_mask_np = np.array([True, False, True])
    value_np = np.array([[-1, -2, -3, -4], [-5, -6, -7, -8]], dtype=np.float32)
    expected_np = input_np.copy()
    expected_np[integer_index_np, bool_mask_np, :] = value_np

    output_ms = Tensor(input_np)
    output_ms[Tensor(integer_index_np), Tensor(bool_mask_np), :] = Tensor(value_np)
    _pynative_executor.sync()
    np.testing.assert_array_equal(output_ms.asnumpy(), expected_np)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_setitem_with_uint8_tensor_index():
    """
    Feature: Test setitem with a uint8 tensor mask index.
    Description: Verify uint8 masks follow the same coordinate conversion as bool masks.
    Expectation: The result is equal to the bool-mask NumPy result.
    """
    input_np = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    integer_index_np = np.array([0, 1], dtype=np.int64)
    uint8_mask_np = np.array([1, 0, 1], dtype=np.uint8)
    expected_np = input_np.copy()
    expected_np[integer_index_np, uint8_mask_np.astype(np.bool_), :] = -1

    output_ms = Tensor(input_np)
    output_ms[Tensor(integer_index_np), Tensor(uint8_mask_np), :] = -1
    _pynative_executor.sync()
    np.testing.assert_array_equal(output_ms.asnumpy(), expected_np)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_setitem_with_bool_list_index():
    """
    Feature: Test setitem with an integer tensor and a bool list index.
    Description: Verify bool lists are converted to integer coordinates before device execution.
    Expectation: The result is equal to the NumPy result.
    """
    input_np = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(np.float32)
    integer_index_np = np.array([0, 1], dtype=np.int64)
    bool_list = [True, False, True]
    expected_np = input_np.copy()
    expected_np[integer_index_np, bool_list, :] = -2

    output_ms = Tensor(input_np)
    output_ms[Tensor(integer_index_np), bool_list, :] = -2
    _pynative_executor.sync()
    np.testing.assert_array_equal(output_ms.asnumpy(), expected_np)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_index_put_with_too_many_expanded_indices():
    """
    Feature: Test index_put_ with more mask dimensions than the input rank.
    Description: A bool or uint8 mask consumes as many dimensions as its rank, so a rank 2 mask followed by a
        rank 1 mask already covers three dimensions of a rank 2 tensor. The op must report the invalid index
        count instead of reading the input shape out of bounds.
    Expectation: ValueError is raised and mentions the dimension of the input.
    """
    input_np = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
    mask_2d = Tensor(np.ones((2, 3), dtype=np.bool_))
    mask_1d = Tensor(np.ones((3,), dtype=np.bool_))

    input_ms = Tensor(input_np)
    with pytest.raises(ValueError) as ex:
        input_ms.index_put_((mask_2d, mask_1d), Tensor(5.0, mstype.float32))
        _pynative_executor.sync()
    assert "too many indices for tensor of dimension 2" in str(ex.value)

    input_ms = Tensor(input_np)
    uint8_mask_2d = Tensor(np.ones((2, 3), dtype=np.uint8))
    with pytest.raises(ValueError) as ex:
        input_ms.index_put_((uint8_mask_2d, mask_1d), Tensor(5.0, mstype.float32))
        _pynative_executor.sync()
    assert "too many indices for tensor of dimension 2" in str(ex.value)

    input_3d_np = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    input_ms = Tensor(input_3d_np)
    int_index = Tensor(np.array([0], dtype=np.int64))
    with pytest.raises(ValueError) as ex:
        input_ms.index_put_((int_index, Tensor(np.ones((3, 4), dtype=np.bool_)),
                             Tensor(np.ones((4,), dtype=np.bool_))), Tensor(5.0, mstype.float32))
        _pynative_executor.sync()
    assert "too many indices for tensor of dimension 3" in str(ex.value)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_index_put_with_zero_dim_bool_mask():
    """
    Feature: Test index_put_ with a 0-dim bool mask.
    Description: A 0-dim mask selects the whole tensor when it is True and selects nothing when it is False.
    Expectation: The whole tensor is assigned for True, and the tensor keeps its values for False.
    """
    input_np = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    output_ms = Tensor(input_np)
    output_ms.index_put_((Tensor(True),), Tensor(4.0, mstype.float32))
    _pynative_executor.sync()
    np.testing.assert_array_equal(output_ms.asnumpy(), np.full((2, 3), 4.0, np.float32))

    output_ms = Tensor(input_np)
    output_ms.index_put_((Tensor(False),), Tensor(4.0, mstype.float32))
    _pynative_executor.sync()
    np.testing.assert_array_equal(output_ms.asnumpy(), input_np)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_index_put_with_zero_dim_mask_and_masks():
    """
    Feature: Test index_put_ that mixes a 0-dim mask with another mask.
    Description: A 0-dim mask consumes no dimension of the input, so it must not make the index list look longer
        than the input rank nor be handed to the device together with the expanded mask.
    Expectation: The remaining mask is applied as if the 0-dim mask was absent.
    """
    input_np = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    mask_2d = Tensor(np.ones((2, 3), dtype=np.bool_))

    output_ms = Tensor(input_np)
    output_ms.index_put_((Tensor(True), mask_2d), Tensor(7.0, mstype.float32))
    _pynative_executor.sync()
    np.testing.assert_array_equal(output_ms.asnumpy(), np.full((2, 3), 7.0, np.float32))

    output_ms = Tensor(input_np)
    output_ms.index_put_((mask_2d, Tensor(True)), Tensor(7.0, mstype.float32))
    _pynative_executor.sync()
    np.testing.assert_array_equal(output_ms.asnumpy(), np.full((2, 3), 7.0, np.float32))


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_index_put_with_multiple_zero_dim_masks():
    """
    Feature: Test index_put_ with more than one 0-dim mask.
    Description: 0-dim masks carry no coordinate on their own, so they are not handed to the device together with
        each other and the call must not fail.
    Expectation: The tensor keeps its values.
    """
    input_np = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    output_ms = Tensor(input_np)
    output_ms.index_put_((Tensor(True), Tensor(True)), Tensor(7.0, mstype.float32))
    _pynative_executor.sync()
    np.testing.assert_array_equal(output_ms.asnumpy(), input_np)


def test_setitem_by_nested_unit_list():
    x = np.arange(24).reshape(2, 3, 4).astype(np.float32)

    def cases(x):
        x[[[[0]]], True] = -1
        x[[1], ..., [[[[2]]]]] = -2
        x[0, [[[2]]], [1]] = -3
        return x
    setup_testcase(x, cases)


def test_setitem_with_broadcast():
    """
    Feature: Test setitem with broadcast
    Description: setitem with value need broadcast
    Expectation: Success
    """
    x = np.arange(2*3*4*5*6).reshape(2, 3, 4, 5, 6).astype(np.float32)

    def cases(x):
        x[..., 4] = -1
        x[0, 2] = -2
        x[1, 0, ..., 3] = -3
        x[0, ..., 1, 3, 5] = -4
        return x
    setup_testcase(x, cases)


def test_setitem_mul_by_scalar():
    x = np.ones((4, 5), dtype=np.float32)

    def cases(x):
        x[1, :] = x[1, :]*2
        x[:, 2] = x[:, 3]*3.0
        return x
    setup_testcase(x, cases)


def test_setitem_by_slice():
    x = np.ones((3, 4, 5), dtype=np.float32)

    def cases(x):
        x[1:2] = 2
        x[-3:1] = 3
        x[-10:3:2] = 4
        x[5:0:3] = 5
        x[5:0] = 2
        x[0:-1] = 0
        x[5:5:5] = 6
        x[-1:2] = 7
        return x
    setup_testcase(x, cases)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard')
def test_setitem_by_tuple_of_slices():
    x = np.ones((3, 4, 5), dtype=np.float32)

    def cases(x):
        x[1:2, 2] = 2
        x[0, -4:1] = 3
        x[1, -10:3:2] = 4
        x[5:0:3, 3] = 5
        x[1:1, 2:2] = 6
        return x
    setup_testcase(x, cases)


class TensorItemSetWithNumber(Cell):
    def construct(self, tensor, number_value):
        ret = tensor.itemset(number_value)
        return ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'],
          level_mark='level1',
          card_mark='onecard', mem_peak=0.01)
def test_itemset_with_number():
    net = TensorItemSetWithNumber()
    input_1d_np = np.array([1]).astype(np.float32)
    input_1d_ms = Tensor(input_1d_np, mstype.float32)

    input_3d_np = np.arange(60).reshape(3, 4, 5).astype(np.int32)
    input_3d_ms = Tensor(input_3d_np, mstype.float32)

    value_np_1, value_np_2 = 1, 2.0

    output_1d_ms_1 = net(input_1d_ms, value_np_1)
    output_1d_ms_2 = net(input_1d_ms, value_np_2)

    input_1d_np.itemset(value_np_1)
    assert np.all(output_1d_ms_1.asnumpy() == input_1d_np)
    input_1d_np.itemset(value_np_2)
    assert np.all(output_1d_ms_2.asnumpy() == input_1d_np)

    with pytest.raises(IndexError):
        net(input_3d_ms, value_np_1)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_3d_ms, value_np_2)
        _pynative_executor.sync()


class TensorItemSetByItemWithNumber(Cell):
    def construct(self, tensor, index, number_value):
        ret = tensor.itemset(index, number_value)
        return ret


def test_setitem_dim_expand():
    """
    Feature: Test setitem dim expand
    Description: setitem with value dim need expand
    Expectation: Success
    """
    x = np.ones((2, 3, 4), dtype=np.float32)
    def cases(x):
        x[None, True, [1, 0], (False, True, True), [2]] = 2
        x[([[0]]), ..., [[1]]] = 3
        x[0:1] = 2
        x[..., (0, 1, 2), None, :, True, None] = 3
        return x
    setup_testcase(x, cases)


def test_itemset_by_number_with_number():
    net = TensorItemSetByItemWithNumber()
    input_1d_np = np.array([1]).astype(np.float32)
    input_1d_ms = Tensor(input_1d_np, mstype.float32)

    input_3d_np = np.arange(60).reshape(3, 4, 5).astype(np.int32)
    input_3d_ms = Tensor(input_3d_np, mstype.float32)

    index_np_1, index_np_2, index_np_3, index_np_4 = 0, 30, 60, 2.0
    value_np_1, value_np_2 = 1, 2.0

    output_1d_ms_1 = net(input_1d_ms, index_np_1, value_np_1)
    output_1d_ms_2 = net(input_1d_ms, index_np_1, value_np_2)
    output_3d_ms_1 = net(input_3d_ms, index_np_1, value_np_1)
    output_3d_ms_2 = net(output_3d_ms_1, index_np_1, value_np_2)
    output_3d_ms_3 = net(output_3d_ms_2, index_np_2, value_np_1)
    output_3d_ms_4 = net(output_3d_ms_3, index_np_2, value_np_2)

    input_1d_np.itemset(index_np_1, value_np_1)
    assert np.all(output_1d_ms_1.asnumpy() == input_1d_np)
    input_1d_np.itemset(index_np_1, value_np_2)
    assert np.all(output_1d_ms_2.asnumpy() == input_1d_np)
    input_3d_np.itemset(index_np_1, value_np_1)
    assert np.all(output_3d_ms_1.asnumpy() == input_3d_np)
    input_3d_np.itemset(index_np_1, value_np_2)
    assert np.all(output_3d_ms_2.asnumpy() == input_3d_np)
    input_3d_np.itemset(index_np_2, value_np_1)
    assert np.all(output_3d_ms_3.asnumpy() == input_3d_np)
    input_3d_np.itemset(index_np_2, value_np_2)
    assert np.all(output_3d_ms_4.asnumpy() == input_3d_np)

    with pytest.raises(IndexError):
        net(input_1d_ms, index_np_2, value_np_1)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_1d_ms, index_np_2, value_np_2)
        _pynative_executor.sync()
    with pytest.raises(TypeError):
        net(input_1d_ms, index_np_4, value_np_1)
        _pynative_executor.sync()
    with pytest.raises(TypeError):
        net(input_1d_ms, index_np_4, value_np_2)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_3, value_np_1)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_3, value_np_2)
        _pynative_executor.sync()
    with pytest.raises(TypeError):
        net(input_3d_ms, index_np_4, value_np_1)
        _pynative_executor.sync()
    with pytest.raises(TypeError):
        net(input_3d_ms, index_np_4, value_np_2)
        _pynative_executor.sync()


def test_itemset_by_tuple_with_number():
    net = TensorItemSetByItemWithNumber()
    input_1d_np = np.array([1]).astype(np.float32)
    input_1d_ms = Tensor(input_1d_np, mstype.float32)

    input_3d_np = np.arange(60).reshape(3, 4, 5).astype(np.int32)
    input_3d_ms = Tensor(input_3d_np, mstype.float32)

    index_np_1, index_np_2, index_np_3, index_np_4, index_np_5 = (0,), (1, 2), (1, 1, 0), (3, 4, 5), (1, 2, 3, 4)
    value_np_1, value_np_2 = 1, 2.0

    output_1d_ms_1 = net(input_1d_ms, index_np_1, value_np_1)
    input_1d_np.itemset(index_np_1, value_np_1)
    assert np.all(output_1d_ms_1.asnumpy() == input_1d_np)

    output_1d_ms_2 = net(input_1d_ms, index_np_1, value_np_2)
    input_1d_np.itemset(index_np_1, value_np_2)
    assert np.all(output_1d_ms_2.asnumpy() == input_1d_np)

    output_3d_ms_1 = net(input_3d_ms, index_np_3, value_np_1)
    input_3d_np.itemset(index_np_3, value_np_1)
    assert np.all(output_3d_ms_1.asnumpy() == input_3d_np)

    output_3d_ms_2 = net(input_3d_ms, index_np_3, value_np_2)
    input_3d_np.itemset(index_np_3, value_np_2)
    assert np.all(output_3d_ms_2.asnumpy() == input_3d_np)

    with pytest.raises(IndexError):
        net(input_1d_ms, index_np_2, value_np_1)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_1d_ms, index_np_2, value_np_2)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_1, value_np_1)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_1, value_np_2)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_2, value_np_1)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_2, value_np_2)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_4, value_np_1)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_4, value_np_2)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_5, value_np_1)
        _pynative_executor.sync()
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_5, value_np_2)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'],
          level_mark='level1',
          card_mark='onecard', mem_peak=1024.01)
def test_itemset_all():
    """
    Feature: Test setitem
    Description: Test setitem
    Expectation: success
    """
    test_setitem_grad()
    test_setitem_by_list()
    test_setitem_dtype()
    test_setitem_by_tuple_with_int()
    test_setitem_by_tuple_with_list()
    test_setitem_by_nested_unit_list()
    test_setitem_mul_by_scalar()
    test_setitem_by_slice()
    test_setitem_dim_expand()
    test_itemset_by_number_with_number()
    test_itemset_by_tuple_with_number()
