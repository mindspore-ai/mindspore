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
        x[[0, 1], [1, 2], [1, 3]] = [3, 4]
        x[([0, 1], [0, 2], [1, 1])] = [10, 5]
        x[[0, 1], ..., [0, 1]] = 4
        return x


class NumpySetItemByList():
    def __call__(self, x):
        x[[0, 1], [1, 2], [1, 3]] = [3, 4]
        x[([0, 1], [0, 2], [1, 1])] = [10, 5]
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
    x = np.ones((2, 3, 4), dtype=np.float32)

    def cases(x):
        x[[0, 1], [1, 2], [1, 3]] = [3, 4]
        x[([0, 1], [0, 2], [1, 1])] = [10, 5]
        x[[0, 1], ..., [0, 1]] = 4
        return x
    setup_testcase(x, cases)


def test_setitem_with_sequence():
    x = np.ones((2, 3, 4), dtype=np.float32)

    def cases(x):
        x[...] = [3]
        x[..., 1] = ([1, 2, 3], [4, 5, 6])
        x[0] = ((0, 1, 2, 3), (4, 5, 6, 7), [8, 9, 10, 11])
        x[1:2] = ((0, 1, 2, 3), (4, 5, 6, 7), [8, 9, 10, 11])
        return x
    setup_testcase(x, cases)


def test_setitem_dtype():
    x = np.ones((2, 3, 4), dtype=np.float32)

    def cases(x):
        x[...] = 3
        x[..., 1] = 3.0
        x[0] = True
        x[1:2] = ((0, False, 2, 3), (4.0, 5, 6, 7), [True, 9, 10, 11])
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
    x = np.arange(24).reshape(2, 3, 4).astype(np.float32)

    def cases(x):
        x[..., 2, False, 1] = [-1]
        x[0, True, 0, None, True] = [-2, -2, -2, -2]
        x[0, ..., None] = [[-3], [-3], [-3], [-3]]
        x[..., 0, None, 1, True, True, None] = [[[-4]], [[-4]]]
        x[None, True, [1, 0], (False, True, True), [2]] = [[2, 3]]
        return x
    setup_testcase(x, cases)


def test_setitem_by_nested_unit_list():
    x = np.arange(24).reshape(2, 3, 4).astype(np.float32)

    def cases(x):
        x[[[[0]]], True] = -1
        x[[1], ..., [[[[2]]]]] = -2
        x[0, [[[2]]], [1]] = -3
        return x
    setup_testcase(x, cases)


def test_setitem_with_broadcast():
    x = np.arange(2*3*4*5*6).reshape(2, 3, 4, 5, 6).astype(np.float32)
    v1 = np.full((1, 4, 5), -1).tolist()
    v2 = np.full((4, 1, 6), -2).tolist()

    def cases(x):
        x[..., 4] = v1
        x[0, 2] = v2
        x[1, 0, ..., 3] = [[-3], [-3], [-3], [-3]]
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
        x[1:0:-1] = 8
        return x
    setup_testcase(x, cases)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
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
          card_mark='onecard',
          essential_mark='essential')
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
    with pytest.raises(IndexError):
        net(input_3d_ms, value_np_2)


class TensorItemSetByItemWithNumber(Cell):
    def construct(self, tensor, index, number_value):
        ret = tensor.itemset(index, number_value)
        return ret


def test_setitem_dim_expand():
    x = np.ones((2, 3, 4), dtype=np.float32)
    def cases(x):
        x[None, True, [1, 0], (False, True, True), [2]] = 2
        x[([[0]]), ..., [[1]]] = [[[3, 3, 3]]]
        x[0:1] = [[2, 3, 4, 5]]
        x[..., (0, 1, 2), None, :, True, None] = [[[3], [3], [3], [3]]]
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
    with pytest.raises(IndexError):
        net(input_1d_ms, index_np_2, value_np_2)
    with pytest.raises(TypeError):
        net(input_1d_ms, index_np_4, value_np_1)
    with pytest.raises(TypeError):
        net(input_1d_ms, index_np_4, value_np_2)
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_3, value_np_1)
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_3, value_np_2)
    with pytest.raises(TypeError):
        net(input_3d_ms, index_np_4, value_np_1)
    with pytest.raises(TypeError):
        net(input_3d_ms, index_np_4, value_np_2)


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
    with pytest.raises(IndexError):
        net(input_1d_ms, index_np_2, value_np_2)
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_1, value_np_1)
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_1, value_np_2)
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_2, value_np_1)
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_2, value_np_2)
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_4, value_np_1)
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_4, value_np_2)
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_5, value_np_1)
    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_5, value_np_2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_itemset_all():
    """
    Feature: Test setitem
    Description: Test setitem
    Expectation: success
    """
    test_setitem_grad()
    test_setitem_by_list()
    test_setitem_with_sequence()
    test_setitem_dtype()
    test_setitem_by_tuple_with_int()
    test_setitem_by_tuple_with_list()
    test_setitem_by_nested_unit_list()
    test_setitem_mul_by_scalar()
    test_setitem_by_slice()
    test_setitem_dim_expand()
    test_itemset_by_number_with_number()
    test_itemset_by_tuple_with_number()
