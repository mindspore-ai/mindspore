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
"""test flatten tensors"""
import numpy as np
import mindspore as ms
import mindspore.common.initializer as init
from mindspore.common import Tensor, Parameter
from mindspore.nn import Cell


def test_flatten_tensors_basic():
    """
    Feature: Flatten tensors.
    Description: Basic function for flatten tensors.
    Expectation: Flatten tensor works as expected.
    """
    t1 = Tensor(np.ones([2], np.float32))
    t2 = Tensor(np.ones([2, 2], np.float32))
    t3 = Tensor(np.ones([2, 2, 2], np.float32))
    # Before flatten.
    assert not Tensor._is_flattened([t1, t2, t3])  # pylint: disable=W0212
    assert not Tensor._get_flattened_tensors([t1, t2, t3])  # pylint: disable=W0212
    # Do flatten.
    chunks = Tensor._flatten_tensors([t1, t2, t3])  # pylint: disable=W0212
    # After flatten.
    assert len(chunks) == 1
    assert Tensor._is_flattened([t1, t2, t3])  # pylint: disable=W0212
    assert chunks[0].dtype == ms.float32
    assert chunks[0].shape == [14]
    assert np.allclose(chunks[0].asnumpy(), np.ones([14], np.float32))
    # Get flattened tensors.
    chunks2 = Tensor._get_flattened_tensors([t1, t2, t3])  # pylint: disable=W0212
    assert chunks == chunks2


def test_flatten_tensors_order():
    """
    Feature: Flatten tensors.
    Description: Test flatten tensors in order.
    Expectation: Flatten tensor works as expected.
    """
    t1 = Tensor([1], ms.float32)
    t2 = Tensor([2], ms.float32)
    t3 = Tensor([3], ms.float32)
    chunks = Tensor._flatten_tensors([t1, t2, t3])  # pylint: disable=W0212
    assert len(chunks) == 1
    assert np.allclose(chunks[0].asnumpy(), np.array([1, 2, 3]))
    chunks = Tensor._flatten_tensors([t3, t1, t2])  # pylint: disable=W0212
    assert len(chunks) == 1
    assert np.allclose(chunks[0].asnumpy(), np.array([3, 1, 2]))


def test_flatten_tensors_float16():
    """
    Feature: Flatten tensors.
    Description: Test flatten tensors for float16.
    Expectation: Flatten tensor works as expected.
    """
    t1 = Tensor([1], ms.float16)
    t2 = Tensor([2], ms.float16)
    t3 = Tensor([3], ms.float16)
    chunks = Tensor._flatten_tensors([t1, t2, t3])  # pylint: disable=W0212
    assert len(chunks) == 1
    assert np.allclose(chunks[0].asnumpy(), np.array([1, 2, 3]))
    chunks = Tensor._flatten_tensors([t3, t1, t2])  # pylint: disable=W0212
    assert len(chunks) == 1
    assert np.allclose(chunks[0].asnumpy(), np.array([3, 1, 2]))


def test_flatten_tensors_scalar():
    """
    Feature: Flatten tensors.
    Description: Test flatten tensors for scalar tensor.
    Expectation: Flatten tensor works as expected.
    """
    t1 = Tensor(1)
    t2 = Tensor(2)
    t3 = Tensor(3)
    chunks = Tensor._flatten_tensors([t1, t2, t3])  # pylint: disable=W0212
    assert len(chunks) == 1
    assert np.allclose(chunks[0].asnumpy(), np.array([1, 2, 3]))
    chunks = Tensor._flatten_tensors([t3, t1, t2])  # pylint: disable=W0212
    assert len(chunks) == 1
    assert np.allclose(chunks[0].asnumpy(), np.array([3, 1, 2]))


def test_flatten_tensors_dtypes():
    """
    Feature: Flatten tensors.
    Description: Flatten tensors group by data types.
    Expectation: Flatten tensor works as expected.
    """
    t1 = Tensor(np.ones([2], np.float32))
    t2 = Tensor(np.ones([2, 2], np.float32))
    t3 = Tensor(np.ones([2, 2, 2], np.float32))
    t4 = Tensor(np.ones([3, 3], np.float64))
    t5 = Tensor(np.ones([3, 3, 3], np.float64))
    chunks = Tensor._flatten_tensors([t1, t2, t3, t4, t5])  # pylint: disable=W0212
    assert len(chunks) == 2
    assert chunks[0].dtype == ms.float32
    assert chunks[0].shape == [14]
    assert np.allclose(chunks[0].asnumpy(), np.ones([14], np.float32))
    assert chunks[1].dtype == ms.float64
    assert chunks[1].shape == [36]
    assert np.allclose(chunks[1].asnumpy(), np.ones([36], np.float64))
    # Different order.
    chunks1 = Tensor._flatten_tensors([t4, t1, t2, t5, t3])  # pylint: disable=W0212
    assert np.allclose(chunks[0].asnumpy(), chunks1[0].asnumpy())


def test_cell_flatten_weights():
    """
    Feature: Flatten tensors.
    Description: Flatten weights for Cell.
    Expectation: Flatten weights works as expected.
    """
    class MyCell(Cell):
        def __init__(self):
            super(MyCell, self).__init__()
            self.para1 = Parameter(Tensor([1, 2], ms.float32))
            self.para2 = Parameter(Tensor([3, 4, 5], ms.float32))
            self.para3 = Parameter(Tensor([6], ms.float32))

        def construct(self, x):
            return x

    net = MyCell()
    assert not Parameter._is_flattened(net.trainable_params())  # pylint: disable=W0212
    net.flatten_weights()
    assert Parameter._is_flattened(net.trainable_params())  # pylint: disable=W0212
    chunks = Parameter._get_flattened_tensors(net.trainable_params())  # pylint: disable=W0212
    assert np.allclose(chunks[0].asnumpy(), np.array([1, 2, 3, 4, 5, 6]))


def test_cell_flatten_weights_with_init():
    """
    Feature: Flatten tensors.
    Description: Flatten weights for Cell with parameter initializer.
    Expectation: Flatten weights works as expected.
    """
    class MyCell(Cell):
        def __init__(self):
            super(MyCell, self).__init__()
            self.para1 = Parameter(Tensor([1, 2], ms.float32))
            self.para2 = Parameter(init.initializer('ones', [3], ms.float32))
            self.para3 = Parameter(Tensor([6], ms.float32))

        def construct(self, x):
            return x

    net = MyCell()
    assert not Parameter._is_flattened(net.trainable_params())  # pylint: disable=W0212
    net.flatten_weights()
    assert Parameter._is_flattened(net.trainable_params())  # pylint: disable=W0212
    chunks = Parameter._get_flattened_tensors(net.trainable_params())  # pylint: disable=W0212
    assert np.allclose(chunks[0].asnumpy(), np.array([1, 2, 1, 1, 1, 6]))
