# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test python parser of pyboost"""

import mindspore as ms
import numpy as np
import pytest

from tests.mark_utils import arg_mark

def test_tensor_narrow_with_dim_argument_provided_in_various_types():
    """
    Feature: Tensor.
    Description: Test convert logic from PyObject to C++\
        int with Tensor.narrow.
    Expectation: Run success
    """
    x = ms.Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), ms.float32)
    expect_out = np.array([[0.3, 3.6], [0.5, -3.2]]).astype(np.float32)
    start = 1
    length = 2

    dims = [1, np.longlong(1), ms.Tensor(1)]
    for dim in dims:
        out = x.narrow(dim, start, length)
        assert np.allclose(out.asnumpy(), expect_out)


def test_tensor_narrow_with_invalid_dim():
    """
    Feature: Tensor.
    Description: Test convert logic from PyObject to C++\
        int with Tensor.narrow.
    Expectation: Run success
    """
    x = ms.Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), ms.float32)
    start = 1
    length = 2

    invalid_dims = [2., ms.Tensor(2.), ms.Tensor([2, 1]), np.float64(2.),
                    np.array(2), np.array([2]), np.array(2).astype(np.float32)]
    for dim in invalid_dims:
        with pytest.raises((TypeError, ValueError)):
            x.narrow(dim, start, length)


def test_mint_argmin_with_the_dim_argument():
    """
    Feature: Function.
    Description: Test convert logic from PyObject to C++\
        Int64Imm with mint.argmin.
    Expectation: Run success
    """
    x = ms.Tensor(np.random.randn(4, 4).astype(np.float32))
    for dim in [-1, np.longlong(-1), ms.Tensor(-1)]:
        out = ms.mint.argmin(x, dim=dim)
        assert out.shape == (4,)


def test_mint_argmin_with_invalid_dim():
    """
    Feature: Function.
    Description: Test convert logic from PyObject to C++\
        Int64Imm with mint.argmin.
    Expectation: Run success
    """
    x = ms.Tensor(np.random.randn(4, 4).astype(np.float32))
    for dim in [4.0, np.float64(4.), np.array(4), ms.Tensor(4.), ms.Tensor([4, 4])]:
        with pytest.raises((TypeError, ValueError)):
            ms.mint.argmin(x, dim)


def test_tensor_reshape_with_the_shape_argument_provided_in_various_types():
    """
    Feature: Tensor.
    Description: Test convert logic from PyObject to C++\
        vector<int> with Tensor.reshape.
    Expectation: Run success
    """
    x = ms.Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), ms.float32)
    outputs = []

    first_elements = [3, ms.Tensor(3), ms.Tensor([3]), np.longlong(3),
                      np.array([3]).astype(np.int8)[0], np.array(3)]
    second_elements = [2, ms.Tensor(2), ms.Tensor([2]), np.longlong(2),
                       np.array([2]).astype(np.int8)[0], np.array(2)]
    for first_element in first_elements:
        for second_element in second_elements:
            # *shape
            outputs.append(x.reshape(first_element, second_element))
            # shape is tuple
            shape = (first_element, second_element)
            outputs.append(x.reshape(shape))
            # shape is list
            shape = [first_element, second_element]
            outputs.append(x.reshape(shape))

    expect_output = np.array([[-0.1, 0.3],
                              [3.6, 0.4],
                              [0.5, -3.2]], dtype=np.float32)
    for output in outputs:
        assert np.allclose(output.asnumpy(), expect_output)


def test_tensor_reshape_with_various_invalid_shape():
    """
    Feature: Tensor.
    Description: Test convert logic from PyObject to C++\
        vector<int> with Tensor.reshape.
    Expectation: Run success
    """
    x = ms.Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), ms.float32)

    first_elements = [3, ms.Tensor(3), ms.Tensor([3]), np.longlong(3),
                      np.array([3]).astype(np.int8)[0], np.array(3)]
    second_elements = [2., ms.Tensor(2.), ms.Tensor([2, 1]), np.float64(2.),
                       np.array([2]), np.array(2).astype(np.float32)]
    for first_element in first_elements:
        for second_element in second_elements:
            with pytest.raises((TypeError, ValueError)):
                # *shape
                x.reshape(first_element, second_element)
            with pytest.raises((TypeError, ValueError)):
                shape = (first_element, second_element)
                x.reshape(shape)
            with pytest.raises((TypeError, ValueError)):
                shape = [first_element, second_element]
                x.reshape(shape)


def test_tensor_repeat_with_the_repeats_argument_provided_in_various_types():
    """
    Feature: Tensor.
    Description: Test convert logic from PyObject to C++ ValueTuple(int)\
        with Tensor.repeat.
    Expectation: Run success
    """
    outputs = []
    x = ms.Tensor([1, 2, 3])
    expect = ms.Tensor([[1, 2, 3, 1, 2, 3],
                        [1, 2, 3, 1, 2, 3]])
    first_elements = [2, ms.Tensor(2), ms.Tensor([2]), np.longlong(2),
                      np.array([2]).astype(np.int8)[0], np.array(2)]
    second_elements = [2, ms.Tensor(2), ms.Tensor([2]), np.longlong(2),
                       np.array([2]).astype(np.int8)[0], np.array(2)]
    for first in first_elements:
        for second in second_elements:
            outputs.append(x.repeat(first, second))
            outputs.append(x.repeat((first, second)))
            outputs.append(x.repeat([first, second]))
    for output in outputs:
        assert np.allclose(output.asnumpy(), expect)


def test_tensor_repeat_with_invalid_repeats_argument():
    """
    Feature: Tensor.
    Description: Test convert logic from PyObject to C++ ValueTuple(int)\
        with Tensor.repeat.
    Expectation: Run success
    """
    x = ms.Tensor([1, 2, 3])
    first_elements = [3, ms.Tensor(3), ms.Tensor([3]), np.longlong(3),
                      np.array([3]).astype(np.int8)[0], np.array(3)]
    second_elements = [2., ms.Tensor(2.), ms.Tensor([2, 1]), np.float64(2.),
                       np.array([2]), np.array(2).astype(np.float32)]
    for first in first_elements:
        for second in second_elements:
            with pytest.raises((TypeError, ValueError)):
                # *repeats
                x.repeat(first, second)
            with pytest.raises((TypeError, ValueError)):
                repeats = (first, second)
                x.repeat(repeats)
            with pytest.raises((TypeError, ValueError)):
                repeats = [first, second]
                x.repeat(repeats)


def test_mint_n_f_elu_with_alpha_argument_provided_in_various_types():
    """
    Feature: Function.
    Description: Test convert logic from PyObject to C++\
        FloatImm with mint.nn.functional.elu.
    Expectation: Run success
    """
    x = ms.Tensor([2., 2.])
    expect_out = np.array([2., 2.])
    for alpha in [2.5, np.float64(2.5), np.array([2.5]).astype(np.float16)[0], ms.Tensor(2.5)]:
        out = ms.mint.nn.functional.elu(x, alpha)
        assert np.allclose(out.asnumpy(), expect_out)


def test_mint_n_f_elu_with_invalid_alpha_argument():
    """
    Feature: Function.
    Description: Test convert logic from PyObject to C++\
        FloatImm with mint.nn.functional.elu.
    Expectation: Run success
    """
    x = ms.Tensor([2., 2.])
    for alpha in [2, True, np.longlong(2), np.array(2.5), ms.Tensor(2), ms.Tensor([2.5])]:
        with pytest.raises((TypeError, ValueError)):
            ms.mint.nn.functional.elu(x, alpha)


def test_mint_n_f_interpolate_with_scales_argument_provided_in_various_types():
    """
    Feature: Function.
    Description: Test convert logic from PyObject to C++\
        ValueTuple[float] with mint.nn.functional.interpolate.
    Expectation: Run success
    """
    x = ms.Tensor(np.random.randn(1, 1, 3, 4).astype(np.float32))
    candidates = [2.5, np.array([2.5]).astype(np.float16)[0]]
    for first in candidates:
        for second in candidates:
            scales = [first, second]
            out = ms.mint.nn.functional.interpolate(x, None, scales)
            assert out.shape == (1, 1, 7, 10)


def test_mint_n_f_interpolate_with_invalid_scales_argument():
    """
    Feature: Function.
    Description: Test convert logic from PyObject to C++\
        ValueTuple[float] with mint.nn.functional.interpolate.
    Expectation: Run success
    """
    x = ms.Tensor(np.random.randn(1, 1, 3, 4).astype(np.float32))
    candidates = [2, True, np.longlong(2), np.array([2.5]), ms.Tensor(2)]
    for first in candidates:
        for second in candidates:
            scales = [first, second]
            with pytest.raises((TypeError, ValueError)):
                ms.mint.nn.functional.interpolate(x, None, scales)


def test_tensor_inplace_masked_fill_with_value_argument_provided_in_various_types():
    """
    Feature: Tensor.
    Description: Test convert logic from PyObject to C++\
        Scalar with tensor.masked_fill_.
    Expectation: Run success
    """
    x = ms.Tensor([1., 2., 3.])
    mask = ms.Tensor([False, False, True])
    values = [1., 2., 3., 4., 1., 0., 1., 1., 3.]
    candidates = [1, np.longlong(2.), 3., np.array([4.]).astype(np.float16)[0],
                  True, np.array([False])[0], ms.Tensor(1), ms.Tensor(True), ms.Tensor(3.)]
    for i, candidate in enumerate(candidates):
        x.masked_fill_(mask, candidate)
        expect_out = np.array([1., 2., values[i]])
        assert np.allclose(x.asnumpy(), expect_out)


def test_tensor_inplace_masked_fill_with_invalid_value_argument():
    """
    Feature: Tensor.
    Description: Test convert logic from PyObject to C++\
        Scalar with tensor.masked_fill_.
    Expectation: Run success
    """
    x = ms.Tensor([1., 2., 3.])
    mask = ms.Tensor([False, False, True])
    candidates = [np.array(1), np.array(True)]
    for _, candidate in enumerate(candidates):
        with pytest.raises((TypeError, ValueError)):
            x.masked_fill_(mask, candidate)


def test_disable_scalar_tensor_with_mint_min():
    """
    Feature: Tensor scalar conversion control
    Description: Test disable scalar tensor conversion in mint.min overload API
        - Verify dim argument handling
        - Verify tensor vs scalar overload selection
    Expectation: Run success
    """
    x = ms.Tensor([[1., 2.], [0.7, 1.5]])

    # mint.min(input, dim, keepdim=False)
    values, index = ms.mint.min(x, 1)
    expect_values = np.array([1., 0.7]).astype(np.float32)
    expect_index = np.array([0, 0])
    np.allclose(values.asnumpy(), expect_values)
    np.allclose(index.asnumpy(), expect_index)

    # mint.min(input, other)
    y = ms.mint.min(x, ms.Tensor(1))
    expect = np.array([[1., 1.], [0.7, 1.]]).astype(np.float32)
    np.allclose(y.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
def test_numpy_support_in_pyboost():
    """
    Feature: Function.
    Description: Test convert logic from numpy PyObject to C++ Object
    Expectation: Run success
    """
    # basic int test
    test_tensor_narrow_with_dim_argument_provided_in_various_types()
    test_tensor_narrow_with_invalid_dim()

    # int test
    test_mint_argmin_with_the_dim_argument()
    test_mint_argmin_with_invalid_dim()

    # basic vector<int> test
    test_tensor_reshape_with_the_shape_argument_provided_in_various_types()
    test_tensor_reshape_with_various_invalid_shape()

    # list[int] test
    test_tensor_repeat_with_the_repeats_argument_provided_in_various_types()
    test_tensor_repeat_with_invalid_repeats_argument()

    # float test
    test_mint_n_f_elu_with_alpha_argument_provided_in_various_types()
    test_mint_n_f_elu_with_invalid_alpha_argument()

    # list[float] test
    test_mint_n_f_interpolate_with_scales_argument_provided_in_various_types()
    test_mint_n_f_interpolate_with_invalid_scales_argument()

    # scalar test
    test_tensor_inplace_masked_fill_with_value_argument_provided_in_various_types()
    test_tensor_inplace_masked_fill_with_invalid_value_argument()

    # disable scalar tensor with overload api
    test_disable_scalar_tensor_with_mint_min()
