# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Numerical and argument-boundary coverage for mathematical function APIs."""
# Unit coverage intentionally exercises internal Python inference and wrapper helpers.
# pylint: disable=protected-access
import itertools
import numpy as np
import pytest

import mindspore as ms
from mindspore.ops.function import math_func as math
from tests.mark_utils import arg_mark


@pytest.fixture(scope='module', autouse=True, name='cpu_context')
def fixture_cpu_context():
    """Use CPU kernels and restore global execution settings."""
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    yield
    ms.set_context(**previous)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('k', [0, 1, 2, 3, -1, 5])
@pytest.mark.parametrize('dims', [(0, 1), (1, 0), (-2, -1)])
def test_rot90_direction_and_wrapped_turns(k, dims):
    """
    Feature: Rot90 direction and wrapped turns.
    Description: Signed turns and axis order must agree with NumPy on nonsquare matrices.
    Expectation: Signed turns and axis order must agree with NumPy on nonsquare matrices.
    """
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = math.rot90(ms.Tensor(array), k, dims)
    np.testing.assert_array_equal(result.asnumpy(), np.rot90(array, k, dims))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind,k,dims,error', [
    ('list', 1, (0, 1), TypeError), ('tensor', 1.5, (0, 1), TypeError),
    ('tensor', 1, 'bad', TypeError), ('tensor', 1, (0,), ValueError),
    ('tensor', 1, (0, 0), RuntimeError), ('tensor', 1, (0, -2), RuntimeError),
    ('tensor', 1, (2, 1), ValueError), ('tensor', 1, (0, -3), ValueError),
])
def test_rot90_rejects_invalid_rotation_specification(kind, k, dims, error):
    """
    Feature: Rot90 rejects invalid rotation specification.
    Description: Malformed inputs and axes must fail before attempting tensor reversal.
    Expectation: Malformed inputs and axes must fail before attempting tensor reversal.
    """
    value = ms.Tensor(np.ones((2, 3), np.float32)) if kind == 'tensor' else [[1, 2]]
    with pytest.raises(error):
        math.rot90(value, k, dims)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('left,right', [
    (np.array(2, np.float32), np.array([1, -2], np.float32)),
    (np.array([1, -2], np.float32), np.array(2, np.float32)),
    (np.arange(6, dtype=np.float32).reshape(2, 3), np.array([2, -1], np.float32)),
    (np.array([2, -1], np.float32), np.arange(6, dtype=np.float32).reshape(2, 3)),
])
def test_kronecker_product_handles_scalar_and_unequal_ranks(left, right):
    """
    Feature: Kronecker product handles scalar and unequal ranks.
    Description: Inserted singleton axes must retain the Kronecker block ordering.
    Expectation: Inserted singleton axes must retain the Kronecker block ordering.
    """
    np.testing.assert_array_equal(math.kron(ms.Tensor(left), ms.Tensor(right)).asnumpy(), np.kron(left, right))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('left,right', [(None, ms.Tensor([1.])), (ms.Tensor([1.]), None)])
def test_kronecker_product_rejects_non_tensor_operands(left, right):
    """
    Feature: Kronecker product rejects non tensor operands.
    Description: Both inputs must be tensors, rather than silently treating None as a scalar.
    Expectation: Both inputs must be tensors, rather than silently treating None as a scalar.
    """
    with pytest.raises(TypeError, match='must be Tensor'):
        math.kron(left, right)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('rank', [1, 2, 3])
def test_atleast_nd_preserves_values_and_inserts_axes(rank):
    """
    Feature: Conversion to at least N dimensions.
    Description: Scalar, vector and matrix inputs follow NumPy's axis insertion rules.
    Expectation: Scalar, vector and matrix inputs follow NumPy's axis insertion rules.
    """
    values = [np.array(2, np.float32), np.array([1, 2], np.float32), np.arange(6, dtype=np.float32).reshape(2, 3)]
    function = getattr(math, f'atleast_{rank}d')
    oracle = getattr(np, f'atleast_{rank}d')
    results = function(tuple(ms.Tensor(value) for value in values))
    for result, value in zip(results, values):
        np.testing.assert_array_equal(result.asnumpy(), oracle(value))
    np.testing.assert_array_equal(function(ms.Tensor(values[1])).asnumpy(), oracle(values[1]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('values', [[np.array(2, np.float32), np.array(3, np.float32)],
                                   [np.array([1, 2], np.float32), np.array([3, 4], np.float32)],
                                   [np.ones((2, 2), np.float32), np.zeros((1, 2), np.float32)]])
def test_vstack_normalizes_scalar_vector_and_matrix_inputs(values):
    """
    Feature: Vstack normalizes scalar vector and matrix inputs.
    Description: Stacking must preserve row order after promoting low-rank operands.
    Expectation: Stacking must preserve row order after promoting low-rank operands.
    """
    result = math.vstack([ms.Tensor(value) for value in values])
    np.testing.assert_array_equal(result.asnumpy(), np.vstack(values))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('value', [None, (), [1, 2]])
def test_vstack_requires_nonempty_tensor_sequence(value):
    """
    Feature: Vstack requires nonempty tensor sequence.
    Description: Invalid containers and non-tensor members are independently rejected.
    Expectation: Invalid containers and non-tensor members are independently rejected.
    """
    with pytest.raises(TypeError):
        math.vstack(value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('replacement', [False, True])
@pytest.mark.parametrize('r', [1, 2, 3, 4])
def test_combinations_matches_lexicographic_enumeration(replacement, r):
    """
    Feature: Combinations matches lexicographic enumeration.
    Description: Combination ordering and replacement behavior must match Python's iterator.
    Expectation: Combination ordering and replacement behavior must match Python's iterator.
    """
    data = [1., 3., 5.]
    result = math.combinations(ms.Tensor(data, ms.float32), r, replacement)
    oracle = itertools.combinations_with_replacement if replacement else itertools.combinations
    expected = np.array(list(oracle(data, r)), np.float32)
    np.testing.assert_array_equal(result.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('value,r,error', [([1, 2], 2, TypeError),
                                        (ms.Tensor([[1, 2]]), 2, ValueError),
                                        (ms.Tensor([1, 2]), 1.5, TypeError)])
def test_combinations_rejects_wrong_input_rank_and_count_type(value, r, error):
    """
    Feature: Combinations rejects wrong input rank and count type.
    Description: Only one-dimensional tensors and integer selection sizes are supported.
    Expectation: Only one-dimensional tensors and integer selection sizes are supported.
    """
    with pytest.raises(error):
        math.combinations(value, r)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('function', ['hann_window', 'kaiser_window'])
@pytest.mark.parametrize('length', [0, 1, 4, 5])
@pytest.mark.parametrize('periodic', [False, True])
def test_window_functions_match_numpy(function, length, periodic):
    """
    Feature: Window functions match numpy.
    Description: Periodic windows omit the repeated final endpoint of a symmetric window.
    Expectation: Periodic windows omit the repeated final endpoint of a symmetric window.
    """
    extra = 1 if periodic and length > 1 else 0
    oracle = np.hanning(length + extra) if function == 'hann_window' else np.kaiser(length + extra, 4.0)
    if extra:
        oracle = oracle[:-1]
    kwargs = {'beta': 4.0} if function == 'kaiser_window' else {}
    result = getattr(math, function)(length, periodic, dtype=ms.float32, **kwargs)
    np.testing.assert_allclose(result.asnumpy(), oracle, rtol=1e-6, atol=1e-7)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('function', ['hann_window', 'kaiser_window'])
@pytest.mark.parametrize('length,periodic,dtype,error', [('4', True, None, TypeError),
                                                      (-1, True, None, ValueError),
                                                      (4, 'yes', None, TypeError),
                                                      (4, True, ms.int32, TypeError)])
def test_window_functions_reject_invalid_arguments(function, length, periodic, dtype, error):
    """
    Feature: Window functions reject invalid arguments.
    Description: Window length, periodic flag and output dtype must have valid types.
    Expectation: Window length, periodic flag and output dtype must have valid types.
    """
    with pytest.raises(error):
        getattr(math, function)(length, periodic, dtype=dtype)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('shape,dim,expected', [(None, None, ([2, 3, 4], [0, 1, 2])),
                                             ((5, -1), None, ([5, 4], [1, 2])),
                                             (None, (-1, 0), ([4, 2], [2, 0])),
                                             ((-1, 6), (0, 2), ([2, 6], [0, 2]))])
def test_fft_shape_canonicalization_preserves_requested_axis_order(shape, dim, expected):
    """
    Feature: Fft shape canonicalization preserves requested axis order.
    Description: Negative FFT axes and implicit lengths must use the input's true dimensions.
    Expectation: Negative FFT axes and implicit lengths must use the input's true dimensions.
    """
    tensor = ms.Tensor(np.ones((2, 3, 4), np.complex64))
    assert math._canonicalize_fft_shape_and_dim(tensor, shape, dim) == expected


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('shape,dim,reason', [(None, (0, 0), 'unique'), ((2,), (0, 1), 'same length'),
                                           ((1, 2, 3, 4), None, 'dimensions'),
                                           ((0,), (1,), 'greater than 0'), (None, (3,), 'out of range')])
def test_fft_shape_canonicalization_rejects_inconsistent_dimensions(shape, dim, reason):
    """
    Feature: Fft shape canonicalization rejects inconsistent dimensions.
    Description: Duplicate axes and impossible transform sizes must fail before kernel selection.
    Expectation: Duplicate axes and impossible transform sizes must fail before kernel selection.
    """
    with pytest.raises(ValueError, match=reason):
        math._canonicalize_fft_shape_and_dim(ms.Tensor(np.ones((2, 3, 4), np.complex64)), shape, dim)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('shape,dim,norm,name,expected_shape', [
    (None, None, None, 'FFTN', (1, 2, 3)),
    ((4,), (1,), 'ortho', 'IFFTN', (2, 4)),
    ((2,), (1,), 'forward', 'FFTN', (2, 2)),
    (3, -1, None, 'FFTN', (2, 3)),
])
def test_fft_input_preparation_resizes_and_tracks_metadata(shape, dim, norm, name, expected_shape):
    """
    Feature: Fft input preparation resizes and tracks metadata.
    Description: FFT preparation must retain axis metadata after padding or truncating samples.
    Expectation: FFT preparation must retain axis metadata after padding or truncating samples.
    """
    tensor = ms.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3).astype(np.complex64))
    result = math._check_fftwithsize_input(tensor, shape, dim, norm, name)
    assert result.input.shape == expected_shape
    assert result.input_dim == 2
    assert result.norm == ('backward' if norm is None else norm)
    if shape == (4,):
        np.testing.assert_array_equal(result.input.asnumpy(), [[0, 1, 2, 0], [3, 4, 5, 0]])
    elif shape == (2,):
        np.testing.assert_array_equal(result.input.asnumpy(), [[0, 1], [3, 4]])
    else:
        np.testing.assert_array_equal(result.input.asnumpy().reshape(2, 3), [[0, 1, 2], [3, 4, 5]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('input_kind,shape,dim,norm,name', [
    ('list', None, None, None, 'FFTN'), ('float', None, None, None, 'FFTN'),
    ('complex', None, None, None, 'RFFT'), ('complex', [2], None, None, 'FFTN'),
    ('complex', (2.5,), None, None, 'FFTN'), ('complex', None, [1], None, 'FFTN'),
    ('complex', None, (1.5,), None, 'FFTN'), ('complex', None, None, 1, 'FFTN'),
])
def test_fft_input_preparation_rejects_invalid_types(input_kind, shape, dim, norm, name):
    """
    Feature: Fft input preparation rejects invalid types.
    Description: FFT type validation must reject invalid input, shape, axes and norm independently.
    Expectation: FFT type validation must reject invalid input, shape, axes and norm independently.
    """
    value = [1, 2] if input_kind == 'list' else ms.Tensor(np.ones((2, 3),
                                                                 np.float32 if input_kind == 'float' else np.complex64))
    with pytest.raises(TypeError):
        math._check_fftwithsize_input(value, shape, dim, norm, name)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('left,right,expected', [((2, 3, 4), (1, 4, 5), (2,)),
                                               ((4, 5), (2, 3, 5, 6), (2, 3)),
                                               ((2, 1, 3, 4), (1, 5, 4, 6), (2, 5))])
def test_matmul_batch_shape_inference_broadcasts_only_leading_axes(left, right, expected):
    """
    Feature: Matmul batch shape inference broadcasts only leading axes.
    Description: Matrix dimensions must not participate in batch broadcasting.
    Expectation: Matrix dimensions must not participate in batch broadcasting.
    """
    assert math._check_matmul_shapes(left, right, 'matmul') == expected


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_matmul_batch_shape_inference_rejects_incompatible_batch_sizes():
    """
    Feature: Matmul batch shape inference rejects incompatible batch sizes.
    Description: Conflicting non-singleton batch dimensions cannot be silently selected.
    Expectation: Conflicting non-singleton batch dimensions cannot be silently selected.
    """
    with pytest.raises(ValueError):
        math._check_matmul_shapes((2, 3, 4), (3, 4, 5), 'matmul')


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('shape1,shape2,axes', [((2, 3), (2, 3), None),
                                           ((2, 3, 4), (2, 4, 5), (2, 1)),
                                           ((2, 3), (2, 3, 4), (1, 1)),
                                           ((2, 3, 4), (2, 4), (2, 1))])
def test_batch_dot_matches_independent_numpy_contractions(shape1, shape2, axes):
    """
    Feature: Batch dot matches independent numpy contractions.
    Description: Rank promotion and post-contraction squeezing preserve each batch's result.
    Expectation: Rank promotion and post-contraction squeezing preserve each batch's result.
    """
    left = np.arange(np.prod(shape1), dtype=np.float32).reshape(shape1) / 5
    right = np.arange(np.prod(shape2), dtype=np.float32).reshape(shape2) / 7
    result = math.batch_dot(ms.Tensor(left), ms.Tensor(right), axes)
    contraction = (1, 1) if axes is None else axes
    expected = np.stack([np.tensordot(a, b, axes=(contraction[0] - 1, contraction[1] - 1))
                         for a, b in zip(left, right)])
    if expected.ndim == 1:
        expected = expected[:, None]
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=1e-5, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('function', ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn'])
@pytest.mark.parametrize('norm', [None, 'forward', 'ortho'])
def test_fourier_transforms_match_numpy(function, norm):
    """
    Feature: Fourier transforms match numpy.
    Description: Normalization and forward/inverse transforms must preserve complex phases.
    Expectation: Normalization and forward/inverse transforms must preserve complex phases.
    """
    data = (np.arange(6, dtype=np.float32).reshape(2, 3) / 3 +
            1j * np.array([[1, -1, 2], [0, 3, -2]], np.float32)).astype(np.complex64)
    expected = getattr(np.fft, function)(data, norm=norm)
    result = getattr(math, function)(ms.Tensor(data), norm=norm)
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=2e-5, atol=2e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('shape,dimension', [((2, 5), (0, 1)), ((3, 2), (1, 0)), ((2, 2), (0, 1))])
def test_fftn_resizes_and_restores_noncanonical_axes(shape, dimension):
    """
    Feature: Fftn resizes and restores noncanonical axes.
    Description: Padding, truncation and axis permutation must return the requested spatial layout.
    Expectation: Padding, truncation and axis permutation must return the requested spatial layout.
    """
    data = np.arange(6, dtype=np.float32).reshape(2, 3).astype(np.complex64)
    result = math.fftn(ms.Tensor(data), s=shape, dim=dimension, norm='ortho')
    np.testing.assert_allclose(result.asnumpy(), np.fft.fftn(data, s=shape, axes=dimension, norm='ortho'),
                               rtol=2e-5, atol=2e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('order', [None, 0, 1, 2, 3, -1, -2, float('inf'), float('-inf')])
@pytest.mark.parametrize('keepdims', [False, True])
def test_vector_norm_orders_match_numpy(order, keepdims):
    """
    Feature: Vector norm orders match numpy.
    Description: Finite and infinite vector norm orders must reduce only the selected axis.
    Expectation: Finite and infinite vector norm orders must reduce only the selected axis.
    """
    data = np.array([[1., -2., 3.], [-4., 5., 6.]], np.float32)
    result = math.vector_norm(ms.Tensor(data), ord=2 if order is None else order, axis=-1, keepdims=keepdims)
    expected = np.linalg.norm(data, ord=order, axis=-1, keepdims=keepdims)
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=2e-5, atol=2e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('order', ['fro', 'nuc', 1, -1, 2, -2, float('inf'), float('-inf')])
def test_matrix_norm_orders_match_numpy(order):
    """
    Feature: Matrix norm orders match numpy.
    Description: Matrix norms must distinguish spectral, nuclear, row and column reductions.
    Expectation: Matrix norms must distinguish spectral, nuclear, row and column reductions.
    """
    data = np.array([[1., -2., 3.], [-4., 5., 6.]], np.float32)
    result = math.matrix_norm(ms.Tensor(data), ord=order, axis=(0, 1), keepdims=True)
    np.testing.assert_allclose(result.asnumpy(), np.linalg.norm(data, ord=order, axis=(0, 1), keepdims=True),
                               rtol=2e-5, atol=2e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dimension', [0, 1, -1])
@pytest.mark.parametrize('use_coordinates', [False, True])
def test_trapezoidal_integration_matches_numpy(dimension, use_coordinates):
    """
    Feature: Trapezoidal integration matches numpy.
    Description: Both constant spacing and explicit uneven coordinates must use the chosen axis.
    Expectation: Both constant spacing and explicit uneven coordinates must use the chosen axis.
    """
    data = np.arange(12, dtype=np.float32).reshape(3, 4) / 3
    if use_coordinates:
        coordinates = np.arange(data.shape[dimension], dtype=np.float32) ** 2
        result = math.trapz(ms.Tensor(data), ms.Tensor(coordinates), dim=dimension)
        expected = (np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(data, coordinates, axis=dimension)
    else:
        result = math.trapz(ms.Tensor(data), dx=0.5, dim=dimension)
        expected = (np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(data, dx=0.5, axis=dimension)
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=1e-5, atol=1e-6)
