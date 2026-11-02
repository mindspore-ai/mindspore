# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Remaining math and Tensor wrappers: independent CPU oracles and native-boundary protocols."""
import inspect
import itertools
import math as pymath

import numpy as np
import pytest
import mindspore as ms
from mindspore.ops.function import math_func as math
from mindspore.ops import tensor_method as tm
from mindspore.ops import functional_overload as overload


@pytest.fixture(scope='module', autouse=True)
def cpu_context():
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    yield
    ms.set_context(**previous)


def tensor(value, dtype=np.float32):
    return ms.Tensor(np.asarray(value, dtype=dtype))


def close(actual, expected, rtol=2e-5, atol=2e-6):
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for item, oracle in zip(actual, expected):
            close(item, oracle, rtol, atol)
    else:
        assert isinstance(actual, (ms.Tensor, math.Tensor_))
        np.testing.assert_allclose(actual.asnumpy(), expected, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize('module,name,oracle', [
    ('math', 'negative', np.negative), ('math', 'positive', np.positive),
    ('math', 'exp2', np.exp2), ('math', 'i0', np.i0),
    ('math', 'arcsin', np.arcsin), ('math', 'arccos', np.arccos),
    ('math', 'arctan', np.arctan), ('math', 'arcsinh', np.arcsinh),
    ('math', 'arctanh', np.arctanh), ('math', 'log2', np.log2),
    ('math', 'deg2rad', np.deg2rad), ('math', 'rad2deg', np.rad2deg),
    ('math', 'tanhshrink', lambda x: x - np.tanh(x)),
    ('tensor', 'tensor_abs', np.abs), ('tensor', 'tensor_ceil', np.ceil),
    ('tensor', 'tensor_cos', np.cos), ('tensor', 'tensor_exp', np.exp),
    ('tensor', 'tensor_floor', np.floor), ('tensor', 'tensor_isfinite', np.isfinite),
    ('tensor', 'tensor_log', np.log), ('tensor', 'tensor_neg', np.negative),
    ('tensor', 'tensor_reciprocal', np.reciprocal),
    ('tensor', 'tensor_round', np.round), ('tensor', 'tensor_rsqrt', lambda x: 1 / np.sqrt(x)),
    ('tensor', 'tensor_sigmoid', lambda x: 1 / (1 + np.exp(-x))),
    ('tensor', 'tensor_sin', np.sin), ('tensor', 'tensor_sqrt', np.sqrt),
    ('tensor', 'tensor_square', np.square), ('tensor', 'tensor_tanh', np.tanh),
    ('tensor', 'tensor_trunc', np.trunc), ('tensor', 'tensor_tan', np.tan),
    ('tensor', 'tensor_atanh', np.arctanh), ('tensor', 'tensor_cosh', np.cosh),
    ('tensor', 'tensor_sinh', np.sinh), ('tensor', 'tensor_sinc', np.sinc),
    ('tensor', 'tensor_expm1', np.expm1), ('tensor', 'tensor_log1p', np.log1p),
    ('tensor', 'tensor_erf', lambda x: np.vectorize(pymath.erf)(x)),
    ('tensor', 'tensor_erfc', lambda x: np.vectorize(pymath.erfc)(x)),
])
def test_unary_numeric_wrappers(module, name, oracle):
    data = np.array([.25, .5, .75], np.float32)
    close(getattr(math if module == 'math' else tm, name)(tensor(data)), oracle(data))


@pytest.mark.parametrize('module,name,oracle', [
    ('math', 'subtract', np.subtract), ('math', 'divide', np.divide),
    ('math', 'arctan2', np.arctan2), ('math', 'truncate_div', lambda a, b: np.trunc(a / b)),
    ('math', 'truncate_mod', np.fmod), ('math', 'copysign', np.copysign),
    ('math', 'logaddexp', np.logaddexp), ('math', 'logaddexp2', np.logaddexp2),
    ('math', 'xdivy', lambda a, b: a / b),
    ('tensor', 'tensor_arctan2', np.arctan2), ('tensor', 'tensor_greater', np.greater),
    ('tensor', 'deprecated_tensor_greater_equal', np.greater_equal),
    ('tensor', 'tensor_le', np.less_equal), ('tensor', 'tensor_maximum', np.maximum),
    ('tensor', 'tensor_minimum', np.minimum), ('tensor', 'deprecated_tensor_mul', np.multiply),
    ('tensor', 'tensor_ne', np.not_equal), ('tensor', 'tensor_not_equal', np.not_equal),
    ('tensor', 'deprecated_tensor_pow', np.power), ('tensor', 'deprecated_tensor_sub', np.subtract),
    ('tensor', 'deprecated_tensor_sub_', np.subtract), ('tensor', 'deprecated_tensor_fmod', np.fmod),
    ('tensor', 'tensor_true_divide', np.divide),
])
def test_binary_numeric_wrappers(module, name, oracle):
    left, right = np.array([1., 2., 4.], np.float32), np.array([2., 3., 2.], np.float32)
    close(getattr(math if module == 'math' else tm, name)(tensor(left), tensor(right)), oracle(left, right))


@pytest.mark.parametrize('name,oracle', [('tensor_logical_and', np.logical_and),
                                      ('tensor_logical_or', np.logical_or),
                                      ('tensor_logical_xor', np.logical_xor)])
def test_logical_wrappers(name, oracle):
    left, right = tensor([True, False, True], np.bool_), tensor([False, False, True], np.bool_)
    close(getattr(tm, name)(left, right), oracle(left.asnumpy(), right.asnumpy()))
    close(tm.tensor_logical_not(left), [False, True, False])


@pytest.mark.parametrize('dtype', [np.float16, np.float32])
@pytest.mark.parametrize('name', ['exp2', 'deg2rad', 'rad2deg'])
def test_float16_conversion_branches(dtype, name):
    values = np.array([.25, .5], dtype)
    close(getattr(math, name)(tensor(values, dtype)), getattr(np, name)(values), rtol=.002, atol=.001)


@pytest.mark.parametrize('weights,minlength', [(None, 0), (None, 7), ([.5, 1., 2., 3.], 0)])
def test_bincount_weighted_and_padded(weights, minlength):
    values = np.array([0, 1, 1, 3], np.int32)
    close(math.bincount(tensor(values, np.int32), None if weights is None else tensor(weights), minlength),
          np.bincount(values, weights=weights, minlength=minlength))
    close(tm.tensor_bincount(tensor(values, np.int32), minlength=minlength), np.bincount(values, minlength=minlength))


@pytest.mark.parametrize('values,weights,minlength,error', [
    ([1, 2], None, 0, TypeError), (tensor([1, 2], np.int32), [1., 2.], 0, TypeError),
    (tensor([1, 2], np.int32), None, True, TypeError),
    (tensor([[1, 2]], np.int32), None, 0, ValueError),
    (tensor([1, 2], np.int32), None, -1, ValueError),
    (tensor([1, 2], np.int32), tensor([1.]), 0, ValueError),
])
def test_bincount_validation(values, weights, minlength, error):
    with pytest.raises(error):
        math.bincount(values, weights, minlength)


def test_bincount_empty_and_sign_and_scalar_math():
    close(math.bincount(tensor([], np.int32)), [])
    close(math.signbit(tensor([-2, 0, 3])), [True, False, False])
    close(math.sgn(tensor([-2., 0., 3.])), [-1., 0., 1.])
    values = np.array([3+4j, 0j, -2j], np.complex64)
    close(math.sgn(tensor(values, np.complex64)), [.6+.8j, 0j, -1j])
    close(math.argmin(tensor(5)), 0)
    close(math.t(tensor([1, 2])), [1, 2])
    close(math.xlogy(2., tensor([1., 2.])), [0., 2 * np.log(2)])
    close(math.xlogy(tensor([False, True], np.bool_), True), [0., 0.])
    close(math.float_power(tensor([2, 4], np.int32), -1), [.5, .25])
    close(math.float_power(tensor([1j, 2j], np.complex64), 2), [-1+0j, -4+0j])
    close(math.float_power(2, tensor([1., 2.])), [2., 4.])


@pytest.mark.parametrize('name,args,kwargs,error', [
    ('positive', ([1],), {}, TypeError), ('signbit', ([1],), {}, TypeError),
    ('sgn', ([1],), {}, TypeError), ('div', (tensor([1.]), 2), {'rounding_mode':'ceil'}, ValueError),
    ('float_power', (2, 3), {}, TypeError), ('float_power', ([2], tensor([1.])), {}, TypeError),
    ('float_power', (tensor([1.]), [2]), {}, TypeError), ('fmod', (2, 3), {}, TypeError),
    ('bucketize', ([1], [1., 2.]), {}, TypeError),
    ('bucketize', (tensor([True], np.bool_), [1., 2.]), {}, TypeError),
    ('deg2rad', ([1],), {}, TypeError), ('rad2deg', ([1],), {}, TypeError),
    ('view_as_real', (tensor([1.]),), {}, TypeError),
    ('vecdot', ([1], tensor([1.])), {}, TypeError),
    ('vecdot', (tensor([1.]), tensor([1.])), {'axis':1.2}, TypeError),
    ('vecdot', (tensor([1.]), tensor([1.])), {'axis':2}, ValueError),
])
def test_math_rejects_invalid_public_arguments(name, args, kwargs, error):
    with pytest.raises(error):
        getattr(math, name)(*args, **kwargs)


@pytest.mark.parametrize('frequency,analytic,correction,dtype', [
    (None, None, 1, np.float64), ([1, 2, 1], None, 1, np.float32),
    (None, [.5, 1., 2.], 1, np.float32), ([1, 2, 1], [.5, 1., 2.], 1, np.float32),
    ([1, 2, 1], [.5, 1., 2.], 0, np.float16),
])
def test_covariance_weights_match_numpy(frequency, analytic, correction, dtype):
    values = np.array([[0., 5., 7.], [3., 5., 0.]], dtype)
    close(math.cov(tensor(values, dtype), correction=correction,
                   fweights=None if frequency is None else tensor(frequency, np.int64),
                   aweights=None if analytic is None else tensor(analytic)),
          np.cov(values, ddof=correction, fweights=frequency, aweights=analytic))


@pytest.mark.parametrize('kind,error', [('rank', ValueError), ('bool', TypeError),
                                      ('weight_rank', ValueError), ('weight_dtype', TypeError),
                                      ('weight_length', ValueError)])
def test_covariance_rejects_invalid_shape_and_weights(kind, error):
    values = tensor(np.ones((2, 3, 4)) if kind == 'rank' else [1., 2., 3.],
                    np.bool_ if kind == 'bool' else np.float32)
    kwargs = {}
    if kind.startswith('weight_'):
        kwargs['fweights'] = tensor([[1, 2, 3]] if kind == 'weight_rank' else [1, 2] if kind == 'weight_length'
                                    else [1, 2, 3], np.float32 if kind == 'weight_dtype' else np.int64)
    with pytest.raises(error):
        math.cov(values, **kwargs)


@pytest.mark.parametrize('kind', ['block', 'dstack', 'cartesian', 'vander', 'row_stack', 'view_real'])
def test_math_shape_construction(kind):
    if kind == 'block':
        close(math.block_diag(tensor(2), tensor([3, 4]), tensor([[5], [6]])),
              [[2, 0, 0, 0], [0, 3, 4, 0], [0, 0, 0, 5], [0, 0, 0, 6]])
    elif kind == 'dstack':
        close(math.dstack([tensor([1, 2]), tensor([3, 4])]), np.dstack(([1, 2], [3, 4])))
    elif kind == 'cartesian':
        close(math.cartesian_prod(tensor([1, 2]), tensor([3, 4]), tensor([5])), list(itertools.product([1, 2], [3, 4], [5])))
    elif kind == 'vander':
        close(math.vander(tensor([1, 2, 3]), 4), np.vander([1, 2, 3], 4))
        close(math.vander(tensor([1, 2, 3])), np.vander([1, 2, 3]))
    elif kind == 'row_stack':
        close(math.row_stack([tensor([1, 2]), tensor([3, 4])]), [[1, 2], [3, 4]])
    else:
        close(math.view_as_real(tensor([1+2j, 3-4j], np.complex64)), [[1, 2], [3, -4]])


@pytest.mark.parametrize('n,prepend,append', [(1, [[0], [1]], None), (1, None, [[8], [9]]), (1, [[0], [1]], [[8], [9]]), (1, None, None)])
def test_difference_with_scalar_endpoints(n, prepend, append):
    data = np.array([[1., 3., 6.], [2., 4., 7.]], np.float32)
    kwargs = {key: value for key, value in [('prepend', prepend), ('append', append)] if value is not None}
    tensor_kwargs = {key: tensor(value) for key, value in kwargs.items()}
    close(math.diff(tensor(data), n=n, axis=1, **tensor_kwargs), np.diff(data, n=n, axis=1, **kwargs))


@pytest.mark.parametrize('axis', [0, 1, -1])
@pytest.mark.parametrize('coordinates', ['none', 'vector', 'matrix'])
def test_trapezoidal_integration(axis, coordinates):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = None if coordinates == 'none' else np.arange(data.shape[axis], dtype=np.float32) ** 2
    if coordinates == 'matrix':
        x = np.broadcast_to(x[:, None] if axis == 0 else x[None, :], data.shape).copy()
    oracle = getattr(np, 'trapezoid', np.trapz)
    close(math.trapz(tensor(data), None if x is None else tensor(x), dx=.5, dim=axis),
          oracle(data, x=x, dx=.5, axis=axis))


@pytest.mark.parametrize('axis', [0, 1])
def test_empty_trapezoid_removes_only_integrated_axis(axis):
    shape = (0, 3) if axis == 0 else (2, 0)
    data = tensor(np.empty(shape))
    close(math.trapz(data, dim=axis), np.zeros(shape[1-axis]))
    close(math.trapz(data, tensor([]), dim=axis), np.zeros(shape[1-axis]))


@pytest.mark.parametrize('y,x,dx,dim,error', [
    ([1, 2], None, 1., -1, TypeError), (tensor([1, 2]), None, 1, -1, TypeError),
    (tensor([1, 2]), None, 1., .5, TypeError), (tensor([1, 2]), None, 1., 2, ValueError),
    (tensor([1, 2]), [1, 2], 1., 0, TypeError), (tensor([1, 2]), tensor([1]), 1., 0, RuntimeError),
])
def test_trapezoid_rejects_invalid_geometry(y, x, dx, dim, error):
    with pytest.raises(error):
        math.trapz(y, x, dx=dx, dim=dim)


@pytest.mark.parametrize('name', ['amin', 'amax'])
def test_masked_extrema_match_numpy(name):
    values = np.array([[2., 5., 1.], [3., 7., 4.]], np.float32)
    mask = np.array([[True, False, True]])
    initial = 0 if name == 'amax' else 10
    close(getattr(math, name)(tensor(values), axis=1, keepdims=True, initial=initial,
                              where=tensor(mask, np.bool_)),
          getattr(np, name)(values, axis=1, keepdims=True, initial=initial, where=mask))
    with pytest.raises(ValueError, match='initial'):
        getattr(math, name)(tensor(values), where=tensor(mask, np.bool_))


@pytest.mark.parametrize('name', ['var', 'std', 'var_mean', 'std_mean'])
@pytest.mark.parametrize('axis,ddof,keepdims', [(None, 0, False), (1, 1, True)])
def test_variance_reductions_match_numpy(name, axis, ddof, keepdims):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    expected = getattr(np, 'std' if name.startswith('std') else 'var')(data, axis=axis, ddof=ddof, keepdims=keepdims)
    if name.endswith('_mean'):
        expected = (expected, np.mean(data, axis=axis, keepdims=keepdims))
    close(getattr(math, name)(tensor(data), axis=axis, ddof=ddof, keepdims=keepdims), expected)


@pytest.mark.parametrize('name', ['quantile', 'nanquantile'])
@pytest.mark.parametrize('q', [.5, [.25, .75]])
def test_quantile_native_boundary_protocol(monkeypatch, name, q):
    """Protocol only: Quantile has no CPU kernel; check its constructor and inputs."""
    values = np.array([[1., 4., 2.], [3., 6., 5.]], np.float32)
    if name == 'nanquantile':
        values[0, 0] = np.nan
    x, query = tensor(values), q if np.isscalar(q) else tensor(q)
    output = tensor(getattr(np, name)(values, q, axis=1, keepdims=True))
    calls = []
    def primitive_call(primitive, input_x, input_q):
        assert primitive.attrs['dim'] == 1 and primitive.attrs['keep_dims'] is True
        assert primitive.attrs.get('ignore_nan', False) == (name == 'nanquantile')
        assert input_x is x and input_q is query
        calls.append((input_x, input_q))
        return output
    monkeypatch.setattr(math.Quantile, '__call__', primitive_call)
    assert getattr(math, name)(x, query, axis=1, keepdims=True) is output
    assert len(calls) == 1
    close(output, getattr(np, name)(values, q, axis=1, keepdims=True))


@pytest.mark.parametrize('name', ['tril_indices', 'triu_indices'])
@pytest.mark.parametrize('offset', [-1, 0, 2])
def test_triangular_coordinate_construction(name, offset):
    oracle = np.tril_indices if name == 'tril_indices' else np.triu_indices
    close(getattr(math, name)(3, 4, offset, dtype=ms.int32), np.array(oracle(3, offset, 4)))


@pytest.mark.parametrize('p', [0, 1, 2, 3])
def test_distance_and_complex_vector_dot(p):
    left, right = np.array([1., 3., 5.]), np.array([0., 3., 1.])
    close(math.dist(tensor(left), tensor(right), p), np.linalg.norm(left-right, ord=p))
    a = np.array([[1+2j, 3-1j]], np.complex64)
    b = np.array([2+1j, 1-2j], np.complex64)
    close(math.vecdot(tensor(a, np.complex64), tensor(b, np.complex64)), np.sum(a.conj()*b, axis=-1))


@pytest.mark.parametrize('name', ['outer', 'mv', 'addmv', 'addmm', 'addbmm', 'baddbmm', 'inner'])
def test_linear_algebra_composition(name):
    a, b = np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(6, dtype=np.float32).reshape(3, 2)
    if name == 'outer':
        close(math.outer(tensor([1, 2]), tensor([3, 4, 5])), np.outer([1, 2], [3, 4, 5]))
    elif name == 'mv':
        close(math.mv(tensor(a), tensor([1, 2, 3])), a @ [1, 2, 3])
    elif name == 'addmv':
        close(math.addmv(tensor([1, 2]), tensor(a), tensor([1, 2, 3]), beta=2, alpha=3),
              2*np.array([1, 2])+3*(a @ [1, 2, 3]))
    elif name == 'addmm':
        close(math.addmm(tensor(np.ones((2, 2))), tensor(a), tensor(b), beta=2, alpha=3), 2+3*(a@b))
    elif name == 'inner':
        close(math.inner(tensor(a), tensor(a)), np.inner(a, a))
    else:
        left, right = np.stack([a, a+1]), np.stack([b, b+1])
        base = np.ones((2, 2)) if name == 'addbmm' else np.ones((2, 2, 2))
        oracle = left @ right
        if name == 'addbmm':
            oracle = oracle.sum(0)
        close(getattr(math, name)(tensor(base), tensor(left), tensor(right), beta=2, alpha=3), 2*base+3*oracle)


@pytest.mark.parametrize('name', ['matrix_determinant', 'log_matrix_determinant', 'slogdet', 'matrix_power'])
def test_matrix_determinant_and_powers(name):
    data = np.array([[3., 1.], [1., 2.]], np.float32)
    expected = np.linalg.det(data) if name == 'matrix_determinant' else np.linalg.slogdet(data)
    args = (tensor(data),)
    if name == 'matrix_power':
        args += (-2,)
        expected = np.linalg.matrix_power(data, -2)
    close(getattr(math, name)(*args), expected)


@pytest.mark.parametrize('name,axis,keep', [('tensor_maxdim', 1, False), ('tensor_mindim', 0, True)])
def test_tensor_dimension_extrema_value_index_order(name, axis, keep):
    values = np.array([[3., 1., 2.], [4., 6., 5.]], np.float32)
    is_max = name == 'tensor_maxdim'
    result = getattr(tm, name)(tensor(values), axis, keep)
    expected_value = (np.max if is_max else np.min)(values, axis=axis, keepdims=keep)
    expected_index = (np.argmax if is_max else np.argmin)(values, axis=axis)
    if keep:
        expected_index = np.expand_dims(expected_index, axis)
    close(result, (expected_value, expected_index))


@pytest.mark.parametrize('name', ['tensor_any', 'reduce_tensor_any'])
@pytest.mark.parametrize('axis', [None, 1])
def test_tensor_any_default_and_explicit_axes(name, axis):
    data = np.array([[0, 0], [1, 0]], np.bool_)
    close(getattr(tm, name)(tensor(data, np.bool_), axis, True), np.any(data, axis=axis, keepdims=True))


@pytest.mark.parametrize('name', ['tensor_max', 'tensor_min', 'tensor_prod', 'tensor_sum_ext',
                                'deprecated_tensor_nansum', 'deprecated_tensor_count_nonzero'])
def test_tensor_reduction_wrappers(name):
    values = np.array([[1., 0., 3.], [4., 5., 6.]], np.float32)
    oracle = {'tensor_max':np.max, 'tensor_min':np.min, 'tensor_prod':np.prod, 'tensor_sum_ext':np.sum,
              'deprecated_tensor_nansum':np.nansum, 'deprecated_tensor_count_nonzero':np.count_nonzero}[name]
    close(getattr(tm, name)(tensor(values)), oracle(values))


@pytest.mark.parametrize('name', ['tensor_to', 'tensor_expand_as', 'tensor_chunk', 'deprecated_tensor_argsort',
                                'tensor_masked_select', 'tensor_tile', 'deprecated_tensor_tile',
                                'deprecated_tensor_permute', 'deprecated_tensor_t', 'deprecated_tensor_unbind',
                                'deprecated_tensor_view_as', 'tensor_triu',
                                'deprecated_tensor_unsqueeze', 'tensor_broadcast_to'])
def test_tensor_shape_and_selection_wrappers(name):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = tensor(data)
    cases = {
        'tensor_to': ((x, ms.float64), data.astype(np.float64)),
        'tensor_expand_as': ((tensor([1., 2., 3.]), x), [[1, 2, 3], [1, 2, 3]]),
        'tensor_chunk': ((x, 2, 0), np.split(data, 2, axis=0)),
        'deprecated_tensor_argsort': ((x, 1, True), np.argsort(-data, axis=1)),
        'tensor_masked_select': ((x, tensor(data > 2, np.bool_)), data[data > 2]),
        'tensor_tile': ((x, (2, 1)), np.tile(data, (2, 1))),
        'deprecated_tensor_tile': ((x, (1, 2)), np.tile(data, (1, 2))),
        'deprecated_tensor_permute': ((x, 1, 0), data.T),
        'deprecated_tensor_t': ((x,), data.T),
        'deprecated_tensor_unbind': ((x, 0), list(data)),
        'deprecated_tensor_view_as': ((x, tensor(np.zeros((3, 2)))), data.reshape(3, 2)),
        'tensor_triu': ((x, 1), np.triu(data, 1)),
        'deprecated_tensor_unsqueeze': ((x, 1), np.expand_dims(data, 1)),
        'tensor_broadcast_to': ((tensor([1., 2., 3.]), (2, 3)), [[1, 2, 3], [1, 2, 3]])}
    args, expected = cases[name]
    close(getattr(tm, name)(*args), expected)


@pytest.mark.parametrize('name', ['tensor_matmul', 'deprecated_tensor_matmul', 'deprecated_tensor_mm',
                                'deprecated_tensor_dot', 'deprecated_tensor_inverse'])
def test_tensor_matrix_wrappers(name):
    a = np.array([[3., 1.], [1., 2.]], np.float32)
    b = np.array([[1., 2.], [3., 4.]], np.float32)
    args, expected = (tensor(a), tensor(b)), a @ b
    if 'inverse' in name:
        args, expected = (tensor(a),), np.linalg.inv(a)
    close(getattr(tm, name)(*args), expected)


@pytest.mark.parametrize('name', ['tensor_div_deal_sequence', 'deprecated_tensor_sub', 'deprecated_tensor_sub_'])
@pytest.mark.parametrize('sequence_side', [0, 1])
def test_tensor_arithmetic_converts_sequence_operand(name, sequence_side):
    args = [tensor([4., 6.]), tensor([2., 3.])]
    args[sequence_side] = args[sequence_side].asnumpy().tolist()
    close(getattr(tm, name)(*args), [2., 2.] if name == 'tensor_div_deal_sequence' else [2., 3.])


@pytest.mark.parametrize('name', ['tensor_isneginf', 'deprecated_tensor_isneginf'])
def test_tensor_negative_infinity(name):
    close(getattr(tm, name)(tensor([-np.inf, -1., np.inf, np.nan])), [True, False, False, False])


@pytest.mark.parametrize('name', ['tensor_diag', 'deprecated_tensor_diag'])
def test_tensor_diagonal_wrappers(name):
    close(getattr(tm, name)(tensor([1., 2.])), [[1., 0.], [0., 2.]])
    if name == 'tensor_diag':
        with pytest.raises(ValueError, match='diagonal'):
            tm.tensor_diag(tensor([1., 2.]), 1)


@pytest.mark.parametrize('name', ['deprecated_tensor_var'])
def test_tensor_variance_wrappers(name):
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    close(getattr(tm, name)(tensor(values), 1), np.var(values, axis=1, ddof=0 if name.startswith('deprecated') else 1))
    if name.startswith('deprecated'):
        for kwargs in ({'ddof': .5}, {'keepdims': 1}):
            with pytest.raises(TypeError):
                tm.deprecated_tensor_var(tensor(values), **kwargs)


@pytest.mark.parametrize('mode', [pytest.param('clip', marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason='Tensor.take clip incorrectly wraps negative indices; NumPy clip requires index 0')), 'wrap'])
@pytest.mark.parametrize('axis', [None, 1])
def test_deprecated_take_matches_numpy(mode, axis):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    indices = np.array([-1, 0, 4], np.int32)
    close(tm.deprecated_tensor_take(tensor(data), tensor(indices, np.int32), axis, mode),
          np.take(data, indices, axis=axis, mode=mode))


@pytest.mark.parametrize('name', [
    'tensor_repeat', 'tensor_scatter_add_empty', 'tensor_inplace_fill_scalar_empty',
    'tensor_inplace_fill_tensor_empty', 'tensor_inplace_fill_diagonal',
    'tensor_inplace_masked_fill_scalar_empty', 'tensor_inplace_masked_fill_tensor_empty',
    'tensor_copy_', 'tensor_empty_like', 'tensor_new_empty', 'tensor_view_dtype',
    'tensor_put_', 'deprecated_einsum', 'tensor_index_fill_', 'tensor_kthvalue',
    'tensor_index_copy_', 'tensor_sub_empty_', 'tensor_new_full', 'tensor_div_empty_',
    'all_gather_matmul', 'conv1d', 'conv2d', 'conv3d', 'matmul_reduce_scatter',
    'tensor_bernoulli_', 'tensor_quant_matmul', 'tensor_dynamic_mx_quant',
    'tensor_dynamic_mx_quant_with_dual_axis', 'tensor_grouped_matmul_v5',
    'tensor_grouped_dynamic_mx_quant', 'tensor_index', 'tensor_gmm', 'raise_func',
    'tensor_nsa_select_attention', 'tensor_lightning_indexer', '_tensor_nsa_compress',
    '_tensor_nsa_compress_attention', '_tensor_sparse_flash_attention',
])
def test_cpu_placeholder_contracts(name):
    """These Python methods explicitly reject unsupported device dispatch or internal entry."""
    function = getattr(tm, name)
    args, kwargs = [], {}
    for param in inspect.signature(function).parameters.values():
        if param.default is not inspect.Parameter.empty or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        value = ms.float32 if param.name in ('dtype', 'dst_type') else tensor([1.])
        if param.kind == param.KEYWORD_ONLY:
            kwargs[param.name] = value
        else:
            args.append(value)
    expected = RuntimeError if name in ('tensor_repeat', 'tensor_put_', 'tensor_bernoulli_',
                                        '_tensor_nsa_compress', '_tensor_nsa_compress_attention',
                                        '_tensor_sparse_flash_attention') else ValueError if name in (
                                            'tensor_scatter_add_empty', 'tensor_inplace_fill_scalar_empty',
                                            'tensor_inplace_fill_tensor_empty', 'tensor_inplace_fill_diagonal',
                                            'tensor_inplace_masked_fill_scalar_empty', 'tensor_inplace_masked_fill_tensor_empty',
                                            'tensor_copy_', 'tensor_kthvalue', 'tensor_sub_empty_', 'tensor_div_empty_') else NotImplementedError
    with pytest.raises(expected, match='supported|support|should not|no branch|implemented|come here'):
        function(*args, **kwargs)


@pytest.mark.parametrize('name,boundary', [
    ('grouped_dynamic_mx_quant', '_grouped_dynamic_mx_quant_instance'),
    ('__truediv__', '_div_instance'), ('xlogy', '_xlogy_instance'),
    ('conv2d', '_conv2d_instance'), ('sparse_flash_attention', '_sparse_flash_attention_instance'),
    ('eq', '_eq_instance'), ('all_gather_matmul', '_all_gather_matmul_instance'),
])
def test_overload_native_boundary_preserves_call_protocol(monkeypatch, name, boundary):
    """Use native CPU arithmetic overloads and exact forwarding protocols for accelerator overloads."""
    left, right = tensor([[1., 2.], [3., 4.]]), tensor([[2., 1.], [4., 3.]])
    index = tensor([0, 2], np.int32)
    query=tensor(np.ones((1,1,1,512)))
    rope=tensor(np.zeros((1,1,1,64)))
    sparse_indices=tensor(np.zeros((1,1,1,1)),np.int32)
    image=tensor(np.arange(9).reshape(1,1,3,3))
    kernel=tensor(np.ones((1,1,2,2)))
    half_left,half_right=tensor(left.asnumpy(),np.float16),tensor(right.asnumpy(),np.float16)
    cases = {
        'grouped_dynamic_mx_quant': ((half_left, index), dict(round_mode='rint', dst_type=23,
                                                       blocksize=32, scale_alg=0, dst_type_max=0.0)),
        '__truediv__': ((left, right), {}), 'xlogy': ((left, right), {}), 'eq': ((left, right), {}),
        'conv2d': ((image, kernel), dict(bias=None, stride=(1, 2), padding=0, dilation=1, groups=1)),
        'sparse_flash_attention': ((query, query, query, sparse_indices, .125),
            dict(block_table=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                 query_rope=rope, key_rope=rope, sparse_block_size=1, layout_query='BSND', layout_kv='BSND',
                 sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807,
                 attention_mode=2, return_softmax_lse=True)),
        'all_gather_matmul': ((half_left, half_right, 'test_group', 2),
            dict(bias=None, gather_index=0, gather_output=True, comm_turn=0, trans_input=False, trans_x2=True))}
    args, kwargs = cases[name]
    if name=='__truediv__':
        expected=left.asnumpy()/right.asnumpy();output=tensor(expected)
    elif name=='xlogy':
        expected=left.asnumpy()*np.log(right.asnumpy());output=tensor(expected)
    elif name=='eq':
        expected=left.asnumpy()==right.asnumpy();output=tensor(expected,np.bool_)
    elif name=='conv2d':
        expected=np.array([[[[8.],[20.]]]],np.float32);output=tensor(expected)
    elif name=='all_gather_matmul':
        gathered=np.concatenate([left.asnumpy()]*2);expected=gathered@right.asnumpy().T
        output=(tensor(expected,np.float16),tensor(gathered,np.float16))
    elif name=='sparse_flash_attention':
        expected=np.ones((1,1,1,512),np.float32);output=(tensor(expected),tensor([1.]),tensor([1.]))
    else:
        expected=left.asnumpy();output=(tensor(expected,np.float16),tensor([127],np.uint8))
    if name in ('__truediv__','xlogy','eq'):
        close(getattr(overload,name)(*args,**kwargs),expected)
        return
    calls = []
    def native_boundary(*received, **options):
        calls.append((received, options))
        assert len(received) == len(args) and options.keys() == kwargs.keys()
        for actual, original in zip(received, args):
            assert actual is original
        for key, original in kwargs.items():
            assert options[key] is original
        return output
    monkeypatch.setattr(overload, boundary, native_boundary)
    assert getattr(overload, name)(*args, **kwargs) is output
    assert len(calls) == 1
    close(output[0] if isinstance(output, tuple) else output, expected)


@pytest.mark.parametrize('name,boundary,args,kwargs,expected', [
    ('tensor_clone', 'clone', (tensor([1., 2.]),), {}, [1., 2.]),
    ('tensor_logaddexp2', 'logaddexp2', (tensor([1., 2.]), tensor([2., 3.])), {}, np.logaddexp2([1., 2.], [2., 3.])),
    ('tensor_roll', 'roll', (tensor([1., 2., 3.]), 1, 0), {}, [3., 1., 2.]),
    ('tensor_dot', 'dot', (tensor([1., 2.]), tensor([3., 4.])), {}, 11.),
    ('tensor_var', 'var_ext', (tensor([[1., 2., 3.], [2., 4., 6.]]), 1),
     {'correction': 1, 'keepdim': True}, [[1.], [4.]])])
def test_tensor_unavailable_cpu_native_boundary(monkeypatch, name, boundary, args, kwargs, expected):
    """Protocol only for missing CPU PyBoost kernels; real wrapper returns real Tensor."""
    output, calls = tensor(expected), []
    def kernel_boundary(*received, **options):
        calls.append((received, options))
        assert len(received) == len(args)
        for actual, original in zip(received, args):
            assert actual is original
        assert options == kwargs
        return output
    monkeypatch.setattr(tm, boundary, kernel_boundary)
    assert getattr(tm, name)(*args, **kwargs) is output
    assert len(calls) == 1
    close(output, expected)


def test_internal_placeholder_logging(caplog):
    with caplog.at_level('ERROR'):
        assert tm.place_holder() is None
        assert tm.tensor_empty(2, 3, dtype=ms.float32) is None
    assert 'place holder' in caplog.text and 'empty' in caplog.text


@pytest.mark.parametrize('name,args,kwargs,error', [
    ('vander', ([1, 2],), {}, TypeError), ('vander', (tensor([[1, 2]]),), {}, ValueError),
    ('vander', (tensor([1, 2]), 1.5), {}, TypeError), ('vander', (tensor([1, 2]), 0), {}, ValueError),
    ('block_diag', (), {}, RuntimeError), ('block_diag', ([1],), {}, TypeError),
    ('block_diag', (tensor(np.ones((2, 2, 2))),), {}, ValueError),
    ('dstack', (tensor([1, 2]),), {}, TypeError), ('dstack', ([],), {}, TypeError),
    ('dstack', ([1],), {}, TypeError),
    ('diff', ([1, 2],), {}, TypeError), ('diff', (tensor(1),), {}, TypeError),
    ('diff', (tensor([]),), {}, ValueError), ('diff', (tensor([1, 2]),), {'n':0}, RuntimeError),
    ('diff', (tensor([1, 2]),), {'n':2}, RuntimeError),
    ('diff', (tensor([1, 2], np.uint16),), {}, TypeError),
    ('dist', ([1], tensor([1.])), {}, TypeError), ('dist', (tensor([1.]), [1]), {}, TypeError),
    ('dist', (tensor([1.]), tensor([1.])), {'p':float('inf')}, TypeError),
    ('copysign', ([1], tensor([1.])), {}, TypeError),
    ('quantile', (tensor([1.]), .5), {'axis':1.5}, TypeError),
    ('quantile', (tensor([1.]), .5), {'keepdims':1}, TypeError),
    ('nanquantile', (tensor([1.]), .5), {'axis':'a'}, TypeError),
    ('nanquantile', (tensor([1.]), .5), {'keepdims':0}, TypeError),
    ('gumbel_softmax', ([1],), {}, TypeError), ('gumbel_softmax', (tensor(1.),), {}, ValueError),
    ('gumbel_softmax', (tensor([1, 2], np.int32),), {}, TypeError),
    ('gumbel_softmax', (tensor([1., 2.]),), {'tau':1}, TypeError),
    ('gumbel_softmax', (tensor([1., 2.]),), {'tau':0.}, ValueError),
    ('gumbel_softmax', (tensor([1., 2.]),), {'hard':1}, TypeError),
    ('gumbel_softmax', (tensor([1., 2.]),), {'dim':.5}, TypeError),
    ('gumbel_softmax', (tensor([1., 2.]),), {'dim':2}, ValueError),
    ('gumbel_softmax', (tensor([[1., 2.]]),), {'hard':True, 'dim':-2}, ValueError),
    ('stft', (tensor(np.ones((1, 2, 3))), 4), {}, ValueError),
    ('zeta', (2, 3), {}, TypeError),
])
def test_math_remaining_validation_branches(name, args, kwargs, error):
    with pytest.raises(error):
        getattr(math, name)(*args, **kwargs)


@pytest.mark.parametrize('hard,axis', [(False, -2), (False, -1), (True, -1), (True, 0)])
def test_gumbel_softmax_fixed_rng_numerical_transform(monkeypatch, hard, axis):
    """Only uniform RNG is fixed; logarithms, normalization and one-hot operators are real CPU kernels."""
    logits = np.array([[.2, 1.2, -.5], [.7, .3, -.2]], np.float32)
    uniform = np.array([[.2, .5, .8], [.3, .6, .9]], np.float32)
    calls = []
    def fixed_uniform(shape, minimum, maximum):
        assert shape == logits.shape
        close(minimum, 0.)
        close(maximum, 1.)
        calls.append(shape)
        return tensor(uniform)
    monkeypatch.setattr(math.C, 'uniform', fixed_uniform)
    transformed = (logits - np.log(-np.log(uniform))) / .75
    expected = np.exp(transformed - transformed.max(axis=axis, keepdims=True))
    expected /= expected.sum(axis=axis, keepdims=True)
    if hard:
        indices = np.argmax(expected, axis=axis)
        expected = np.zeros_like(expected)
        np.put_along_axis(expected, np.expand_dims(indices, axis), 1., axis=axis)
    close(math.gumbel_softmax(tensor(logits), tau=.75, hard=hard, dim=axis), expected)
    assert calls == [logits.shape]


@pytest.mark.parametrize('data,pivots', [(True, True), (True, False), (False, True), (False, False)])
def test_lu_unpack_flags_preserve_factorization(data, pivots):
    packed = tensor([[2., 3.], [.5, 4.]])
    permutation, lower, upper = math.lu_unpack(packed, tensor([1, 2], np.int32), data, pivots)
    if pivots:
        close(permutation, np.eye(2))
    else:
        assert permutation is None
    if data:
        close(lower, [[1, 0], [.5, 1]])
        close(upper, [[2, 3], [0, 4]])
        close(lower @ upper, [[2., 3.], [1., 5.5]])
    else:
        assert lower is None and upper is None


@pytest.mark.parametrize('steps,tensor_limits', [(1, False), (4, False), (4, True)])
def test_linear_and_logarithmic_spacing(steps, tensor_limits):
    start, end = (tensor(1.), tensor(3.)) if tensor_limits else (1., 3.)
    close(math.linspace(start, end, steps), np.linspace(1., 3., steps))
    close(math.logspace(start, end, steps, base=2), np.logspace(1., 3., steps, base=2))


@pytest.mark.parametrize('order', [0, 1, 2, 3])
def test_polygamma_and_lgamma_literal_identities(order):
    # psi^(m)(1)=(-1)^(m+1) m! zeta(m+1); psi(1)=-EulerGamma.
    expected = [-np.euler_gamma, np.pi**2/6, -2*1.202056903159594, np.pi**4/15][order]
    close(math.polygamma(tensor(order, np.int64), tensor([1.])), [expected])
    close(math.lgamma(tensor([1., 2., 3., 4.])), [0., 0., np.log(2), np.log(6)])


@pytest.mark.parametrize('side', ['left_scalar', 'right_scalar', 'left_broadcast', 'right_broadcast'])
def test_zeta_scalar_cast_and_broadcast(side):
    if side == 'left_scalar':
        left, right, expected = 2., tensor([1., 2.]), [np.pi**2/6, np.pi**2/6-1]
    elif side == 'right_scalar':
        left, right, expected = tensor([2., 4.]), 1., [np.pi**2/6, np.pi**4/90]
    elif side == 'left_broadcast':
        left, right, expected = tensor([2.]), tensor([1., 2.]), [np.pi**2/6, np.pi**2/6-1]
    else:
        left, right, expected = tensor([2., 4.]), tensor([1.]), [np.pi**2/6, np.pi**4/90]
    close(math.zeta(left, right), expected)


@pytest.mark.parametrize('kind', ['addcdiv', 'logit', 'cosine', 'nansum', 'prod'])
def test_miscellaneous_cpu_math(kind, monkeypatch):
    if kind == 'addcdiv':
        close(math.addcdiv(tensor([1., 2.]), tensor([2., 6.]), tensor([2., 3.]), value=2), [3., 6.])
        close(tm.deprecated_tensor_addcdiv(tensor([1., 2.]), tensor([2., 6.]), tensor([2., 3.])), [2., 4.])
    elif kind == 'logit':
        values = np.array([.2, .5, .8], np.float32)
        close(math.logit(tensor(values)), np.log(values/(1-values)))
    elif kind == 'cosine':
        close(math.cosine_similarity(tensor([[3., 4.], [0., 0.]]), tensor([[4., 3.], [1., 2.]]), dim=1), [.96, 0.])
    elif kind == 'nansum':
        values = np.array([[1., np.nan], [3., 4.]], np.float32)
        close(math.nansum(tensor(values), axis=1, keepdims=True, dtype=ms.float64), np.nansum(values, axis=1, keepdims=True))
    elif kind == 'prod':
        close(math.prod(tensor([[1, 2], [3, 4]], np.int32), axis=1, keep_dims=True, dtype=ms.float32), [[2], [12]])


def test_accumulate_lazy_native_factory_protocol(monkeypatch):
    """Protocol only: AccumulateNV2 is absent on CPU; verify lazy reuse and operand identity."""
    monkeypatch.setattr(math, '_accumulate_prim', None)
    operands = [tensor([1., 2.]), tensor([3., 4.])]
    output = tensor([4., 6.])
    constructed, called = [], []
    def native_call(primitive, received):
        assert received is operands
        called.append(primitive)
        return output
    original_init = math.P.AccumulateNV2.__init__
    def native_init(primitive):
        constructed.append(primitive)
        original_init(primitive)
    monkeypatch.setattr(math.P.AccumulateNV2, '__init__', native_init)
    monkeypatch.setattr(math.P.AccumulateNV2, '__call__', native_call)
    assert math.accumulate_n(operands) is output
    assert math.accumulate_n(operands) is output
    assert len(constructed) == 1 and called == constructed * 2
    close(output, [4., 6.])


def test_fmin_native_boundary_protocol(monkeypatch):
    """Protocol only: Python wrapper constructs Fmin and passes original operands."""
    left, right = tensor([np.nan, 2., 3.]), tensor([1., np.nan, 2.])
    output, calls = tensor([1., 2., 2.]), []
    def primitive_call(primitive, x, y):
        assert primitive.name == 'Fmin' and x is left and y is right
        calls.append(primitive)
        return output
    monkeypatch.setattr(math.Fmin, '__call__', primitive_call)
    assert math.fmin(left, right) is output
    assert len(calls) == 1


@pytest.mark.parametrize('name', ['bitwise_left_shift', 'bitwise_right_shift'])
@pytest.mark.parametrize('side', ['left', 'right', 'tensor'])
def test_bit_shifts_promote_scalar_integer(name, side):
    left = 4 if side == 'left' else tensor([4, 8], np.int32)
    right = 1 if side == 'right' else tensor([1, 2], np.int32)
    oracle = np.left_shift if name == 'bitwise_left_shift' else np.right_shift
    close(getattr(math, name)(left, right), oracle(4 if side == 'left' else [4, 8], 1 if side == 'right' else [1, 2]))


@pytest.mark.parametrize('name', ['bitwise_left_shift', 'bitwise_right_shift'])
@pytest.mark.parametrize('left,right', [(1, 2), (1.5, tensor([1], np.int32)), (tensor([1], np.int32), 1.5)])
def test_bit_shifts_reject_noninteger_scalars(name, left, right):
    with pytest.raises(TypeError):
        getattr(math, name)(left, right)


@pytest.mark.parametrize('name,args,kwargs,error', [
    ('arccosh', (tensor([2.]),), {}, None),
    ('float_power_ext', (tensor([1.]), 1j), {}, TypeError),
    ('logaddexp', ([1], tensor([1.])), {}, TypeError),
    ('logaddexp', (tensor([1.]), [1]), {}, TypeError),
    ('logaddexp', (tensor([1], np.int32), tensor([1], np.int32)), {}, TypeError),
    ('outer', ([1], tensor([1.])), {}, TypeError), ('outer', (tensor([1.]), [1]), {}, TypeError),
    ('mv', ([1], tensor([1.])), {}, TypeError), ('mv', (tensor([[1.]]), [1]), {}, TypeError),
    ('histc', (tensor([1.]),), {'min':'a'}, TypeError), ('histc', (tensor([1.]),), {'max':[]}, TypeError),
    ('copysign', (tensor([1.]), []), {}, TypeError),
    ('copysign', (tensor([1j], np.complex64), tensor([1.])), {}, TypeError),
    ('copysign', (tensor([1.]), tensor([1j], np.complex128)), {}, TypeError),
    ('nansum', (tensor([1j], np.complex64),), {}, TypeError),
    ('nansum', (tensor([1.]),), {'dtype':ms.complex64}, TypeError),
    ('diag_embed', ([1],), {}, TypeError), ('diag_embed', (tensor([True], np.bool_),), {}, TypeError),
    ('diag_embed', (tensor([1.]),), {'dim1':3}, ValueError),
    ('diag_embed', (tensor([1.]),), {'dim2':-3}, ValueError),
    ('diag_embed', (tensor([1.]),), {'dim1':0,'dim2':0}, ValueError),
    ('atleast_1d', ([1],), {}, TypeError), ('atleast_2d', ([1],), {}, TypeError),
    ('atleast_3d', ([1],), {}, TypeError), ('count_nonzero', (tensor([1.]),), {'axis':[.5]}, TypeError),
    ('rotated_iou', (tensor(np.ones((1, 2, 5)), np.int32), tensor(np.ones((1, 2, 5)))), {}, ValueError),
    ('rotated_iou', (tensor(np.ones((1, 2, 5))), tensor(np.ones((1, 2, 5)), np.int32)), {}, ValueError),
])
def test_additional_math_type_and_axis_boundaries(name, args, kwargs, error):
    if error is None:
        close(getattr(math, name)(*args, **kwargs), np.arccosh([2.]))
    else:
        with pytest.raises(error):
            getattr(math, name)(*args, **kwargs)


@pytest.mark.parametrize('name', ['addmm', 'addbmm'])
@pytest.mark.parametrize('attribute', ['alpha', 'beta'])
def test_matrix_accumulation_rejects_nonscalar_scaling(name, attribute):
    values = tensor(np.ones((2, 2)))
    with pytest.raises(TypeError, match=attribute):
        getattr(math, name)(values, values, values, **{attribute: 'bad'})


@pytest.mark.parametrize('kind', ['type', 'dtype', 'float_attribute', 'int_attribute'])
def test_batched_matmul_accumulation_validation(kind):
    dtype = np.int32 if kind == 'int_attribute' else np.float32
    values = tensor(np.ones((2, 2, 2)), dtype)
    left = [1] if kind == 'type' else values
    right = tensor(np.ones((2, 2, 2)), np.float64) if kind == 'dtype' else values
    kwargs = {'alpha':.5} if kind == 'int_attribute' else {'beta':'bad'} if kind == 'float_attribute' else {}
    with pytest.raises(TypeError):
        math.baddbmm(values, left, right, **kwargs)


@pytest.mark.parametrize('kind', ['input', 'dtype', pytest.param('integer', marks=pytest.mark.xfail(condition=np.lib.NumpyVersion(np.__version__) >= '2.0.0', strict=True, raises=AttributeError, reason='ops.scalar_cast still calls removed np.cast under NumPy 2'))])
def test_addmv_types_and_integer_scale_cast(kind):
    if kind == 'integer':
        close(math.addmv(tensor([1, 2], np.int32), tensor([[1, 2], [3, 4]], np.int32),
                         tensor([1, 2], np.int32), beta=2, alpha=3), [17, 37])
    else:
        with pytest.raises(TypeError):
            math.addmv(tensor([1., 2.]), [[1, 2], [3, 4]] if kind == 'input' else tensor([[1, 2], [3, 4]]),
                       tensor([1, 2], np.float64))


@pytest.mark.parametrize('name,axis,expected', [('tensor_argmax', 1, [2, 2]), ('tensor_argmin', 0, [0, 0, 0])])
def test_tensor_arg_extrema(name, axis, expected):
    close(getattr(tm, name)(tensor([[1., 2., 3.], [4., 5., 6.]]), axis), expected)


@pytest.mark.parametrize('name', ['tensor_clamp_scalar', 'tensor_clamp_tensor'])
def test_tensor_clamping(name):
    values = tensor([-2., 1., 4.])
    lower, upper = (tensor(0.), tensor(2.)) if name.endswith('tensor') else (0., 2.)
    close(getattr(tm, name)(values, lower, upper), [0., 1., 2.])


@pytest.mark.parametrize('kind', ['isclose', 'nan_to_num', 'where', 'old_where', 'gcd',
                                'take', 'hardshrink', 'new_ones', 'new_zeros', 'pixel_shuffle'])
def test_remaining_tensor_cpu_wrappers(kind):
    if kind == 'isclose':
        close(tm.deprecated_tensor_isclose(tensor([1., 2.]), tensor([1.000001, 3.])), [True, False])
    elif kind == 'nan_to_num':
        close(tm.tensor_nan_to_num(tensor([np.nan, np.inf, -np.inf]), nan=2., posinf=3., neginf=-4.), [2., 3., -4.])
    elif kind in ('where', 'old_where'):
        cond, left, right = tensor([True, False], np.bool_), tensor([1., 2.]), tensor([3., 4.])
        close(tm.tensor_where(cond, left, right) if kind == 'where' else tm.deprecated_tensor_where(left, cond, right), [1., 4.])
    elif kind == 'gcd':
        close(tm.tensor_gcd(tensor([6, 12], np.int32), tensor([9, 8], np.int32)), [3, 4])
    elif kind == 'take':
        close(tm.tensor_take(tensor([1., 2., 3.]), tensor([2, 0], np.int32)), [3., 1.])
        with pytest.raises(ValueError, match='mode'):
            tm.deprecated_tensor_take(tensor([1.]), tensor([0], np.int32), mode='bad')
    elif kind == 'hardshrink':
        close(tm.tensor_hardshrink(tensor([-1., -.2, .2, 1.])), [-1., 0., 0., 1.])
    elif kind in ('new_ones', 'new_zeros'):
        close(getattr(tm, 'tensor_' + kind)(tensor([5.]), (2, 3), dtype=ms.float64),
              np.ones((2, 3)) if kind == 'new_ones' else np.zeros((2, 3)))
    else:
        data = np.arange(16, dtype=np.float32).reshape(1, 4, 2, 2)
        close(tm.deprecated_pixel_shuffle(tensor(data), 2), data.reshape(1, 1, 2, 2, 2, 2).transpose(0, 1, 4, 2, 5, 3).reshape(1, 1, 4, 4))


@pytest.mark.parametrize('name', ['tensor_addbmm', 'deprecated_tensor_addbmm', 'tensor_addmm',
                                'deprecated_tensor_addmm', 'deprecated_tensor_addmv'])
def test_tensor_scaled_matrix_sum(name):
    a, b = np.array([[1., 2.], [3., 4.]], np.float32), np.array([[2., 1.], [4., 3.]], np.float32)
    if name.endswith('addmv'):
        close(getattr(tm, name)(tensor([1., 2.]), tensor(a), tensor([2., 3.]), beta=2, alpha=3), [26., 58.])
    elif name.endswith('addbmm'):
        close(getattr(tm, name)(tensor(np.ones((2, 2))), tensor(a[None]), tensor(b[None]), beta=2, alpha=3), 2+3*(a@b))
    else:
        close(getattr(tm, name)(tensor(np.ones((2, 2))), tensor(a), tensor(b), beta=2, alpha=3), 2+3*(a@b))


@pytest.mark.parametrize('kind', ['scalar', 'tensor', 'wrong_condition', 'wrong_shape'])
def test_deprecated_select_scalar_conversion(kind):
    x = tensor([1., 2.])
    condition = [True, False] if kind == 'wrong_condition' else tensor([True, False], np.bool_)
    y = tensor([[3., 4.]]) if kind == 'wrong_shape' else tensor([3., 4.]) if kind == 'tensor' else 3.
    if kind == 'wrong_condition':
        with pytest.raises((TypeError, ValueError)):
            tm.deprecated_tensor_select(x, condition, y)
    else:
        close(tm.deprecated_tensor_select(x, condition, y), [[1., 4.]] if kind == 'wrong_shape' else [1., 4.] if kind == 'tensor' else [1., 3.])


@pytest.mark.parametrize('name', ['tensor_xlogy'])
@pytest.mark.parametrize('other', [2., tensor([1., 2.])])
def test_tensor_xlogy_scalar_or_tensor(name, other):
    close(getattr(tm, name)(tensor([1., 2.]), other), [np.log(2), 2*np.log(2)] if isinstance(other, float) else [0., 2*np.log(2)])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='tensor_unsqueeze placeholder returns None instead of the expanded tensor')
def test_tensor_unsqueeze_returns_expanded_tensor():
    close(tm.tensor_unsqueeze(tensor([1., 2.]), 0), [[1., 2.]])


@pytest.mark.parametrize('name,boundary,args,kwargs,native_args,native_kwargs,expected', [
    ('tensor_gather_ext','gather_ext',(tensor([[1.,2.],[3.,4.]]),1,tensor([[1],[0]],np.int32)),{},None,{},[[2.],[3.]]),
    ('tensor_repeat_interleave_ext','repeat_interleave_ext',(tensor([1.,2.]),2,0),{'output_size':4},
     'append_output_size',{},[1.,1.,2.,2.]),
    ('tensor_scatter','scatter',(tensor([[1.,2.],[3.,4.]]),1,tensor([[0],[1]],np.int32),tensor([[8.],[9.]])),{},None,{},[[8.,2.],[3.,9.]]),
    ('tensor_sigmoid_','sigmoid_',(tensor([0.]),),{},None,{},[.5]),
    ('tensor_inplace_scatter_src','inplace_scatter_src_op',(tensor([1.,2.]),0,tensor([1],np.int32),tensor([3.])),{},None,{},[1.,3.]),
    ('tensor_inplace_scatter_src_reduce','inplace_scatter_src_reduce_op',(tensor([1.,2.]),0,tensor([1],np.int32),tensor([3.])),{'reduce':'add'},None,{'reduce':'add'},[1.,5.]),
    ('tensor_inplace_scatter_value','inplace_scatter_value_op',(tensor([1.,2.]),0,tensor([1],np.int32),3.),{},None,{},[1.,3.]),
    ('tensor_inplace_scatter_value_reduce','inplace_scatter_value_reduce_op',(tensor([1.,2.]),0,tensor([1],np.int32),3.),{'reduce':'multiply'},None,{'reduce':'multiply'},[1.,6.]),
    ('tensor_acos','acos_ext',(tensor([.5]),),{},None,{},[np.pi/3]),
    ('tensor_acosh','acosh_ext',(tensor([2.]),),{},None,{},[np.arccosh(2)]),
    ('tensor_asin','asin_ext',(tensor([.5]),),{},None,{},[np.pi/6]),
    ('tensor_asinh','asinh_ext',(tensor([1.]),),{},None,{},[np.arcsinh(1)]),
    ('tensor_atan','atan_ext',(tensor([1.]),),{},None,{},[np.pi/4]),
    ('tensor_median','median',(tensor([3.,1.,2.]),),{},None,{},(2.,2.)),
    ('tensor_median_dim','median',(tensor([[3.,1.,2.]]),1,True),{},None,{},([[2.]],[[2]])),
    ('tensor_bitwise_not','bitwise_not_op',(tensor([0,1],np.int32),),{},None,{},[-1,-2]),
    ('tensor_addcdiv_ext','addcdiv_ext_op',(tensor([1.]),tensor([2.]),tensor([4.])),{'value':3},None,{'value':3},[2.5]),
    ('tensor_lerp','lerp',(tensor([1.,2.]),tensor([3.,4.]),tensor([.25,.5])),{},None,{},[1.5,3.]),
    ('tensor_lerp_scalar','lerp_scalar',(tensor([1.,2.]),tensor([3.,4.]),.5),{},None,{},[2.,3.]),
    ('tensor_log_','inplace_log_op',(tensor([1.,np.e]),),{},None,{},[0.,1.]),
    ('tensor_exp_','inplace_exp_op',(tensor([0.,1.]),),{},None,{},[1.,np.e]),
    ('tensor_floor_div_scalar','floor_div_scalar_op',(tensor([-3.,5.]),2.),{},None,{},[-2.,2.]),
    ('tensor_subtract','tensor_sub_ext',(tensor([3.,4.]),tensor([1.,2.])),{'alpha':2},None,{'alpha':2},[1.,0.]),
    ('tensor_gelu','gelu',(tensor([0.]),),{'approximate':'tanh'},'append_approximate',{},[0.]),
    ('tensor_masked_scatter','masked_scatter',(tensor([1.,2.]),tensor([False,True],np.bool_),tensor([3.])),{},None,{},[1.,3.]),
])
def test_tensor_remaining_numeric_or_native_protocol(monkeypatch, name, boundary, args, kwargs, native_args, native_kwargs, expected):
    """Use real CPU numerics where available; otherwise verify complete native-boundary protocols."""
    if name in ('tensor_gather_ext', 'tensor_scatter', 'tensor_median', 'tensor_median_dim',
                'tensor_lerp', 'tensor_subtract', 'tensor_gelu', 'tensor_masked_scatter'):
        close(getattr(tm, name)(*args, **kwargs), expected)
        return
    forwarded = args + (kwargs['output_size'],) if native_args == 'append_output_size' else args + (kwargs['approximate'],) if native_args == 'append_approximate' else args
    output = tuple(tensor(item) for item in expected) if name == 'tensor_median_dim' else tensor(expected)
    calls = []
    def native_call(*received, **options):
        assert len(received) == len(forwarded) and options == native_kwargs
        for actual, original in zip(received, forwarded):
            assert actual is original
        calls.append(received)
        return output
    monkeypatch.setattr(tm, boundary, native_call)
    assert getattr(tm, name)(*args, **kwargs) is output
    assert len(calls) == 1
    close(output, expected)


@pytest.mark.parametrize('name', ['tensor_std', 'deprecated_tensor_std'])
def test_tensor_std_variance_boundary_and_real_square_root(monkeypatch, name):
    """Tensor.var has no CPU PyBoost kernel; its arguments and the real square-root transform are checked."""
    values = tensor([[1., 2., 3.], [2., 4., 6.]])
    calls = []
    def native_variance(input_x, axis, correction, keepdim):
        assert input_x is values and axis == 1 and correction == 1 and keepdim is True
        calls.append(input_x)
        return tensor([[1.], [4.]])
    monkeypatch.setattr(ms.Tensor, 'var', native_variance)
    kwargs = {'ddof':1,'keepdims':True} if name.startswith('deprecated') else {'correction':1,'keepdim':True}
    close(getattr(tm, name)(values, 1, **kwargs), [[1.], [2.]])
    assert len(calls) == 1


@pytest.mark.parametrize('name', ['deprecated_tensor_sub', 'deprecated_tensor_sub_'])
def test_tensor_subtraction_of_sparse_coo(name):
    sparse = ms.COOTensor(tensor([[0,1],[1,0]],np.int32), tensor([2.,3.]), (2,2))
    close(getattr(tm, name)(tensor([[5.,6.],[7.,8.]]), sparse), [[5.,4.],[4.,8.]])


@pytest.mark.parametrize('dtype,y', [(np.float32, 1), (np.int32, 1.), (np.float32, [])])
def test_deprecated_select_rejects_scalar_dtype_mismatch(dtype, y):
    with pytest.raises(TypeError):
        tm.deprecated_tensor_select(tensor([1,2],dtype),tensor([True,False],np.bool_),y)


def test_deprecated_select_integer_scalar():
    close(tm.deprecated_tensor_select(tensor([1,2],np.int32),tensor([True,False],np.bool_),3),[1,3])


@ms.jit
def unique_with_mutable_flags(value, flag):
    return tm.deprecated_tensor_unique(value, return_inverse=flag)


def test_unique_rejects_mutable_return_flags():
    with pytest.raises(ValueError, match='mutable'):
        unique_with_mutable_flags(tensor([1,1,2]), ms.mutable(True))


@pytest.mark.parametrize('axis', [0,pytest.param(1, marks=pytest.mark.xfail(strict=True, raises=IndexError, reason='trapz indexes unpadded lower-rank coordinate tensor with the y axis')),2])
def test_trapezoid_lower_rank_coordinates(axis):
    data=np.arange(24,dtype=np.float32).reshape(2,3,4)
    # The lower-rank path pads leading axes; integrate its actual trailing coordinate axis.
    if axis==0:
        data=data[:0]
        close(math.trapz(tensor(data),tensor(np.zeros((3,4))),dim=0),np.zeros((3,4)))
    elif axis==1:
        coords=np.array([[0.,1.,3.,6.],[0.,2.,4.,6.],[0.,3.,6.,9.]],np.float32)
        # After leading broadcast, coordinates vary along y axis 2.
        oracle=getattr(np,'trapezoid',np.trapz)(data,x=np.broadcast_to(coords,data.shape),axis=2)
        close(math.trapz(tensor(data),tensor(coords),dim=2),oracle)
    else:
        close(math.trapz(tensor(np.array([1,3,5],np.int32)),dx=.5),3.)


@pytest.mark.parametrize('axes', [0,1,([1],[0]),([-1],[0])])
def test_tensor_dot_contraction_and_outer_product(axes):
    left=np.arange(6,dtype=np.float32).reshape(2,3)
    right=np.arange(12,dtype=np.float32).reshape(3,4)
    close(math.tensor_dot(tensor(left),tensor(right),axes),np.tensordot(left,right,axes))


@pytest.mark.parametrize('axes,error', [(-1,ValueError),(3,ValueError),((0,),ValueError),
    (([0],[0,1]),ValueError),(([0,0],[0,1]),ValueError),(([2],[0]),ValueError),
    (([0,1,2],[0,1,2]),ValueError),(([0],[0]),ValueError),('bad',TypeError)])
def test_tensor_dot_rejects_invalid_contraction_axes(axes,error):
    with pytest.raises(error):
        math.tensor_dot(tensor(np.ones((2,3))),tensor(np.ones((3,4))),axes)


@pytest.mark.parametrize('left_dtype,right_dtype',[(np.int32,np.float32),(np.float32,np.int32),(np.float16,np.float32)])
def test_tensor_dot_rejects_unsupported_or_different_dtypes(left_dtype,right_dtype):
    with pytest.raises(TypeError):
        math.tensor_dot(tensor(np.ones((2,2)),left_dtype),tensor(np.ones((2,2)),right_dtype),1).asnumpy()


@pytest.mark.parametrize('labels,expected', [([0,26],'Aa'),([Ellipsis,1],'...B')])
def test_einsum_sublist_converter(labels,expected):
    x=tensor([[1.,2.],[3.,4.]])
    equation,operands=math._einsum_convert_sublist(x,(labels,labels))
    assert equation==expected+'->'+expected and len(operands)==1 and operands[0] is x
    with pytest.raises(ValueError,match='range'):
        math._einsum_convert_sublist(x,([52],))
    with pytest.raises(ValueError,match='at least one'):
        math._einsum_convert_sublist('i->',())


@pytest.mark.parametrize('style',['equation','explicit_output','implicit_output','ellipsis'])
def test_einsum_gpu_native_boundary_protocol(monkeypatch,style):
    """Protocol only: sublist label conversion and operand order before the GPU-only Einsum primitive."""
    a,b=tensor([[1.,2.],[3.,4.]]),tensor([[2.,1.],[4.,3.]])
    specs={
        'equation':(('ij,jk->ik',a,b),'ij,jk->ik',(a,b)),
        'explicit_output':((a,[0,26],b,[26,1],[0,1]),'Aa,aB->AB',(a,b)),
        'implicit_output':((a,[0,26],b,[26,1]),'Aa,aB',(a,b)),
        'ellipsis':((a,[Ellipsis,0],[Ellipsis,0]),'...A->...A',(a,))}
    args,equation,operands=specs[style]
    output=tensor(np.einsum(equation,*(value.asnumpy() for value in operands)))
    calls=[]
    def native_call(primitive,received):
        assert primitive.attrs['equation']==equation
        assert len(received)==len(operands)
        assert all(item is wanted for item,wanted in zip(received,operands))
        calls.append(received)
        return output
    monkeypatch.setattr(math.P.Einsum,'__call__',native_call)
    assert math.einsum(*args) is output
    assert len(calls)==1


@pytest.mark.parametrize('number,expected',[(0,'A'),(25,'Z'),(26,'a'),(51,'z'),(Ellipsis,'...')])
def test_einsum_label_alphabet_boundaries(number,expected):
    assert math._einsum_convert_num_to_char(number)==expected
    with pytest.raises(ValueError,match='range'):
        math._einsum_convert_num_to_char(52)


@pytest.fixture(scope='module')
def source_math():
    """Load the complete repository module under its own alias, during fixture setup."""
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[4]/'mindspore/python/mindspore/ops/function/math_func.py'
    spec=importlib.util.spec_from_file_location('mindspore.ops.function._remaining_source_math',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('order', [float('inf'),-float('inf'),'nuc','fro'])
@pytest.mark.parametrize('dims', [(0,1),(1,0)])
def test_repository_matrix_other_norm_axis_order(source_math,order,dims):
    data=np.array([[1.,2.,3.],[4.,5.,7.]],np.float32)
    close(source_math.norm(tensor(data),ord=order,dim=dims,keepdim=True),
          np.linalg.norm(data,ord=order,axis=dims,keepdims=True))


def test_repository_matrix_other_norm_duplicate_axes(source_math):
    with pytest.raises(ValueError,match='duplicate'):
        source_math.norm(tensor([[1.,2.],[3.,4.]]),ord=float('inf'),dim=(0,0))


@pytest.mark.parametrize('order', [0.,.5,1.5,tensor(1.5)])
def test_fractional_and_tensor_vector_norm_orders(order):
    values=np.array([1.,4.,9.],np.float32)
    exponent=float(order.asnumpy()) if isinstance(order,ms.Tensor) else order
    close(math.norm(tensor(values),ord=order,dim=0,keepdim=True),np.linalg.norm(values,ord=exponent,keepdims=True))


@pytest.mark.parametrize('name,kwargs,error',[
    ('norm',{'dim':[0]},TypeError),('norm',{'dim':(0,1,2)},ValueError),
    ('norm',{'dim':0,'ord':'fro'},TypeError),('norm',{'dim':(0,1),'ord':3},ValueError),
    ('vector_norm',{'axis':[0]},TypeError),('vector_norm',{'axis':(0,0)},ValueError),
    ('vector_norm',{'ord':'bad'},ValueError),('matrix_norm',{'axis':1},TypeError),
    ('matrix_norm',{'axis':(0,)},ValueError),('matrix_norm',{'axis':(0,0)},ValueError),
    ('linalg_norm',{'ord':1,'dim':None},ValueError),('linalg_norm',{'dim':(0,1,2)},ValueError),
    ('norm_ext',{'dim':(0,1,2),'p':'fro'},ValueError),
])
def test_norm_axis_and_order_validation(name,kwargs,error):
    values=tensor(np.ones((2,3,4)) if name in ('linalg_norm','norm_ext') else np.ones((2,3)))
    with pytest.raises(error):
        getattr(math,name)(values,**kwargs)


@pytest.mark.parametrize('name', ['norm','vector_norm','matrix_norm'])
def test_norm_dtype_and_keep_dimensions(name):
    values=np.array([[1.,2.],[3.,4.]],np.float32)
    kwargs={'keepdim':True} if name=='norm' else {'keepdims':True}
    expected=np.linalg.norm(values,keepdims=True)
    close(getattr(math,name)(tensor(values),dtype=ms.float64,**kwargs),expected)


@pytest.mark.parametrize('left,right,expected',[(ms.float32,ms.float16,ms.float32),(ms.float16,ms.float16,ms.float16)])
def test_legacy_norm_dtype_resolution(left,right,expected):
    assert math._check_dtype(left,right)==expected
    with pytest.raises(ValueError,match='dtype'):
        math._check_dtype(ms.float16,ms.float64)


@pytest.mark.parametrize('shapes,transpose,expected',[
    (((2,3),(3,4)),False,(2,4)),(((2,3),(4,3)),True,(2,4)),
    (((3,),(3,4)),False,(4,)),(((),()),True,()),(((2,3),(3,)),True,(2,))])
def test_matmul_shape_remainder_helper(shapes,transpose,expected):
    left,right=shapes
    assert math._infer_shape_rem(left,right,len(left),len(right),transpose)==expected


@pytest.mark.parametrize('name,args,expected',[
    ('_check_linalg_norm_input',(None,None,3),((0,1,2),True)),
    ('_check_linalg_norm_input',(None,1,2),((0,1),False)),
    ('_check_linalg_norm_input',(1,2,2),((1,),False)),
    ('_check_need_broadcast',((2,3,4),(1,4,5)),True),
    ('_check_need_broadcast',((2,3),(3,4)),False),
    ('_max',(1,3,2),3),('_min',(1,3,2),1),
    ('add_padding_to_shape',((3,4),4),[1,1,3,4]),
    ('add_padding_to_shape',((3,4),1),[3,4]),
])
def test_legacy_shape_helper_contracts(name,args,expected):
    assert getattr(math,name)(*args)==expected


@pytest.mark.parametrize('name,inputs,kwargs,expected',[
    ('prod',[[1,2],[3,4]],{'axis':(1,),'keep_dims':True,'dtype':ms.float32},[[2],[12]]),
    ('nansum',[[True,False],[True,True]],{'axis':1},[1,2]),
    ('cumsum',[[1,2],[3,4]],{'axis':1,'dtype':ms.float64},[[1,3],[3,7]]),
    ('copysign',[1,-2],{'other':-1.},[-1,-2]),
    ('dist',3.,{'other':tensor(7.)},4.),
])
def test_more_math_scalar_and_reduction_branches(name,inputs,kwargs,expected):
    dtype=np.bool_ if name=='nansum' else np.int32 if name=='prod' else np.float32
    close(getattr(math,name)(tensor(inputs,dtype),**kwargs),expected)


@pytest.mark.parametrize('name', ['fft','ifft'])
@pytest.mark.parametrize('input_kind,kwargs,error',[
    ('list',{},TypeError),('bool',{},TypeError),('complex',{'n':0},ValueError),
    ('complex',{'n':1.5},TypeError)])
def test_fft_public_input_and_size_validation(name,input_kind,kwargs,error):
    x=[1,2] if input_kind=='list' else tensor([True,False],np.bool_) if input_kind=='bool' else tensor([1+2j,3+4j],np.complex64)
    with pytest.raises(error):
        getattr(math,name)(x,**kwargs)


@pytest.mark.parametrize('shape,center', [((8,),True),((2,8),True),((8,),False)])
def test_stft_window_defaults_and_padding_native_protocol(monkeypatch,shape,center):
    """Protocol only for native STFT; actual CPU Pad/window creation remains exercised."""
    data=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
    output=tensor([1.])
    calls=[]
    def native_call(x,window,n_fft,hop,win,normalized,onesided,return_complex):
        expected=np.pad(data,((2,2),) if len(shape)==1 else ((0,0),(2,2)),mode='reflect') if center else data
        close(x,expected)
        close(window,np.ones(4))
        assert (n_fft,hop,win,normalized,onesided,return_complex)==(4,1,4,False,True,False)
        calls.append(x)
        return output
    monkeypatch.setattr(math,'stft_impl',native_call)
    assert math.stft(tensor(data),4,center=center) is output
    assert len(calls)==1


@pytest.mark.parametrize('name', ['var_mean','std_mean'])
def test_variance_nonstandard_degrees_of_freedom(name):
    values=np.arange(12,dtype=np.float32).reshape(3,4)
    first=(np.var if name=='var_mean' else np.std)(values,axis=1,ddof=2,keepdims=True)
    close(getattr(math,name)(tensor(values),axis=1,ddof=2,keepdims=True),(first,np.mean(values,axis=1,keepdims=True)))


@pytest.mark.parametrize('name,args,kwargs,error',[
    ('is_complex',([1],),{},TypeError),('norm_ext',([1],),{},TypeError),
    ('inner',(tensor([1.,2.]),tensor([1.,2.,3.])),{},ValueError),
    ('logaddexp2',(tensor([1],np.int32),tensor([1],np.int32)),{},TypeError),
    ('addr',(tensor([[1.]]),tensor([1.]),tensor([1.],np.float64)),{},TypeError),
    ('copysign',(tensor([1.]),tensor([1j],np.complex64)),{},TypeError),
    ('logcumsumexp',(tensor([1.]),1.5),{},TypeError),
    ('nansum',(tensor([[1.,2.]]),),{'axis':(0,-2)},RuntimeError),
    ('tanhshrink',([1],),{},TypeError),('zeta',([],2.),{},TypeError),
])
def test_final_public_validation_branches(name,args,kwargs,error):
    with pytest.raises(error):
        getattr(math,name)(*args,**kwargs)


@pytest.mark.parametrize('name', ['_typecheck_input','_typecheck_input_dot'])
@pytest.mark.parametrize('left,right',[(ms.float16,ms.float32),(ms.int32,ms.float32),(ms.float32,ms.int32)])
def test_legacy_contraction_dtype_validators(name,left,right):
    with pytest.raises(TypeError):
        getattr(math,name)(left,right,'test_contraction')
    assert getattr(math,name)(ms.float32,ms.float32,'test_contraction') is None


@pytest.mark.parametrize('axes',[(0,1),(1,),4,0,(4,1),'bad'])
def test_batch_dot_rejects_invalid_axes(axes):
    with pytest.raises(ValueError):
        math.batch_dot(tensor(np.ones((2,3,4))),tensor(np.ones((2,4,5))),axes)


def test_batch_dot_rank_and_batch_size_validation():
    with pytest.raises(ValueError,match='dimension'):
        math._get_batch_size((3,),(2,3),'batch_dot')
    with pytest.raises(ValueError,match='batch sizes'):
        math._check_batch_size(2,3,'batch_dot')


@pytest.mark.parametrize('axis',[(0,1),[0,-1],0])
def test_count_nonzero_legacy_axis_validation(axis):
    assert math._check_validate_axis(axis,'count_nonzero')==axis
    with pytest.raises(TypeError):
        math._check_validate_axis([.5],'count_nonzero')


def test_norm_and_axis_private_validation_contracts():
    with pytest.raises(ValueError,match='aligned'):
        math._check_last_dim_shape_eq(tensor([1.,2.]),tensor([1.,2.,3.]))
    assert math._check_last_dim_shape_eq(tensor([1.,2.]),tensor([3.,4.])) is None
    with pytest.raises(TypeError,match='integers'):
        math._check_dim_in_range(1.5,2)
    with pytest.raises(ValueError,match='not support'):
        math._norm_matrix_int_ord(tensor([[1.,2.],[3.,4.]]),3,(0,1),False,2)
    with pytest.raises(ValueError,match='op input'):
        math._multi_svd_norm(tensor([[1.,2.],[3.,4.]]),0,1,'invalid')
    assert math._maybe_wrap_dims_n([-1],0)==[0]


@pytest.mark.parametrize('kind',['eps','orgqr','nanmedian','renorm','lerp','complexnorm','intnorm','smallestnorm',
                                'floatdiv','boolxlogy',pytest.param('boollog2', marks=pytest.mark.xfail(strict=True, raises=TypeError, reason='log2 casts bool to int64 but Log CPU kernel rejects int64')),'inttanh','atleast3d','cummin','listrepeat'])
def test_remaining_cpu_mathematical_identities(kind):
    if kind=='eps':
        close(math.eps(tensor([1.,2.])),np.full(2,np.finfo(np.float32).eps))
    elif kind=='orgqr':
        # Zero Householder coefficients produce the corresponding columns of identity.
        close(math.orgqr(tensor([[2.,3.],[.5,4.]]),tensor([0.,0.])),np.eye(2))
    elif kind=='nanmedian':
        value,index=math.nanmedian(tensor([[1.,np.nan,3.]]),axis=1,keepdims=True)
        close(value,[[1.]])
        close(index,[[0]])
    elif kind=='renorm':
        close(math.renorm(tensor([[3.,4.],[0.,2.]]),2,0,2.),[[1.2,1.6],[0.,2.]])
    elif kind=='lerp':
        close(math.lerp(tensor([1.,2.]),tensor([3.,4.]),.5),[2.,3.])
    elif kind=='complexnorm':
        close(math.norm(tensor([3+4j,0j],np.complex64)),5.)
    elif kind=='intnorm':
        close(math.norm(tensor([1.,0.,3.]),ord=0,dim=0),2.)
    elif kind=='smallestnorm':
        values=np.array([[1.,2.],[3.,5.]],np.float32)
        close(math.norm(tensor(values),ord=-2,dim=(0,1),keepdim=True),np.linalg.norm(values,ord=-2,keepdims=True))
    elif kind=='floatdiv':
        value=tensor([1.,2.],np.float64)
        assert math._get_default_div_type(value) is value
    elif kind=='boolxlogy':
        close(math.xlogy(tensor([False,True],np.bool_),tensor([True,True],np.bool_)),[0.,0.])
    elif kind=='boollog2':
        close(math.log2(tensor([True,True],np.bool_)),[0.,0.])
    elif kind=='inttanh':
        close(math.tanhshrink(tensor([0,1],np.int32)),[0.,1-np.tanh(1.)])
    elif kind=='atleast3d':
        value=tensor(np.arange(8).reshape(2,2,2))
        assert math.atleast_3d(value) is value
    elif kind=='cummin':
        assert math._create_cummin_perm(-1,(2,3,4))==(2,1,0)
    else:
        value=tensor([1.,2.])
        result=math._list_comprehensions(3,value)
        assert len(result)==3 and all(item is value for item in result)


@pytest.mark.parametrize('name', ['arccosh_ext','arcsin_ext','arctan_ext','arctan2_ext','arccos_ext','arcsinh_ext',
                                'remainder_ext','mul_ext','mean_ext'])
def test_accelerator_math_native_boundary_protocol(monkeypatch,name):
    """Protocol only for CPU-missing mathematical kernels; wrapper arguments and return identity are checked."""
    values,other=tensor([.5,1.]),tensor([2.,3.])
    simple={
        'arccosh_ext':('acosh_ext',(other,),{},np.arccosh([2.,3.])),
        'arcsin_ext':('asin_ext',(values,),{},np.arcsin([.5,1.])),
        'arctan_ext':('atan_ext',(values,),{},np.arctan([.5,1.])),
        'arctan2_ext':('atan2_ext',(values,other),{},np.arctan2([.5,1.],[2.,3.])),
        'arccos_ext':('acos_ext',(values,),{},np.arccos([.5,1.])),
        'arcsinh_ext':('asinh_ext',(values,),{},np.arcsinh([.5,1.])),
        'remainder_ext':('remainder_tensor_tensor_',(other,values),{},[0.,0.]),
        'mul_ext':('mul',(values,other),{},[1.,3.]),
        'mean_ext':('mean_ext_op',(values,0,True,ms.float64),{},[.75])}
    boundary,args,kwargs,expected=simple[name]
    if name in ('mul_ext', 'mean_ext'):
        close(getattr(math,name)(*args,**kwargs),expected)
        return
    output,calls=tensor(expected),[]
    def native_call(*received,**options):
        assert len(received)==len(args) and options==kwargs
        assert all(actual is original for actual,original in zip(received,args))
        calls.append(received)
        return output
    monkeypatch.setattr(math,boundary,native_call)
    assert getattr(math,name)(*args,**kwargs) is output
    assert len(calls)==1


@pytest.mark.parametrize('scalar_side',['left','right'])
def test_remainder_scalar_native_protocol(monkeypatch,scalar_side):
    x=tensor([3.,5.]);args=(2.,x) if scalar_side=='left' else (x,2.)
    boundary='remainder_scalar_tensor_' if scalar_side=='left' else 'remainder_tensor_scalar_'
    output=tensor([2.,2.] if scalar_side=='left' else [1.,1.]);calls=[]
    def native_call(left,right):
        assert left is args[0] and right is args[1]
        calls.append((left,right));return output
    monkeypatch.setattr(math,boundary,native_call)
    assert math.remainder_ext(*args) is output and len(calls)==1
    with pytest.raises(TypeError):
        math.remainder_ext(1,2)


@pytest.mark.parametrize('p,seed',[(.5,None),(tensor([.25,.75]),7)])
def test_bernoulli_rng_boundary_protocol(monkeypatch,p,seed):
    """Protocol only: preserve seed normalization and tensor probability conversion without drawing RNG."""
    x=tensor([0.,0.]);output=tensor([0.,1.]);calls=[]
    def native_call(primitive,input_x,probability):
        assert primitive.attrs['seed']==(-1 if seed is None else seed)
        assert input_x is x
        if isinstance(p,ms.Tensor): assert probability is p
        else: close(probability,[p])
        calls.append(probability);return output
    monkeypatch.setattr(math.Bernoulli,'__call__',native_call)
    assert math.bernoulli(x,p=p,seed=seed) is output and len(calls)==1
    with pytest.raises(TypeError): math.bernoulli(x,seed=.5)


def test_deprecated_index_add_tensor_method():
    x=tensor([[1.,2.],[3.,4.]])
    close(tm.deprecated_tensor_index_add(x,tensor([0],np.int32),tensor([[5.,6.]]),0,
                                       use_lock=True,check_index_bound=True),[[6.,8.],[3.,4.]])


@pytest.mark.xfail(strict=True, raises=AttributeError, reason='tensor_inplace_masked_scatter calls missing functional.inplace_masked_scatter')
def test_inplace_masked_scatter_returns_updated_tensor():
    close(tm.tensor_inplace_masked_scatter(tensor([1.,2.]),tensor([False,True],np.bool_),tensor([3.])),[1.,3.])


@pytest.mark.parametrize('name', ['diff_ext'])
@pytest.mark.parametrize('prepend,append',[(tensor([0.]),None),(None,tensor([8.]))])
def test_difference_optional_endpoint_helpers(name,prepend,append):
    data=tensor([1.,3.,6.])
    close(getattr(math,name)(data,n=tensor(1,np.int64),dim=0,prepend=prepend,append=append),
          np.diff([0.,1.,3.,6.] if prepend is not None else [1.,3.,6.,8.]))


@pytest.mark.parametrize('function',['fft','ifft'])
def test_fft_explicit_resize_changes_transform_length(function):
    values=np.array([1+1j,2-1j,3+2j],np.complex64)
    close(getattr(math,function)(tensor(values,np.complex64),n=5),getattr(np.fft,function)(values,n=5))


def test_fft_resize_equal_length_preserves_input_identity():
    values=tensor([1.,2.,3.])
    result=math._resize_input(values,1,[0],[3],[3])
    assert result is values


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='as_strided uses bfloat16 byte strides on its temporary float32 array')
def test_bfloat16_strided_view_uses_element_strides():
    data=ms.Tensor(np.array([1.,2.,3.,4.],np.float32),ms.bfloat16)
    view=math.as_strided(data,(2,2),(2,1))
    assert view.dtype==ms.bfloat16
    close(view.float(),[[1.,2.],[3.,4.]])


@pytest.mark.parametrize('order,dtype',[(.5,None),(.5,ms.float64)])
def test_vector_norm_float64_native_boundary_cast_protocol(monkeypatch,order,dtype):
    """Protocol only at LpNormV2; float64 downcast/upcast and requested dtype use real CPU Cast."""
    values=tensor([1.,4.],np.float64);calls=[]
    def native_call(x,p,dim,keepdim,epsilon):
        assert x.dtype==ms.float32 and p==order and dim==0 and keepdim is True and epsilon==0.
        close(x,[1.,4.]);calls.append(x)
        return tensor([9.])
    monkeypatch.setattr(math,'lp_norm_v2_op',native_call)
    result=math.vector_norm_ext(values,ord=order,dim=0,keepdim=True,dtype=dtype)
    close(result,[9.]);assert result.dtype==ms.float64 and len(calls)==1


@pytest.mark.parametrize('ord,dim',[(None,None),(1,0)])
def test_linalg_norm_vector_dispatch_native_protocol(monkeypatch,ord,dim):
    """Protocol only: confirm linalg.norm's flatten/default/order dispatch to native vector norm."""
    values=tensor([[1.,2.],[3.,4.]]);calls=[]
    def native_call(x,p,axes,keepdim,dtype):
        assert x is values and p==float(2 if ord is None else ord) and keepdim is True and dtype==ms.float64
        assert axes==((0,1) if dim is None else (0,) if isinstance(dim,int) else ())
        calls.append(x);return tensor([[np.sqrt(30)]] if ord is None else [[4.,6.]],np.float64)
    monkeypatch.setattr(math,'linalg_vector_norm_op',native_call)
    result=math.linalg_norm(values,ord=ord,dim=dim,keepdim=True,dtype=ms.float64)
    close(result,[[np.sqrt(30)]] if ord is None else [[4.,6.]]);assert len(calls)==1


@pytest.mark.parametrize('transpose',[False,True])
def test_ormqr_gpu_native_protocol(monkeypatch,transpose):
    """Protocol only for GPU-only Ormqr, including both boolean primitive attributes."""
    packed,tau,other=tensor([[2.,3.],[.5,4.]]),tensor([0.,0.]),tensor([[1.,2.],[3.,4.]])
    calls=[]
    def native_call(primitive,x,coefficients,y):
        assert primitive.attrs['left'] is False and primitive.attrs['transpose'] is transpose
        assert x is packed and coefficients is tau and y is other
        calls.append(x);return other
    monkeypatch.setattr(math.Ormqr,'__call__',native_call)
    assert math.ormqr(packed,tau,other,left=False,transpose=transpose) is other and len(calls)==1


@pytest.mark.parametrize('scalar_side',['left','right'])
def test_mul_ext_scalar_dispatch(scalar_side):
    x=tensor([1.,2.]);close(math.mul_ext(2.,x) if scalar_side=='left' else math.mul_ext(x,2.),[2.,4.])


def test_fft_equal_stride_axes_use_stable_permutation():
    values=tensor(np.arange(3,dtype=np.float32).reshape(1,1,3))
    output,permutation=math._permute_input(values,3,[2])
    close(output,np.arange(3).reshape(1,1,3))
    assert permutation==[0,1,2]


@pytest.mark.xfail(strict=True, raises=AttributeError, reason='empty replacement combinations returns TensorPy without the .size property required by its wrapper')
def test_empty_combinations_with_replacement():
    close(math.combinations(tensor([]),r=2,with_replacement=True),[])


def test_negative_infinity_math_wrapper():
    close(math.isneginf(tensor([-np.inf,0.,np.inf])),[True,False,False])


@pytest.mark.parametrize('dtype',[np.float16,np.float32])
def test_rotated_iou_casts_query_before_native_boundary(monkeypatch,dtype):
    """Protocol only: permute/float promotion and final output dtype are real CPU operations."""
    box=np.array([[[0.,0.,2.,2.,0.]]],dtype)
    calls=[]
    def native_call(left,right,trans,mode,is_cross,v_threshold,e_threshold):
        assert left.dtype==right.dtype==ms.float32
        close(left,box.transpose(0,2,1));close(right,box.transpose(0,2,1))
        assert (trans,mode,is_cross,v_threshold,e_threshold)==(False,0,True,0.,0.)
        calls.append(left);return tensor([[[1.]]])
    monkeypatch.setattr(math,'rotated_iou_op',native_call)
    result=math.rotated_iou(tensor(box,dtype),tensor(box,dtype))
    close(result,[[[1.]]]);assert result.dtype==(ms.float16 if dtype==np.float16 else ms.float32)
    assert len(calls)==1


@pytest.mark.parametrize('mode',['iou','iof'])
def test_box_overlap_cpu_identity_and_disjoint_boxes(mode):
    anchors=tensor([[0.,0.,2.,2.],[5.,5.,7.,7.]])
    query=tensor([[0.,0.,2.,2.]])
    close(math.iou(anchors,query,mode),[[1.,0.]])


def test_repository_trapezoid_integer_input_cast(source_math):
    close(source_math.trapz(tensor([1,3,5],np.int32),dx=.5),3.)


def test_fft_resize_unspecified_length_preserves_input():
    values=tensor([1.,2.,3.])
    assert math._resize_input(values,1,[0],[-1],[3]) is values


@pytest.mark.parametrize('values,dim,index,expected',[
    ([2.,5.,8.],0,1,[5.]),([[2.,5.,8.]],1,2,[8.])])
def test_legacy_nested_selection_helper(values,dim,index,expected):
    selected=[]
    assert math.get(values,0,dim,index,selected)==0
    assert selected==expected


@pytest.mark.parametrize('probability',[0.,1.])
def test_bernoulli_probability_extremes_use_real_cpu_kernel(probability):
    values=tensor(np.ones((2,3)))
    close(math.bernoulli(values,p=probability,seed=92374),np.full((2,3),probability))
