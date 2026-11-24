# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Direct Python gradient rules with real CPU tensors and independent oracles."""
# Direct private-helper calls intentionally cover the internal rules under test.
# pylint: disable=protected-access
import math
import os
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, COOTensor, CSRTensor, context, ops
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops._grad_experimental import grad_inner_ops as rules
from mindspore.ops.operations import array_ops as A, math_ops as M, sparse_ops as S, nn_ops as N
from mindspore.ops._grad_experimental import grad_array_ops as ga
from mindspore.ops._grad_experimental import grad_math_ops as gm
from mindspore.ops._grad_experimental import grad_sparse_ops as gs
from mindspore.ops._grad_experimental import grad_sparse as gc
from mindspore.ops._grad_experimental import grad_nn_ops as gn
from mindspore.ops._grad_experimental import grad_implementations as gi
from mindspore.ops._grad_experimental import taylor_rule as tr
from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True, scope='module')
def cpu_context():
    previous = {k: context.get_context(k) for k in ('mode', 'device_target', 'pynative_synchronize')}
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU', pynative_synchronize=True)
    yield
    context.set_context(**previous)


def t(value, dtype=ms.float32):
    return Tensor(value, dtype)


def check(actual, expected, dtype=None, atol=2e-5):
    assert isinstance(actual, Tensor)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    if dtype is not None:
        assert actual.dtype == dtype
    np.testing.assert_allclose(actual.asnumpy(), expected, atol=atol, rtol=2e-5)


def pair(actual, expected, dtype=ms.float32):
    assert isinstance(actual, tuple) and len(actual) == len(expected)
    for a, e in zip(actual, expected):
        check(a, e, dtype)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name', ['exp', 'log', 'sin', 'cos', 'tan'])
def test_taylor_unary_third_derivative(name):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Taylor unary third derivative.
    Expectation: All four Taylor coefficients match analytic chain-rule derivatives through third order in
        float32.
    """
    x = np.array([0.3, 0.7], np.float32)
    a, b, c = 0.7, -0.2, 0.1
    inputs = t(np.stack([x, np.full_like(x, a), np.full_like(x, b), np.full_like(x, c)]))
    if name == 'exp':
        f = f1 = f2 = f3 = np.exp(x)
    elif name == 'log':
        f, f1, f2, f3 = np.log(x), 1 / x, -1 / x**2, 2 / x**3
    elif name == 'sin':
        f, f1, f2, f3 = np.sin(x), np.cos(x), -np.sin(x), -np.cos(x)
    elif name == 'cos':
        f, f1, f2, f3 = np.cos(x), -np.sin(x), -np.cos(x), np.sin(x)
    else:
        f = np.tan(x)
        f1 = 1 + f * f
        f2 = 2 * f * f1
        f3 = 2 * f1 * (1 + 3 * f * f)
    result = getattr(tr, 'taylor_' + name)(None)(inputs)
    check(result, np.stack([f, f1 * a, f2 * a * a + f1 * b, f3 * a**3 + 3 * f2 * a * b + f1 * c]), ms.float32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name', ['mul', 'div', 'realdiv'])
def test_taylor_binary(name):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Taylor binary.
    Expectation: Multiplication and division coefficients match the product and quotient recurrences for scalar
        and series inputs.
    """
    x, y = t([2.0, 3.0, 5.0, 7.0]), t([4.0, 2.0, 0.0, 0.0])
    fn = getattr(tr, 'taylor_' + name)(None)
    if name == 'mul':
        expected = [8, 16, 32, 58]
    else:
        # Quotient recurrence, independently from polynomial coefficients.
        expected = [0.5, 0.5, 0.75, 0.625]
    check(fn(x, y), expected, ms.float32)
    check(fn(x, t(2.0)), np.array([2.0, 3.0, 5.0, 7.0]) * (2 if name == 'mul' else 0.5))
    check(fn(t(2.0), y), [8, 4, 0, 0] if name == 'mul' else [0.5, -0.25, 0.25, -0.375])
    check(fn(t(3.0), t(2.0)), 6 if name == 'mul' else 1.5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('prim', [P.Add(), P.Sub(), 'Add'])
def test_taylor_add_sub(prim):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Taylor add sub.
    Expectation: Only the constant coefficient absorbs scalar operands, while series coefficients and factorial
        helpers stay correct.
    """
    fn = tr.taylor_add_or_sub(prim)
    sign = -1 if isinstance(prim, P.Sub) else 1
    check(fn(t([2.0, 3.0]), t(4.0)), [2 + sign * 4, 3])
    check(fn(t(4.0), t([2.0, 3.0])), [4 + sign * 2, sign * 3])
    check(fn(t(4.0), t(2.0)), 4 + sign * 2)
    check(fn(t([2.0, 3.0]), t([4.0, 5.0])), [2 + sign * 4, 3 + sign * 5])
    check(tr._factorial(5), [1, 1, 2, 6, 24, 120])
    check(tr._trans_scalar_inputs(2.0, t([1.0, 3.0]))[0], [2.0, 0.0])
    check(tr._trans_scalar_inputs(t([1.0, 3.0]), 2.0)[1], [2.0, 0.0])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_scalar_and_structure_bprops():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Scalar and structure bprops.
    Expectation: Scalar derivatives use analytic values; identity, reshape and dependency rules preserve or zero
        the correct cotangents.
    """
    assert gi.bprop_scalar_exp(2.0, 7.0, 3.0) == (21.0,)
    assert gi.bprop_scalar_uadd(2.0, 2.0, 3.0) == (3.0,)
    assert gi.bprop_scalar_usub(2.0, -2.0, 3.0) == (-3.0,)
    assert gi.bprop_scalar_log(2.0, 0.0, 3.0) == (1.5,)
    assert gi.bprop_scalar_calc(2.0, 3.0, True, 1.0) == (0.0, 0.0)
    assert gi.bprop_scalar_not(True, False, 1.0) == (False,)
    for fn in (gi.bprop_identity, gi.bprop_tensor_move):
        pair(fn(t([1, 2]), None, t([3, 4])), ([3, 4],))
    check(gi.bprop_array_to_scalar(t(2), 2, 7)[0], 7, ms.int32)
    check(gi.bprop_array_to_scalar(t(2), 2, 7.0)[0], 7.0, ms.float32)
    pair(gi.bprop_shape(t([1, 2]), None, None), ([0, 0],))
    assert gi.bprop_broadcast_shape((2, 1), (1, 3), None, None) == (0, 0)
    pair(gi.bprop_depend(t([1]), t([2]), None, t([3])), ([3], [0]))
    pair(gi.bprop_embed(t([1, 2]), None, None), ([0, 0],))
    pair(gi.bprop_load(t([1]), t([2]), None, t([3])), ([3], [0]))
    result = gi.bprop_reshape(t([[1, 2], [3, 4]]), (4,), None, t([4, 3, 2, 1]))
    check(result[0], [[4, 3], [2, 1]])
    assert result[1] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('adjoint', [True, False])
@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('complex_', [False, True])
def test_triangular_solve_rule(lower, adjoint, batch, complex_):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Triangular solve rule.
    Expectation: Matrix and right-hand-side cotangents match NumPy solves with conjugation and triangular
        masking.
    """
    a = np.array([[3.0, 0.0], [1.0, 2.0]])
    if not lower:
        a = a.T
    if complex_:
        a = a.astype(np.complex64) * (1 + 0.2j)
    b = np.array([[2.0], [5.0]], dtype=a.dtype)
    d = np.array([[0.5], [-2]], dtype=a.dtype)
    if batch:
        a, b, d = a[None], b[None], d[None]
    dtype = ms.complex64 if complex_ else ms.float32
    ah = np.swapaxes(a.conj(), -1, -2)
    out = np.linalg.solve(ah if adjoint else a, b)
    db = np.linalg.solve(a if adjoint else ah, d)
    da = -(out @ np.swapaxes(db.conj(), -1, -2) if adjoint else db @ np.swapaxes(out.conj(), -1, -2))
    da = np.tril(da) if lower else np.triu(da)
    result = gm.get_bprop_matrix_triangular_solve(M.MatrixTriangularSolve(lower=lower, adjoint=adjoint))(
        t(a, dtype), t(b, dtype), t(out, dtype), t(d, dtype)
    )
    pair(result, (da, db), dtype)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('adjoint', [False, True])
@pytest.mark.parametrize('batch', [False, True])
def test_matrix_solve_rule(adjoint, batch):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Matrix solve rule.
    Expectation: Both cotangents match independent batched or unbatched NumPy matrix-solve derivatives.
    """
    a, b, d = np.array([[3.0, 1.0], [0.0, 2.0]]), np.array([[1.0], [4.0]]), np.array([[2.0], [-1.0]])
    if batch:
        a, b, d = a[None], b[None], d[None]
    out = np.linalg.solve(np.swapaxes(a, -1, -2) if adjoint else a, b)
    db = np.linalg.solve(a if adjoint else np.swapaxes(a, -1, -2), d)
    da = -(out @ np.swapaxes(db, -1, -2) if adjoint else db @ np.swapaxes(out, -1, -2))
    pair(gm.get_bprop_matrix_solve(M.MatrixSolve(adjoint=adjoint))(t(a), t(b), t(out), t(d)), (da, db))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('under', [True, False])
@pytest.mark.parametrize('batch', [True, False])
def test_least_squares_rule(under, batch):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Least squares rule.
    Expectation: Matrix and right-hand-side cotangents match centered finite differences; the regularizer
        cotangent is float64 zero.
    """
    a = np.array([[2.0, 1.0], [1.0, 3.0], [1.0, -1.0]], np.float32)
    if under:
        a = a.T
    b = np.arange(1, a.shape[0] + 1, dtype=np.float32).reshape(-1, 1)
    d = np.arange(1, a.shape[1] + 1, dtype=np.float32).reshape(-1, 1) / 3
    if batch:
        a, b, d = a[None], b[None], d[None]
    l2 = 0.2

    def solve(mat, rhs):
        if under:
            return np.swapaxes(mat, -1, -2) @ np.linalg.solve(
                mat @ np.swapaxes(mat, -1, -2) + l2 * np.eye(mat.shape[-2]), rhs
            )
        return np.linalg.solve(
            np.swapaxes(mat, -1, -2) @ mat + l2 * np.eye(mat.shape[-1]), np.swapaxes(mat, -1, -2) @ rhs
        )

    out = solve(a, b)

    # Centered numerical derivative of the independent NumPy least-squares objective.
    def finite(arr, other, which):
        result = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            plus, minus = arr.copy(), arr.copy()
            plus[idx] += 0.001
            minus[idx] -= 0.001
            p = solve(plus, other) if which == 0 else solve(other, plus)
            m = solve(minus, other) if which == 0 else solve(other, minus)
            result[idx] = np.sum((p - m) * d) / 0.002
        return result

    got = gm.get_bprop_matrix_solve_ls(M.MatrixSolveLs())(t(a), t(b), t(l2, ms.float64), t(out), t(d))
    check(got[0], finite(a, b, 0), ms.float32, atol=8e-4)
    check(got[1], finite(b, a, 1), ms.float32, atol=8e-4)
    check(got[2], 0, ms.float64)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('upper', [True, False])
@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('dtype', [ms.float32, ms.float64])
def test_cholesky_solve_rule(upper, batch, dtype):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Cholesky solve rule.
    Expectation: Factor and right-hand-side derivatives match the independent upper or lower Cholesky formula in
        the input dtype.
    """
    factor = np.array([[2.0, 0.0], [0.5, 1.5]])
    if upper:
        factor = factor.T
    a = factor.T @ factor if upper else factor @ factor.T
    b, d = np.array([[1.0], [3.0]]), np.array([[2.0], [-1.0]])
    out = np.linalg.solve(a, b)
    db = np.linalg.solve(a, d)
    common = db @ out.T + out @ db.T
    df = -factor @ common if upper else -common @ factor
    if batch:
        factor, b, out, d, df, db = [v[None] for v in (factor, b, out, d, df, db)]
    pair(
        gm.get_bprop_cholesky_solve(M.CholeskySolve(upper=upper))(
            t(b, dtype), t(factor, dtype), t(out, dtype), t(d, dtype)
        ),
        (db, df),
        dtype,
    )


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('keep', [False, True])
def test_norm_and_nan_rules(keep):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Norm and nan rules.
    Expectation: Norm derivatives scale by the sensitivity, while nonfinite NanToNum inputs receive zero
        cotangents.
    """
    x = t([[3.0, 4.0], [0.0, 5.0]])
    out = t([[5.0], [5.0]] if keep else [5.0, 5.0])
    d = t([[2.0], [3.0]] if keep else [2.0, 3.0])
    got = gm.get_bprop_euclidean_norm(M.EuclideanNorm(keep_dims=keep))(x, 1, out, d)
    check(got[0], [[1.2, 1.6], [0.0, 3.0]])
    assert got[1] == 0
    check(
        gm.get_bprop_nan_to_num(M.NanToNum())(t([np.nan, np.inf, -np.inf, 2]), None, t([1, 2, 3, 4]))[0], [0, 0, 0, 4]
    )
    assert gm.renew_dim((2, 3, 4), -1) == (0, 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dtype', [ms.float32, ms.int16, ms.int32, ms.int64])
@pytest.mark.parametrize('dynamic', [False, True])
def test_broadcast_gradient_reductions(dtype, dynamic):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Broadcast gradient reductions.
    Expectation: Each operand receives sensitivity sums over its broadcast axes, including shifted dimensions
        and integer dtypes.
    """
    x, y, d = t(np.ones((2, 1)), dtype), t(np.ones((1, 3)), dtype), t(np.arange(6).reshape(2, 3), dtype)
    fn = gm.dyn_binop_grad_common if dynamic else gm.binop_grad_common
    pair(
        fn(x, y, d, d),
        (np.arange(6).reshape(2, 3).sum(1, keepdims=True), np.arange(6).reshape(2, 3).sum(0, keepdims=True)),
        dtype,
    )
    x, y, d = t(np.ones((2, 1, 2)), dtype), t(np.ones((1, 3, 2)), dtype), t(np.arange(12).reshape(2, 3, 2), dtype)
    fn = gm.dyn_binop_grad_common_with_shift if dynamic else gm.binop_grad_common_with_shift
    pair(
        fn(x, y, d, d, 1),
        (np.arange(12).reshape(2, 3, 2).sum(1, keepdims=True), np.arange(12).reshape(2, 3, 2).sum(0, keepdims=True)),
        dtype,
    )
    pair(gm.get_bprop_tensor_add(P.Add())(t([1, 2]), t([3, 4]), None, t([5, 6])), ([5, 6], [5, 6]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('scalar', [True, False])
def test_masked_fill_and_scatter_rules(scalar):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Masked fill and scatter rules.
    Expectation: Masked input positions receive zero; fill and scatter sources receive only the selected
        sensitivities.
    """
    x, mask, d = t([[1.0, 2.0], [3.0, 4.0]]), t([[True, False], [False, True]], ms.bool_), t([[2.0, 3.0], [5.0, 7.0]])
    value = 4.0 if scalar else t(4.0)
    got = ga.get_bprop_masked_select(P.MaskedFill())(x, mask, value, None, d)
    check(got[0], [[0, 3], [5, 0]], ms.float32)
    check(got[1], np.zeros((2, 2)))
    if scalar:
        assert got[2] == 0
    else:
        check(got[2], 9.0)
    got = ga.get_bprop_masked_scatter(A.MaskedScatter())(x, mask, t([8.0, 9.0, 10.0]), None, d)
    check(got[0], [[0, 3], [5, 0]])
    check(got[1], np.zeros((2, 2)), ms.bool_)
    check(got[2], [2, 7, 0])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_index_fill_and_matrix_diagonals():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Index fill and matrix diagonals.
    Expectation: Filled rows and replaced diagonals route sensitivities to the replacement values and zero the
        overwritten input positions.
    """
    x, d, indices = t([[1, 2], [3, 4]]), t([[2, 3], [5, 7]]), t([1], ms.int32)
    got = ga.get_bprop_index_fill(A.IndexFill())(x, 0, indices, t(6.0), None, d)
    check(got[0], [[2, 3], [0, 0]])
    check(got[3], 12.0)
    k, zero = t(0, ms.int32), t(0.0)
    got = ga.get_bprop_matrix_diag_part_v3(A.MatrixDiagPartV3())(x, k, zero, None, t([2.0, 7.0]))
    check(got[0], [[2, 0], [0, 7]])
    got = ga.get_bprop_matrix_set_diag_v3(A.MatrixSetDiagV3())(x, t([1.0, 4.0]), k, None, d)
    check(got[0], [[0, 3], [5, 0]])
    check(got[1], [2, 7])
    got = ga.get_bprop_matrix_band_part(A.MatrixBandPart())(x, 0, -1, None, d)
    check(got[0], [[2, 3], [0, 7]])
    with pytest.raises(ValueError, match='bad shape 2'):
        ga._raise_value_error('bad ', 'shape ', 2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name', ['max', 'min'])
def test_tensor_scatter_extrema_ties(name):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Tensor scatter extrema ties.
    Expectation: Tied extrema split sensitivity equally, while selected updates and untouched input positions
        retain their shares.
    """
    x, idx, upd, d = t([2.0, 4.0, 5.0]), t([[0], [1]], ms.int32), t([2.0, 3.0]), t([6.0, 8.0, 10.0])
    out = t([2.0, 4.0, 5.0] if name == 'max' else [2.0, 3.0, 5.0])
    got = getattr(ga, 'get_bprop_tensor_scatter_' + name)(None)(x, idx, upd, out, d)
    check(got[0], [3, 8, 10] if name == 'max' else [3, 0, 10])
    check(got[1], [[0], [0]], ms.int32)
    check(got[2], [3, 0] if name == 'max' else [3, 8])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dtype', [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.float32, ms.float64])
def test_segment_rules(dtype):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Segment rules.
    Expectation: Segment sum gathers sensitivities and segment mean divides them by the corresponding segment
        sizes.
    """
    x, ids, d = t([[2, 4], [4, 8], [5, 10]], dtype), t([0, 0, 1], ms.int32), t([[6, 10], [3, 4]], dtype)
    got = ga.get_bprop_segment_sum(A.SegmentSum())(x, ids, None, d)
    check(got[0], [[6, 10], [6, 10], [3, 4]], dtype)
    got = ga.get_bprop_segment_mean(A.SegmentMean())(x, ids, None, d)
    check(got[0], [[3, 5], [3, 5], [3, 4]], dtype)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('xshape,outshape', [((1, 3), (2, 3)), ((3,), (2, 3)), ((2, 3), (2, 3)), ((), (2, 3))])
def test_expand_rule(xshape, outshape):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Expand rule.
    Expectation: Broadcast axes reduce back to the original scalar or tensor shape, and shape metadata receives
        int64 zeros.
    """
    d = np.arange(6, dtype=np.float32).reshape(outshape)
    got = ga.get_bprop_expand(A.Expand())(t(np.ones(xshape)), t(outshape, ms.int64), None, t(d))
    expect = (
        d
        if xshape == outshape
        else d.sum(0) if xshape == (3,) else d.sum(0, keepdims=True) if xshape == (1, 3) else d.sum()
    )
    check(got[0], expect)
    check(got[1], [0, 0], ms.int64)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name', ['scatter_nd_update', 'scatter_non_aliasing_add_update', 'scatter_update'])
def test_scatter_update_rule(name):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Scatter update rule.
    Expectation: The input cotangent retains the sensitivity and update cotangents gather the indexed rows.
    """
    d = t([[2, 3], [5, 7]])
    idx = t([1] if name == 'scatter_update' else [[1]], ms.int32)
    got = getattr(ga, 'get_bprop_' + name)(None)(t([[0, 0], [0, 0]]), idx, t([[1, 2]]), None, d)
    check(got[0], [[2, 3], [5, 7]])
    check(got[2], [[5, 7]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize(
    'axis,scalar',
    [
        (0, True),
        (0, False),
        (1, False),
        (-1, False),
        pytest.param(
            1,
            True,
            marks=pytest.mark.xfail(
                strict=True, raises=ValueError, reason='Scalar indices produce an invalid transpose permutation'
            ),
        ),
        pytest.param(
            -1,
            True,
            marks=pytest.mark.xfail(
                strict=True, raises=ValueError, reason='Scalar indices produce wrong segment prefix shape'
            ),
        ),
    ],
)
def test_sparse_gather_rule(axis, scalar):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Sparse gather rule.
    Expectation: Axis-zero gradients retain sparse row metadata; other supported axes scatter repeated
        sensitivities into dense storage.
    """
    x = t(np.arange(6).reshape(2, 3))
    idx = t(1 if scalar else [1, 0, 1], ms.int32)
    outshape = np.take(x.asnumpy(), idx.asnumpy(), axis=axis).shape
    d = np.arange(np.prod(outshape), dtype=np.float32).reshape(outshape) + 1
    got = ga.get_bprop_sparse_gather_v2(None)(x, idx, axis, None, t(d))
    if axis == 0:
        assert got[0].dense_shape == (2, 3)
        check(got[0].indices, np.asarray(idx.asnumpy()).reshape(-1), ms.int32)
        check(got[0].values, d.reshape(-1, 3))
    else:
        expect = np.zeros((2, 3), np.float32)
        if scalar:
            expect[:, 1] += d
        else:
            for j, col in enumerate([1, 0, 1]):
                expect[:, col] += d[:, j]
        check(got[0], expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_embedding_unstack_random_and_unique_rules():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Embedding unstack random and unique rules.
    Expectation: Embedding offsets and unstack axes route sensitivities correctly; nondifferentiable random and
        shape inputs receive zeros.
    """
    got = ga.get_bprop_embedding_lookup(None)(
        t(np.zeros((4, 2))), t([[3, 1]], ms.int32), 1, None, t([[[2, 3], [5, 7]]])
    )
    assert got[0].dense_shape == (4, 2)
    check(got[0].indices, [2, 0], ms.int32)
    check(got[0].values, [[2, 3], [5, 7]])
    pair(
        ga.get_bprop_unstack(P.Unstack(axis=1))(t([[1, 2], [3, 4]]), None, (t([2, 5]), t([3, 7]))), ([[2, 3], [5, 7]],)
    )
    pair(ga.get_bprop_log_normal_reverse(None)(t([2, 3]), None, t([1, 1])), ([0, 0],))
    pair(ga.get_bprop_parameterized_truncated_normal(None)(*[t([2, 3])] * 5, None, None), ([0, 0],) * 5)
    assert ga.get_bprop_eye(None)(2, 3, ms.float32, None, None) == (0, 0, ms.float32)
    ds = (t([1, 2]), t([3, 4]), t([5, 6]))
    assert ga.get_bprop_coalesce(None)(None, None, None, None, ds) is ds


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=RuntimeError, reason='UniqueGrad has no CPU kernel')
def test_unique_rule_cpu_kernel_gap():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Unique rule CPU kernel gap.
    Expectation: The intended gradient gathers [5, 7, 5, 11]; missing UniqueGrad CPU support remains a strict
        expected failure.
    """
    x = t([2, 1, 2, 3], ms.int32)
    out = P.Unique()(x)
    got = ga.get_bprop_unique(P.Unique())(x, out, (t([5, 7, 11], ms.int32), t([0, 0, 0, 0], ms.int32)))
    check(got[0], [5, 7, 5, 11], ms.int32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('small', [True, False])
def test_scatter_add_axis_rule(small):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Scatter add axis rule.
    Expectation: The input cotangent is unchanged and update cotangents gather sensitivities along the
        configured axis.
    """
    idx = np.array([[1, 0]]) if small else np.array([[1, 0], [0, 1]])
    d = np.array([[2, 3], [5, 7]], np.float32)
    got = ga.get_bprop_scatter_add_with_axis(A.ScatterAddWithAxis(axis=1))(
        t(d), t(idx, ms.int32), t(np.ones(idx.shape)), None, t(d)
    )
    check(got[0], d)
    check(got[2], np.take_along_axis(d[: idx.shape[0]], idx, axis=1))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dims', [(2, 3), (1, 1), (2, 1), (1, 2), (2, 2, 3), (1, 1, 1), (2, 1, 1)])
@pytest.mark.parametrize('align', [False, True])
@pytest.mark.parametrize('tensor_size', [False, True])
def test_affinegrid_gpu_math_on_cpu(monkeypatch, dims, align, tensor_size):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Select the portable GPU Python rule; every arithmetic kernel stays on CPU.
    Expectation: The theta cotangent equals the independent coordinate design-matrix product across dimensions,
        alignment modes and size representations.
    """
    real_context = ga.context.get_context
    with monkeypatch.context() as patch:
        patch.setattr(ga.context, 'get_context', lambda key: 'GPU' if key == 'device_target' else real_context(key))
        fn = ga.get_bprop_affinegrid(A.AffineGrid(align_corners=align))
    ndim, n = len(dims), 2
    output_size = (n, 1) + dims
    size = t(output_size, ms.int32) if tensor_size else output_size
    theta = t(np.zeros((n, ndim, ndim + 1)))
    d = np.arange(n * np.prod(dims) * ndim, dtype=np.float32).reshape((n,) + dims + (ndim,)) / 10
    axes = [np.linspace(-1, 1, k) if k > 1 else np.zeros(1) for k in dims]
    if not align:
        axes = [v * (k - 1) / k for v, k in zip(axes, dims)]
    coords = np.meshgrid(*axes, indexing='ij')
    design = np.stack([*coords[::-1], np.ones(dims)], -1).reshape(-1, ndim + 1)
    expected = np.stack([d[i].reshape(-1, ndim).T @ design for i in range(n)])
    if tensor_size:
        # The graph executor converts a Shape tuple to Tensor at ReduceSum.
        # This eager boundary adapter supplies the same exact shape values.
        real_shape = ga.shape_op
        monkeypatch.setattr(ga, 'shape_op', lambda value: t(real_shape(value), ms.int32))
        real_reshape = ga.reshape

        def reshape_with_ready_shape(value, shape):
            # Direct eager rule calls need the graph executor's shape-value dependency.
            # Wait for the producer; forward the original Tensor to the real kernel.
            if isinstance(shape, Tensor):
                shape.asnumpy()
            return real_reshape(value, shape)

        monkeypatch.setattr(ga, 'reshape', reshape_with_ready_shape)
    got = fn(theta, size, None, t(d))
    check(got[0], expected, ms.float32, atol=1e-4)
    assert isinstance(got, tuple) and len(got) == 2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('division', [False, True])
@pytest.mark.parametrize('broadcast', [False, True])
def test_sparse_dense_cwise_rules(division, broadcast):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Sparse dense cwise rules.
    Expectation: Sparse values follow product or quotient derivatives, and dense cotangents accumulate correctly
        across broadcast coordinates.
    """
    idx = t([[0, 0], [0, 2], [1, 1]], ms.int64)
    vals, shape, d = t([2.0, 3.0, 5.0]), t([2, 3], ms.int64), t([7.0, 11.0, 13.0])
    dense_np = np.array([2.0, 4.0, 5.0]) if broadcast else np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    selected = dense_np[[0, 2, 1]] if broadcast else dense_np[[0, 0, 1], [0, 2, 1]]
    dx = d.asnumpy() / selected if division else d.asnumpy() * selected
    dd_vals = -d.asnumpy() * vals.asnumpy() / selected**2 if division else d.asnumpy() * vals.asnumpy()
    dd = np.zeros_like(dense_np)
    np.add.at(dd, [0, 2, 1] if broadcast else ([0, 0, 1], [0, 2, 1]), dd_vals)
    fn = gs.get_bprop_sparse_dense_cwise_div if division else gs.get_bprop_sparse_dense_cwise_mul
    got = fn(None)(idx, vals, shape, t(dense_np), None, d)
    assert isinstance(got, tuple) and len(got) == 4
    check(got[0], np.zeros((3, 2)), ms.int64)
    check(got[1], dx)
    check(got[2], [0, 0], ms.int64)
    check(got[3], dd)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_sparse_softmax_rule():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Sparse softmax rule.
    Expectation: Stored sparse values receive the rowwise softmax Jacobian-vector product, with zero derivative
        for singleton rows.
    """
    idx, shape = t([[0, 0], [0, 2], [1, 1]], ms.int64), t([2, 3], ms.int64)
    values, d = t([1.0, 2.0, 4.0]), t([2.0, 5.0, 7.0])
    p = np.array([1 / (1 + np.e), np.e / (1 + np.e), 1.0])
    got = gs.get_bprop_sparse_softmax(None)(idx, values, shape, t(p), d)
    expected = p * np.array([2 - np.dot(p[:2], [2, 5]), 5 - np.dot(p[:2], [2, 5]), 0])
    check(got[1], expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('module', [gs, gc])
def test_sparse_to_dense_and_dense_add_rules(module):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Sparse to dense and dense add rules.
    Expectation: Stored values gather dense sensitivities, defaults sum unstored positions, and dense-add
        operands retain their sensitivities.
    """
    idx, shape, vals = t([[0, 0], [1, 2]], ms.int64), t([2, 3], ms.int64), t([2.0, 4.0])
    d = t([[2, 3, 5], [7, 11, 13]])
    got = module.get_bprop_sparse_to_dense_v2(None)(idx, shape, vals, t(1.0), None, d)
    check(got[0], np.zeros((2, 2)), ms.int64)
    check(got[1], [0, 0], ms.int64)
    check(got[2], [2, 13])
    check(got[3], 26.0)
    got = gc.get_bprop_sparse_to_dense(None)(idx, vals, (2, 3), None, d)
    check(got[1], [2, 13])
    got = gc.get_bprop_sparse_tensor_dense_add(None)(idx, vals, shape, t(np.ones((2, 3))), None, d)
    check(got[1], [2, 13])
    check(got[3], d.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['sum', 'sum_n', 'sqrt', 'sqrt_n', 'mean_n'])
def test_sparse_segment_rules(kind):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Sparse segment rules.
    Expectation: Repeated sparse indices accumulate segment sensitivities with the sum, mean or square-root
        normalization factor.
    """
    x, idx, ids = t(np.ones((4, 2))), t([2, 0, 2], ms.int64), t([0, 0, 1], ms.int64)
    d = t([[2, 4], [6, 8]])
    names = {
        'sum': 'sparse_segment_sum',
        'sum_n': 'sparse_segment_sum_with_num_segments',
        'sqrt': 'sparse_segment_sqrt_n',
        'sqrt_n': 'sparse_segment_sqrt_n_with_num_segments',
        'mean_n': 'sparse_segment_mean_with_num_segments',
    }
    fn = getattr(gs, 'get_bprop_' + names[kind])(None)
    args = (x, idx, ids, t(2, ms.int32)) if kind.endswith('_n') else (x, idx, ids)
    factor = 2 if kind == 'mean_n' else np.sqrt(2) if kind.startswith('sqrt') else 1
    expect = np.zeros((4, 2), np.float32)
    expect[0] = np.array([2, 4]) / factor
    expect[2] = expect[0] + np.array([6, 8])
    got = fn(*args, None, d)
    check(got[0], expect)
    check(got[1], [0, 0, 0])
    check(got[2], [0, 0, 0])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('adj_s', [False, True])
@pytest.mark.parametrize('adj_d', [False, True])
def test_sparse_dense_matmul_rule(adj_s, adj_d):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Sparse dense matmul rule.
    Expectation: Sparse-value and dense-matrix cotangents match independent matrix products for both adjoint
        flags.
    """
    idx, vals, shape = t([[0, 0], [0, 2], [1, 1]], ms.int64), t([2.0, 3.0, 5.0]), (2, 3)
    a = np.array([[2.0, 0.0, 3.0], [0.0, 5.0, 0.0]], np.float32)
    mat = a.T if adj_s else a
    b = np.arange(mat.shape[1] * 2, dtype=np.float32).reshape(mat.shape[1], 2) + 1
    dense = b.T if adj_d else b
    d = np.arange(mat.shape[0] * 2, dtype=np.float32).reshape(mat.shape[0], 2) + 1
    da, db = d @ b.T, mat.T @ d
    if adj_s:
        da = da.T
    if adj_d:
        db = db.T
    got = gc.get_bprop_sparse_tensor_dense_matmul(P.SparseTensorDenseMatmul(adjoint_st=adj_s, adjoint_dt=adj_d))(
        idx, vals, shape, t(dense), None, t(d)
    )
    check(got[1], da[[0, 0, 1], [0, 2, 1]])
    check(got[3], db)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_coo_csr_constructor_and_accessor_rules(monkeypatch):
    # Graph-only Make* primitives are represented by their real public eager constructors.
    """Feature: Numerical Python gradient and Taylor rules.
    Description: COO CSR constructor and accessor rules.
    Expectation: Sparse cotangents retain shape and field dtypes, and constructor or accessor rules route the
        requested storage field.
    """
    monkeypatch.setattr(gc.F, 'make_coo_tensor', COOTensor)
    monkeypatch.setattr(gc.F, 'make_csr_tensor', CSRTensor)
    idx, vals, ptr, col = t([[0, 0], [1, 1]], ms.int32), t([2.0, 3.0]), t([0, 1, 2], ms.int32), t([0, 1], ms.int32)
    coo = COOTensor(idx, vals, (2, 2))
    csr = CSRTensor(ptr, col, vals, (2, 2))
    check(gc.bprop_make_coo_tensor(idx, vals, (2, 2), coo, coo)[1], [2, 3])
    check(gc.bprop_make_csr_tensor(ptr, col, vals, (2, 2), csr, csr)[2], [2, 3])
    for fn, obj, dout, field in [
        (gc.bprop_coo_tensor_get_indices, coo, idx, 'indices'),
        (gc.bprop_coo_tensor_get_values, coo, vals, 'values'),
        (gc.bprop_csr_tensor_get_indptr, csr, ptr, 'indptr'),
        (gc.bprop_csr_tensor_get_indices, csr, col, 'indices'),
        (gc.bprop_csr_tensor_get_values, csr, vals, 'values'),
    ]:
        got = fn(obj, None, dout)
        assert isinstance(got, tuple) and len(got) == 1
        assert got[0].shape == (2, 2)
        check(getattr(got[0], field), dout.asnumpy(), dout.dtype)
    for fn in (gc.get_bprop_csr2coo, gc.get_bprop_coo2csr):
        got = fn(None)(col, 2, None, None)
        check(got[0], [0, 0], ms.int32)
        assert got[1] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=RuntimeError, reason='CSR CPU arithmetic requires unavailable LLVM 12.0.1')
@pytest.mark.parametrize('divide', [False, True])
@pytest.mark.parametrize('column', [False, True])
def test_csr_elementwise_rules(divide, column, monkeypatch):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: CSR elementwise rules.
    Expectation: Intended value and dense cotangents match independent product or quotient formulas; unavailable
        LLVM kernels remain expected failures.
    """
    monkeypatch.setattr(gc.F, 'make_csr_tensor', CSRTensor)
    monkeypatch.setattr(gc.F, 'csr_mul', ops.csr_mul)
    monkeypatch.setattr(gc.F, 'csr_div', ops.csr_div)
    monkeypatch.setattr(gc.F, 'csr_reduce_sum', ops.csr_reduce_sum)
    ptr, idx, vals, d = t([0, 2, 3], ms.int32), t([0, 2, 1], ms.int32), t([2.0, 3.0, 5.0]), t([7.0, 11.0, 13.0])
    dense = np.array([[2.0], [4.0]]) if column else np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    sel = dense[[0, 0, 1], [0, 0, 0] if column else [0, 2, 1]]
    out = vals.asnumpy() / sel if divide else vals.asnumpy() * sel
    dv = d.asnumpy() / sel if divide else d.asnumpy() * sel
    ddval = -d.asnumpy() * vals.asnumpy() / sel**2 if divide else d.asnumpy() * vals.asnumpy()
    dd = np.zeros_like(dense)
    np.add.at(dd, ([0, 0, 1], [0, 0, 0] if column else [0, 2, 1]), ddval)
    fn = gc.get_bprop_csr_div if divide else gc.get_bprop_csr_mul
    got = fn(None)(ptr, idx, vals, (2, 3), t(dense), t(out), d)
    check(got[2], dv)
    check(got[4], dd)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=RuntimeError, reason='CSR CPU kernel requires unavailable LLVM 12.0.1')
def test_csr_reduce_mv_and_conversion_rules():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: CSR reduce mv and conversion rules.
    Expectation: Intended cotangents gather CSR positions and transpose the matvec; missing LLVM support remains
        a strict expected failure.
    """
    ptr, idx, vals = t([0, 2, 3], ms.int32), t([0, 2, 1], ms.int32), t([2.0, 3.0, 5.0])
    got = gc.get_bprop_csr_reduce_sum(None)(ptr, idx, vals, (2, 3), 1, None, t([[7.0], [11.0]]))
    check(got[2], [7, 7, 11])
    got = gc.get_bprop_csr_mv(None)(ptr, idx, vals, (2, 3), t([[2.0], [3.0], [4.0]]), None, t([[7.0], [11.0]]))
    check(got[2], [14, 28, 33])
    check(got[4], [[14], [55], [21]])
    shape, batch = t([2, 3], ms.int32), t([0, 3], ms.int32)
    d = t([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
    got = gc.bprop_csr_sparse_matrix_to_dense(shape, batch, ptr, idx, vals, None, d)
    check(got[4], [2, 5, 11])
    out = (shape, batch, ptr, idx, vals)
    got = gc.bprop_dense_to_csr_sparse_matrix(d, t([[0, 0], [0, 2], [1, 1]], ms.int32), out, out)
    check(got[0], [[2, 0, 3], [0, 5, 0]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_sparse_reorder_slice_fill_and_add_rules():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Sparse reorder slice fill and add rules.
    Expectation: Reordered, sliced, inserted and merged sparse entries route each sensitivity back to its
        original value or default.
    """
    idx, vals, shape = t([[1, 1], [0, 2], [0, 0]], ms.int64), t([2.0, 3.0, 5.0]), t([2, 3], ms.int64)
    out = S.SparseReorder()(idx, vals, shape)
    got = gs.get_bprop_sparse_reorder(None)(idx, vals, shape, out, (out[0], t([7.0, 11.0, 13.0])))
    check(got[1], [13, 11, 7])
    idx = t([[0, 0], [0, 2], [1, 1]], ms.int64)
    start, size = t([0, 1], ms.int64), t([2, 2], ms.int64)
    out = S.SparseSlice()(idx, vals, shape, start, size)
    got = gs.get_bprop_sparse_slice(None)(idx, vals, shape, start, size, out, (out[0], t([7.0, 11.0]), out[2]))
    check(got[1], [0, 7, 11])
    idx = t([[0, 1], [2, 0]], ms.int64)
    shape = t([3, 2], ms.int64)
    vals = t([2.0, 3.0])
    out = S.SparseFillEmptyRows()(idx, vals, shape, t(5.0))
    got = gs.get_bprop_sparsefillemptyrows(None)(
        idx, vals, shape, t(5.0), out, (out[0], t([7.0, 11.0, 13.0]), out[2], out[3])
    )
    check(got[1], [7, 13])
    check(got[3], 11.0)
    idx2 = t([[0, 1], [1, 1]], ms.int64)
    out = S.SparseAdd()(idx, vals, shape, idx2, t([4.0, 5.0]), shape, t(0.0))
    got = gc.get_bprop_sparse_add(None)(
        idx, vals, shape, idx2, t([4.0, 5.0]), shape, t(0.0), out, (out[0], t([7.0, 11.0, 13.0]), out[2])
    )
    check(got[1], [7, 13])
    check(got[4], [7, 11])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize(
    'is_grad',
    [
        pytest.param(
            False, marks=pytest.mark.xfail(strict=True, raises=TypeError, reason='Rule indexes scalar loss output')
        ),
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason='Rule returns first gradient row instead of Hessian vector product',
            ),
        ),
    ],
)
def test_sparse_softmax_loss_rule(is_grad):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Sparse softmax loss rule.
    Expectation: Intended loss and gradient derivatives retain full logits shape; the documented scalar and
        Hessian defects remain strict failures.
    """
    logits, labels = t([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]]), t([2, 0], ms.int32)
    op = P.SparseSoftmaxCrossEntropyWithLogits(is_grad=is_grad)
    out = op(logits, labels)
    if is_grad:
        dout = t(np.ones((2, 3)))
        expected = np.zeros((2, 3))
    else:
        dout = t(2.0)
        exp = np.exp(logits.asnumpy() - logits.asnumpy().max(1, keepdims=True))
        expected = exp / exp.sum(1, keepdims=True)
        expected[np.arange(2), [2, 0]] -= 1
    got = gn.get_bprop_sparse_softmax_cross_entropy_with_logits(op)(logits, labels, out, dout)
    check(got[0], expected)
    check(got[1], [0, 0], ms.int32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name', ['max', 'avg'])
@pytest.mark.parametrize('overlap', [False, True])
def test_fractional_pool_rule(name, overlap):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Fractional pool rule.
    Expectation: Max pooling routes sensitivity to maxima; average pooling divides it over each actual pooling
        region, including overlaps.
    """
    x = t(np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1))
    op = (
        N.FractionalMaxPool([1.0, 2.0, 2.0, 1.0], overlapping=overlap, deterministic=True, seed=1)
        if name == 'max'
        else N.FractionalAvgPool([1.0, 2.0, 2.0, 1.0], overlapping=overlap, deterministic=True, seed=1)
    )
    out = op(x)
    d = np.arange(1, 5, dtype=np.float32).reshape(1, 2, 2, 1)
    expected = np.zeros((1, 4, 4, 1), np.float32)
    rows, cols = out[1].asnumpy(), out[2].asnumpy()
    for i in range(2):
        for j in range(2):
            end_i, end_j = min(rows[i + 1] + overlap, 4), min(cols[j + 1] + overlap, 4)
            block = x.asnumpy()[0, rows[i] : end_i, cols[j] : end_j, 0]
            if name == 'max':
                r, c = np.unravel_index(np.argmax(block), block.shape)
                expected[0, rows[i] + r, cols[j] + c, 0] += d[0, i, j, 0]
            else:
                expected[0, rows[i] : end_i, cols[j] : end_j, 0] += d[0, i, j, 0] / block.size
    fn = gn.get_bprop_fractional_max_pool if name == 'max' else gn.get_bprop_fractional_avg_pool
    got = fn(op)(x, out, (t(d), ops.zeros_like(out[1]), ops.zeros_like(out[2])))
    pair(got, (expected,))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize(
    'dim',
    [
        pytest.param(
            2,
            marks=pytest.mark.xfail(
                strict=True, raises=TypeError, reason='2D rule supplies x where CPU gradient requires input shape'
            ),
        ),
        3,
    ],
)
def test_fractional_fixed_pool_rule(dim):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Fractional fixed pool rule.
    Expectation: Sensitivities scatter to reported maximum indices and random samples receive zeros; the 2D API
        gap remains expected.
    """
    shape = (1, 1) + (4,) * dim
    x = t(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    op = N.FractionalMaxPoolWithFixedKsize(2, (2, 2)) if dim == 2 else N.FractionalMaxPool3DWithFixedKsize(2, (2, 2, 2))
    samples = t([[[0.2, 0.5] if dim == 2 else [0.2, 0.5, 0.7]]])
    out = op(x, samples)
    d = np.arange(1, np.prod(out[0].shape) + 1, dtype=np.float32).reshape(out[0].shape)
    expected = np.zeros(shape, np.float32)
    for index, value in zip(out[1].asnumpy().ravel(), d.ravel()):
        expected.ravel()[index] += value
    fn = (
        gn.get_bprop_fractional_max_pool_with_fixed_ksize
        if dim == 2
        else gn.get_bprop_fractional_max_pool3d_with_fixed_ksize
    )
    got = fn(op)(x, samples, out, (t(d), ops.zeros_like(out[1])))
    check(got[0], expected)
    check(got[1], np.zeros(samples.shape))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_adaptive_and_average_pool_rules():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Adaptive and average pool rules.
    Expectation: The adaptive 3D maximum receives sensitivity seven and the output-size tensor receives int32
        zeros.
    """
    x = t(np.arange(27, dtype=np.float32).reshape(1, 1, 3, 3, 3))
    size = t([1, 1, 1], ms.int32)
    out = N.AdaptiveMaxPool3D()(x, size)
    got = gn.get_bprop_adaptive_max_pool_3d(None)(x, size, out, (t([[[[[7.0]]]]]), ops.zeros_like(out[1])))
    expected = np.zeros(x.shape, np.float32)
    expected.ravel()[-1] = 7
    check(got[0], expected)
    check(got[1], [0, 0, 0], ms.int32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=RuntimeError, reason='AvgPoolGradV1 has no CPU kernel')
def test_average_pool_rule():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Average pool rule.
    Expectation: The intended derivative distributes each sensitivity evenly over its 2x2 window; the absent CPU
        kernel remains expected.
    """
    x = t(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4))
    op = N.AvgPoolV1(kernel_size=2, strides=2)
    got = gn.get_bprop_avg_pool_v1_grad(op)(x, None, t([[[[4, 8], [12, 16]]]]))
    pair(got, (np.array([[[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]]]),))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('norm', ['backward', 'forward', 'ortho'])
@pytest.mark.parametrize(
    'real,onesided,shape',
    [
        (False, True, (4,)),
        (True, False, (4,)),
        (True, True, (4,)),
        (True, True, (5,)),
        (True, True, (2, 4)),
        (True, True, (2, 2, 4)),
    ],
)
def test_fft_rule(inverse, norm, real, onesided, shape, request):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: FFT rule.
    Expectation: The input cotangent matches the transpose of an independent NumPy real-linear FFT Jacobian for
        each normalization and shape.
    """
    rank = len(shape)
    if real and onesided and rank == 3 and not inverse and norm != 'backward':
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True, raises=AssertionError, reason='3D rFFT rule applies complex FFT normalization twice'
            )
        )
    fn = gm.get_bprop_fft_with_size(None)
    rng = np.random.default_rng(9)

    # Build independent real-linear Jacobian columns with NumPy FFT, then apply its transpose.
    def transform(a):
        if not real:
            return (np.fft.ifftn if inverse else np.fft.fftn)(a, norm=norm)
        if inverse:
            if onesided:
                return np.fft.irfftn(a, s=shape, axes=tuple(range(rank)), norm=norm)
            return np.fft.ifftn(a, norm=norm).real
        return np.fft.rfftn(a, norm=norm) if onesided else np.fft.fftn(a, norm=norm)

    input_shape = shape[:-1] + (shape[-1] // 2 + 1,) if real and inverse and onesided else shape
    input_complex = not real or inverse
    x = rng.normal(size=input_shape)
    if input_complex:
        x = x + 1j * rng.normal(size=input_shape)
    out = transform(x)
    d = rng.normal(size=out.shape)
    if np.iscomplexobj(out):
        d = d + 1j * rng.normal(size=out.shape)
    expected = np.zeros(input_shape, np.complex64 if input_complex else np.float32)
    for idx in np.ndindex(input_shape):
        basis = np.zeros(input_shape, np.complex64 if input_complex else np.float32)
        basis[idx] = 1
        value = np.real(np.sum(np.conj(d) * transform(basis)))
        if input_complex:
            basis[idx] = 1j
            value += 1j * np.real(np.sum(np.conj(d) * transform(basis)))
        expected[idx] = value
    dtype = ms.complex64 if input_complex else ms.float32
    outdtype = ms.complex64 if np.iscomplexobj(out) else ms.float32
    got = fn(
        t(x, dtype),
        rank,
        inverse,
        real,
        ['backward', 'forward', 'ortho'].index(norm),
        onesided,
        shape if inverse else (),
        t(out, outdtype),
        t(d, outdtype),
    )
    assert isinstance(got, tuple) and len(got) == 7
    check(got[0], expected, dtype, atol=1e-4)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_fft_shape_helpers():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: FFT shape helpers.
    Expectation: FFT helper shapes retain their required leading unit dimensions and an unknown normalization
        enum maps to an empty string.
    """
    assert gm._rfft_reshape((2, 3, 4, 5), (3, 3)) == (1, 1, 3, 3)
    assert gm._rfft_tile_reshape((2, 3, 4, 5)) == (2, 3, 1, 1)
    assert gm._rfft_last_term_shape((2, 3, 4), [5]) == (1, 1, 5)
    assert gm._batch_matmul_shape_increase((3, 4)) == (1, 3, 4)
    assert gm._batch_matmul_shape_decrease((2, 3, 4)) == (2, 3, 1)
    assert gm._norm_enum_to_string(7) == ''


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_csr_matrix_conversion_rules():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: CSR matrix conversion rules.
    Expectation: Dense, COO and CSR conversions gather or scatter cotangents at stored positions and sparse
        transpose reverses itself.
    """
    ptr, idx, vals = t([0, 2, 3], ms.int32), t([0, 2, 1], ms.int32), t([2.0, 3.0, 5.0])
    shape, batch = t([2, 3], ms.int32), t([0, 3], ms.int32)
    coo_idx = t([[0, 0], [0, 2], [1, 1]], ms.int32)
    d = t([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
    got = gc.bprop_csr_sparse_matrix_to_dense(shape, batch, ptr, idx, vals, None, d)
    check(got[4], [2, 5, 11])
    out = (shape, batch, ptr, idx, vals)
    got = gc.bprop_dense_to_csr_sparse_matrix(d, coo_idx, out, out)
    check(got[0], [[2, 0, 3], [0, 5, 0]])
    got = gs.get_bprop_sparse_tensor_to_csr_sparse_matrix(None)(coo_idx, vals, shape, out, out)
    check(got[0], [[0, 0], [0, 2], [1, 1]], ms.int32)
    check(got[1], [2, 3, 5])
    got = gs.get_bprop_csr_sparse_matrix_to_sparse_tensor(None)(*out, None, (coo_idx, vals, shape))
    for actual, expected in zip(got, out):
        check(actual, expected.asnumpy(), expected.dtype)
    # Transposition is an involution, so the derivative transposes the cotangent back.
    forward = S.SparseMatrixTranspose(conjugate=False)(*out)
    got = gc.get_bprop_sparse_matrix_transpose(S.SparseMatrixTranspose(conjugate=False))(*out, forward, forward)
    for actual, expected in zip(got, out):
        check(actual, expected.asnumpy(), expected.dtype)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=TypeError, reason='Eager ZerosLike does not accept sparse tensors')
@pytest.mark.parametrize('kind', ['coo', 'csr'])
def test_sparse_dense_shape_gradient_graph_boundary(kind):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Sparse dense shape gradient graph boundary.
    Expectation: The intended shape cotangent has zero sparse values; eager sparse ZerosLike rejection remains a
        strict failure.
    """
    vals = t([2.0, 3.0])
    if kind == 'coo':
        obj = COOTensor(t([[0, 0], [1, 1]], ms.int32), vals, (2, 2))
        fn = gc.bprop_coo_tensor_get_dense_shape
    else:
        obj = CSRTensor(t([0, 1, 2], ms.int32), t([0, 1], ms.int32), vals, (2, 2))
        fn = gc.bprop_csr_tensor_get_dense_shape
    check(fn(obj, None, None)[0].values, [0, 0])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('exclusive', [False, True])
def test_cumulative_logsumexp_rule(dtype, reverse, exclusive):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Cumulative logsumexp rule.
    Expectation: Prefix or suffix softmax weights accumulate the signed sensitivities, including exclusive
        endpoints and float16 tolerances.
    """
    x = np.array([0.3, -0.5, 0.7, 0.9])
    d = np.array([2.0, -3.0, 0.0, 5.0])
    expected = np.zeros(4)
    out = np.empty(4)
    for i in range(4):
        selected = np.arange(i + exclusive, 4) if reverse else np.arange(0, i + 1 - exclusive)
        if selected.size == 0:
            out[i] = -np.inf
            continue
        weights = np.exp(x[selected])
        weights /= weights.sum()
        out[i] = np.log(np.exp(x[selected]).sum())
        expected[selected] += d[i] * weights
    got = gm.get_brop_cumulative_logsumexp(M.CumulativeLogsumexp(exclusive=exclusive, reverse=reverse))(
        t(x, dtype), t(0, ms.int32), t(out, dtype), t(d, dtype)
    )
    check(got[0], expected, dtype, atol=0.012 if dtype == ms.float16 else 2e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('scalar', [False, True])
def test_euclidean_scalar_and_mvlgamma(scalar):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Euclidean scalar and mvlgamma.
    Expectation: The scalar norm preserves its sensitivity and multivariate log-gamma matches the independent
        digamma formula.
    """
    if scalar:
        got = gm.get_bprop_euclidean_norm(M.EuclideanNorm())(t(3.0), t([], ms.int32), t(3.0), t(2.0))
        check(got[0], 2.0)
    else:
        # psi(n) = harmonic(n-1) - Euler's constant, for integer n.
        x = np.array([2.0, 3.0], np.float32)
        expected = np.array([1 - np.euler_gamma, 1.5 - np.euler_gamma]) * [2, 3]
        got = ga.get_bprop_mvlgamma(A.Mvlgamma(p=1))(t(x), None, t([2.0, 3.0]))
        check(got[0], expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('align', [False, True])
def test_resize_nearest_rule(align):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Resize nearest rule.
    Expectation: Each input pixel receives the sum of sensitivities from its corresponding upsampled 2x2 block.
    """
    x = t([[[[1.0, 2.0], [3.0, 4.0]]]])
    op = A.ResizeNearestNeighborV2(align_corners=align)
    size = t([4, 4], ms.int32)
    d = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    got = ga.get_bprop_resize_nearest_neighbor_v2(op)(x, size, None, t(d))
    expected = d.reshape(1, 1, 2, 2, 2, 2).sum((3, 5))
    check(got[0], expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('accumulate', [0, 1])
def test_index_put_broadcast_values(accumulate):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Index put broadcast values.
    Expectation: Overwritten input positions are zeroed unless accumulating, and a broadcast replacement
        receives the summed sensitivity.
    """
    x, values, idx = t(np.zeros((2, 3))), t([5.0]), [t([0, 1], ms.int32), t([1, 2], ms.int32)]
    d = np.arange(6, dtype=np.float32).reshape(2, 3)
    got = ga.get_bprop_index_put(A.IndexPut(accumulate=accumulate))(x, values, idx, None, t(d))
    expected = d.copy()
    if not accumulate:
        expected[[0, 1], [1, 2]] = 0
    check(got[0], expected)
    check(got[1], [6.0])
    assert len(got[2]) == 2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='AffineGrid GPU rule returns coordinates instead of zero output-size gradient',
)
def test_affinegrid_size_gradient_contract(monkeypatch):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: AffineGrid size gradient contract.
    Expectation: The intended output-size cotangent is integer zero; the documented GPU-rule metadata defect
        remains a strict failure.
    """
    real_context = ga.context.get_context
    with monkeypatch.context() as patch:
        patch.setattr(ga.context, 'get_context', lambda key: 'GPU' if key == 'device_target' else real_context(key))
        fn = ga.get_bprop_affinegrid(A.AffineGrid(align_corners=False))
    got = fn(t(np.zeros((1, 2, 3))), (1, 1, 2, 2), None, t(np.ones((1, 2, 2, 2))))
    assert isinstance(got[1], int) and got[1] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_scalar_pow_rule():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Scalar pow rule.
    Expectation: Base and exponent cotangents equal 24 and 16 times log(2), respectively.
    """
    dx, dy = gi.bprop_scalar_pow(2.0, 3.0, 8.0, 2.0)
    assert dx == 24.0
    assert dy == pytest.approx(16 * math.log(2))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(
    not hasattr(np, 'cast'), strict=True, raises=AttributeError, reason='ScalarCast calls np.cast removed by NumPy 2'
)
def test_scalar_cast_numpy_compatibility():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Scalar cast numpy compatibility.
    Expectation: The derivative returns the converted sensitivity and dtype; NumPy versions without cast retain
        the strict compatibility failure.
    """
    got = gi.bprop_scalar_cast(2, ms.float64, 2.0, 3.0)
    assert got == (3.0, ms.float64)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_least_squares_rejects_slow_gradient():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Least squares rejects slow gradient.
    Expectation: Requesting a gradient for fast=False raises the documented ValueError.
    """
    fn = gm.get_bprop_matrix_solve_ls(M.MatrixSolveLs(fast=False))
    with pytest.raises(ValueError, match='gradient not defined for fast=False'):
        fn(t([[2.0]]), t([[1.0]]), t(0.0, ms.float64), t([[0.5]]), t([[1.0]]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('accumulate', [0, 1])
def test_index_put_partial_indices(accumulate):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Index put partial indices.
    Expectation: Replacement cotangents sum over selected rows, while input cotangents retain or zero those rows
        according to accumulation.
    """
    x, values, idx = t(np.zeros((2, 3))), t([2.0, 3.0, 5.0]), [t([0, 1], ms.int32)]
    d = np.arange(6, dtype=np.float32).reshape(2, 3)
    got = ga.get_bprop_index_put(A.IndexPut(accumulate=accumulate))(x, values, idx, None, t(d))
    check(got[0], d if accumulate else np.zeros((2, 3)))
    check(got[1], d.sum(0))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('axis', [0, -1])
def test_sparse_gather_vector_scalar_output(axis):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Sparse gather vector scalar output.
    Expectation: The selected scalar sensitivity becomes sparse row [1] or dense values [0, 7, 0] according to
        the axis.
    """
    x, idx = t([2.0, 3.0, 5.0]), t(1, ms.int32)
    got = ga.get_bprop_sparse_gather_v2(None)(x, idx, axis, None, t(7.0))
    if axis == 0:
        check(got[0].indices, [1], ms.int32)
        check(got[0].values, [7.0])
    else:
        check(got[0], [0.0, 7.0, 0.0])


@pytest.fixture(name='allocated_ascend')
def allocated_ascend_fixture():
    device = os.environ.get('MS_GRAD_RULE_ASCEND_DEVICE')
    if device is None:
        pytest.skip('Set MS_GRAD_RULE_ASCEND_DEVICE to an allocated Ascend device')
    previous = {key: context.get_context(key) for key in ('device_target', 'device_id', 'mode', 'pynative_synchronize')}
    try:
        context.set_context(
            device_target='Ascend', device_id=int(device), mode=context.PYNATIVE_MODE, pynative_synchronize=True
        )
        yield
    finally:
        context.set_context(**previous)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard')
def test_ascend_depthwise_rule(allocated_ascend):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Ascend depthwise rule.
    Expectation: On an allocated Ascend device, input and weight gradients match the explicit
        depthwise-convolution arrays in float16.
    """
    op = P.DepthwiseConv2dNative(1, 2)
    x, w = t(np.arange(9).reshape(1, 1, 3, 3), ms.float16), t([[[[1.0, 2.0], [3.0, 4.0]]]], ms.float16)
    out = op(x, w)
    got = gn.get_bprop_depthwise_conv2d_native(op)(x, w, out, ops.ones_like(out))
    pair(got, (np.array([[[[1, 3, 2], [4, 10, 6], [3, 7, 4]]]]), np.array([[[[8, 12], [20, 24]]]])), ms.float16)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard')
@pytest.mark.parametrize('align', [False, True])
def test_ascend_parallel_resize_rule(allocated_ascend, align):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Ascend parallel resize rule.
    Expectation: On an allocated Ascend device, identity-sized resize retains its sensitivity and returns zero
        for size metadata.
    """
    op = inner.ParallelResizeBilinear([2, 2], [2, 2], 0, 0, align)
    x, d = t([[[[1.0, 2.0], [3.0, 4.0]]]]), t([[[[2.0, 3.0], [5.0, 7.0]]]])
    got = rules.get_bprop_parallel_resize_bilinear(op)(x, (2, 2), None, d)
    check(got[0], d.asnumpy(), ms.float32)
    assert got[1] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('align', [False, True])
@pytest.mark.parametrize('dims', [(2, 3), (1, 1), (2, 2, 3)])
def test_affinegrid_native_cpu_rule(align, dims):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: AffineGrid native CPU rule.
    Expectation: The theta gradient matches the coordinate design-matrix product and output-size metadata
        receives zero.
    """
    ndim = len(dims)
    theta = t(np.zeros((1, ndim, ndim + 1)))
    d = np.arange(np.prod(dims) * ndim, dtype=np.float32).reshape((1,) + dims + (ndim,)) / 10
    axes = [np.linspace(-1, 1, k) if k > 1 else np.zeros(1) for k in dims]
    if not align:
        axes = [v * (k - 1) / k for v, k in zip(axes, dims)]
    coords = np.meshgrid(*axes, indexing='ij')
    design = np.stack([*coords[::-1], np.ones(dims)], -1).reshape(-1, ndim + 1)
    expected = (d.reshape(-1, ndim).T @ design)[None]
    got = ga.get_bprop_affinegrid(A.AffineGrid(align_corners=align))(theta, (1, 1) + dims, None, t(d))
    assert isinstance(got, tuple) and len(got) == 2
    check(got[0], expected, ms.float32)
    assert got[1] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='CumulativeLogsumexp returns x-shaped float zero for its scalar integer axis',
)
def test_cumulative_logsumexp_axis_gradient_contract():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Cumulative logsumexp axis gradient contract.
    Expectation: The intended axis cotangent is scalar int32 zero; the documented shape and dtype defect remains
        a strict failure.
    """
    x, axis, out, d = t([0.0, 0.0]), t(0, ms.int32), t([0.0, math.log(2)]), t([1.0, 1.0])
    got = gm.get_brop_cumulative_logsumexp(M.CumulativeLogsumexp())(x, axis, out, d)
    check(got[1], 0, ms.int32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason='MaskedFill rule casts mask before constructing its zero cotangent'
)
def test_masked_fill_mask_gradient_dtype():
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Masked fill mask gradient dtype.
    Expectation: The intended mask cotangent contains boolean zeros; the documented dtype defect remains a
        strict failure.
    """
    got = ga.get_bprop_masked_select(P.MaskedFill())(
        t([1.0, 2.0]), t([True, False], ms.bool_), t(3.0), None, t([2.0, 3.0])
    )
    check(got[1], [False, False], ms.bool_)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('container', [tuple, list])
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason='Ragged rule zeros enumerate pairs instead of split tensors'
)
def test_ragged_split_gradient_contract(container):
    """Feature: Numerical Python gradient and Taylor rules.
    Description: Ragged split gradient contract.
    Expectation: The intended split cotangents retain their container and int64 zero tensors; the documented
        split-enumeration defect remains expected.
    """
    splits, values = container([t([0, 2], ms.int64)]), t([1.0, 2.0])
    got = gs.get_bprop_ragged_tensor_to_sparse(S.RaggedTensorToSparse(Tsplits=ms.int64))(
        splits, values, None, (None, t([3.0, 5.0]), None)
    )
    check(got[1], [3.0, 5.0])
    assert isinstance(got[0], container) and len(got[0]) == 1
    check(got[0][0], [0, 0], ms.int64)
