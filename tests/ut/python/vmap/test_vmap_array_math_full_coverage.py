# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Direct eager checks of array, math, gradient and sparse batching contracts.

Only the graph-only general-preprocess boundary is adapted for eager execution.
Every transformation and numerical primitive under test executes on real Tensors.
"""
import numpy as np
import pytest
import mindspore as ms
from mindspore import ops
from mindspore._c_expression import UMonad
from mindspore.ops import operations as P
from mindspore.ops import auto_generate as gen
from mindspore.ops.operations import math_ops as M, array_ops as A, _grad_ops as G
from mindspore.ops.primitive import Primitive
from mindspore.ops._vmap import vmap_base as B
from mindspore.ops._vmap import vmap_array_ops as V
from mindspore.ops._vmap import vmap_math_ops as W
from mindspore.ops._vmap import vmap_grad_math_ops as D
from mindspore.ops._vmap import vmap_sparse_ops as S


def _eager_preprocess(prim, *args):
    """Implement the complete all-unmapped/otherwise contract in eager mode."""
    if isinstance(prim, str):
        prim = Primitive(prim)
    if not callable(prim):
        raise TypeError("preprocess requires a callable primitive or registered name")
    def is_leaf(x):
        return isinstance(x, tuple) and len(x) == 2 and (x[1] is None or isinstance(x[1], int))
    def unmapped(x):
        return x[1] is None if is_leaf(x) else all(unmapped(y) for y in x)
    def unwrap(x):
        return x[0] if is_leaf(x) else tuple(unwrap(y) for y in x)
    def bind(x):
        return tuple(bind(y) for y in x) if isinstance(x, tuple) else (x, None)
    if all(unmapped(x) for x in args):
        return True, bind(prim(*(unwrap(x) for x in args)))
    return False, None


@pytest.fixture(autouse=True)
def eager_cpu(monkeypatch):
    old = {k: ms.get_context(k) for k in ('mode', 'device_target')}
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    for module in (B, V, W, D, S):
        monkeypatch.setattr(module, 'vmap_general_preprocess', _eager_preprocess)
    yield
    ms.set_context(**old)


def t(x, dtype=np.float32):
    return ms.Tensor(np.asarray(x, dtype=dtype))


def check(result, expected, axis=0):
    value, dim = result
    assert dim == axis
    np.testing.assert_allclose(value.asnumpy(), expected, rtol=2e-5, atol=2e-6, equal_nan=True)


@pytest.mark.parametrize('xd,yd', [(0, 0), (1, 0), (None, 0), (0, None), (None, None)])
def test_binary_broadcast(xd, yd):
    x = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    y = x + 1
    xa = x[0] if xd is None else np.moveaxis(x, 0, xd)
    ya = y[0] if yd is None else np.moveaxis(y, 0, yd)
    expected = xa + ya if xd == yd else (x[0] if xd is None else x) + (y[0] if yd is None else y)
    check(W.get_broadcast_binary_op_vmap_rule(P.Add(), 2)((t(xa), xd), (t(ya), yd)), expected,
          xd if xd == yd else 0)


def test_binary_rank_and_scalar_broadcast():
    x = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    check(W.get_broadcast_binary_op_vmap_rule(P.Add(), 2)((t(x), 0), (t([1, 2, 3]), None)),
          x + [1, 2, 3])
    check(W.get_broadcast_binary_op_vmap_rule(P.Mul(), 2)((t(2), None), (t(x), 0)), x * 2)
    assert W._broadcast_shape(4, 2, (2, 3)) == (2, 3, 1, 1)


@pytest.mark.parametrize('name', ['Addcmul', 'Addcdiv'])
@pytest.mark.parametrize('align', [True, False])
def test_addcxxx(name, align):
    x = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    val = np.full_like(x, 2) if align else np.full((3,), 2, np.float32)
    inputs = ((t(x), 0), (t(x + 1), 0), (t(x + 2), 0), (t(val), 0 if align else None))
    expected = x + (x + 1) * (x + 2) * 2 if name == 'Addcmul' else x + (x + 1) / (x + 2) * 2
    check(W.get_addcxxx_vmap_rule(getattr(P, name)(), 2)(*inputs), expected)


@pytest.mark.parametrize('weight_kind', ['scalar', 'tensor', 'broadcast_scalar', 'broadcast_tensor'])
def test_lerp(weight_kind):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    broadcast = weight_kind.startswith('broadcast')
    y = x + 10 if not broadcast else np.array([11, 12, 13], np.float32)
    weight = t(np.full_like(x, .25)) if weight_kind.endswith('tensor') else .25
    weight_dim = 0 if isinstance(weight, ms.Tensor) else None
    check(W.get_lerp_vamp_rule(M.Lerp(), 2)((t(x), 0), (t(y), None if broadcast else 0),
                                         (weight, weight_dim)), x + .25 * (y - x))


@pytest.mark.parametrize('prob', [0., 1.])
@pytest.mark.parametrize('tensor_prob', [False, True])
def test_bernoulli_endpoints(prob, tensor_prob):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    p = t(np.full_like(x, prob)) if tensor_prob else prob
    check(W.get_bernoulli_op_vmap_rule(M.Bernoulli(seed=17), 2)((t(x), 0), (p, 0 if tensor_prob else None)),
          np.full_like(x, prob))


def test_addn_stack_concat_unstack():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    inputs = ((t(x), 0), (t([10, 20, 30]), None))
    check(W.get_add_n_vmap_rule('AddN', 2)(inputs), x + [10, 20, 30])
    check(V.get_stack_vmap_rule('Stack', 2)(inputs), np.stack((x, np.tile([10, 20, 30], (2, 1))), axis=1))
    check(V.get_concat_vmap_rule(Primitive('Concat'), 2)(inputs, (-1, None)), np.concatenate((x, np.tile([10, 20, 30], (2, 1))), axis=-1))
    for out, exp in zip(V.get_unstack_vmap_rule('Unstack', 2)((t(x), 0)), x.T):
        check(out, exp)


@pytest.mark.parametrize('xd', [0, 1, None])
def test_matmul_and_solve(xd):
    a = np.array([[[2, 0], [0, 4]], [[3, 0], [0, 6]]], np.float32)
    xa = a[0] if xd is None else np.moveaxis(a, 0, xd)
    rhs = np.array([[[4], [8]], [[3], [12]]], np.float32)
    sample_a = np.broadcast_to(a[0], a.shape) if xd is None else a
    check(W.get_matmul_vmap_rule('MatMul', 2)((t(xa), xd), (t(rhs), 0), (False, None), (False, None)),
          np.matmul(sample_a, rhs))
    check(W.get_matrix_solve_vmap_rule('MatrixSolve', 2)((t(xa), xd), (t(rhs), 0)),
          np.linalg.solve(sample_a, rhs))


@pytest.mark.parametrize('axis', [0, 1, -1])
@pytest.mark.parametrize('keep', [False, True])
def test_reduce_and_arg_extrema(axis, keep):
    x = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    logical = np.moveaxis(x, 1, 0)
    actual = axis % 2
    out_dim = 1 if actual == 1 or keep else 0
    expected = np.mean(x, axis=actual if actual < 1 else actual + 1, keepdims=keep)
    check(W.get_reduce_vmap_rule(Primitive('ReduceMean'), 2)((t(x), 1), (axis, None), (keep, None)), expected, out_dim)
    check(W.get_reduce_sum_vmap_rule(Primitive('ReduceSum'), 2)((t(x), 1), (axis, None), (keep, None), (False, None)),
          np.sum(x, axis=actual if actual < 1 else actual + 1, keepdims=keep), out_dim)
    result = V.get_arg_min_max_with_value_vmap_rule(P.ArgMaxWithValue(), 2)((t(x), 1), (axis, None), (keep, None))
    idx = np.argmax(x, axis=actual if actual < 1 else actual + 1)
    if keep:
        idx = np.expand_dims(idx, actual if actual < 1 else actual + 1)
    check(result[0], idx, out_dim)
    check(result[1], np.max(x, axis=actual if actual < 1 else actual + 1, keepdims=keep), out_dim)
    check(V.get_argmin_vmap_rule(Primitive('Argmin'), 2)((t(x), 1), (axis, None), (34, None)),
          np.argmin(x, axis=actual if actual < 1 else actual + 1), 1 if actual == 1 else 0)
    assert logical.shape == (2, 3, 4)


@pytest.mark.parametrize('xd', [0, 2])
def test_matrix_determinants(xd):
    a = np.array([[[2, 0], [0, 3]], [[4, 0], [0, 5]]], np.float32)
    x = np.moveaxis(a, 0, xd)
    check(W.get_matrix_determinant_vmap_rule('MatrixDeterminant', 2)((t(x), xd)), [6, 20])
    sign, logabs = W.get_log_matrix_determinant_vmap_rule('LogMatrixDeterminant', 2)((t(x), xd))
    check(sign, [1, 1])
    check(logabs, np.log([6, 20]))


@pytest.mark.parametrize('xd,yd', [(0, 0), (0, None), (1, 0)])
def test_isclose(xd, yd):
    a = np.array([[1, 2, 3], [1, 4, 3]], np.float32)
    b = np.array([[1, 3, 3], [1, 5, 3]], np.float32)
    y = b[0] if yd is None else np.moveaxis(b, 0, yd)
    expected = np.isclose(a, b[0] if yd is None else b)
    check(W.get_isclose_vmap_rule(Primitive('IsClose'), 2)((t(np.moveaxis(a, 0, xd)), xd), (t(y), yd),
                                                   (1e-5, None), (1e-8, None), (False, None)),
          np.moveaxis(expected, 0, xd), xd)


@pytest.mark.parametrize('axis', [0, -1])
def test_cumulative_max(axis):
    x = np.array([[3, 1, 4], [2, 5, 0]], np.float32)
    result = W.get_cum_min_max_vmap_rule(Primitive('Cummax'), 2)((t(x), 0), (axis, None))
    check(result[0], np.maximum.accumulate(x, axis=1))
    check(result[1], [[0, 0, 2], [0, 1, 1]])


@pytest.mark.parametrize('xd,gd', [(0, 0), (None, 0), (0, None), (1, 0)])
def test_logit_gradient(xd, gd):
    x = np.array([[.2, .4, .7], [.3, .6, .8]], np.float32)
    g = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    xx = x[0] if xd is None else np.moveaxis(x, 0, xd)
    gg = g[0] if gd is None else np.moveaxis(g, 0, gd)
    expected = (g[0] if gd is None else g) / ((x[0] if xd is None else x) * (1 - (x[0] if xd is None else x)))
    dim = gd if xd is None else xd
    check(D.get_logit_grad_vmap_rule(Primitive('LogitGrad'), 2)((t(gg), gd), (t(xx), xd), (-1., None)),
          np.moveaxis(expected, 0, dim), dim)
    check(W.get_logit_vmap_rule(Primitive('Logit'), 2)((t(xx), xd), (-1., None)), np.log(xx / (1 - xx)), xd)


@pytest.mark.parametrize('name', ['MaximumGrad', 'MinimumGrad'])
@pytest.mark.parametrize('broadcast', [False, True])
def test_extremum_gradient(name, broadcast):
    x = np.array([[1, 4, 3], [6, 2, 5]], np.float32)
    y = np.array([2, 3, 4], np.float32) if broadcast else np.array([[2, 3, 4], [3, 3, 3]], np.float32)
    g = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    result = D.get_broadcast_binary_op_grad_vmap_rule(Primitive(name), 2)((t(x), 0), (t(y), None if broadcast else 0),
                                                             (t(g), 0), (True, None), (True, None))
    mask = x >= y if name == 'MaximumGrad' else x <= y
    check(result[0], np.where(mask, g, 0))
    check(result[1], np.where(mask, 0, g))


@pytest.mark.parametrize('axis', [0, 1, 2, -1])
def test_transpose_and_reshape(axis):
    logical = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    physical = np.moveaxis(logical, 0, axis)
    check(V.get_transpose_vmap_rule('Transpose', 2)((t(physical), axis), ((-1, -2), None)), logical.transpose(0, 2, 1))
    out, dim = V.get_reshape_vmap_rule(P.Reshape(), 2)((t(physical), axis), ((-1,), None))
    assert dim == (1 if axis in (2, -1) else 0)
    np.testing.assert_array_equal(np.moveaxis(out.asnumpy(), dim, 0), logical.reshape(2, 12))


@pytest.mark.parametrize('axis', [0, 2])
def test_tile_and_expand(axis):
    logical = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    x = np.moveaxis(logical, 0, axis)
    out, dim = V.get_tile_vmap_rule(P.Tile(), 2)((t(x), axis), ((1, 2), None))
    np.testing.assert_array_equal(np.moveaxis(out.asnumpy(), dim, 0), np.tile(logical, (1, 1, 2)))
    for insert in [0, -1]:
        out, dim = V.get_expand_dims_vmap_rule(P.ExpandDims(), 2)((t(x), axis), (insert, None))
        np.testing.assert_array_equal(np.moveaxis(out.asnumpy(), dim, 0), np.expand_dims(logical, 1 if insert == 0 else -1))


@pytest.mark.parametrize('axis', [None, (0,), (-1,)])
@pytest.mark.parametrize('batch', [1, 2])
def test_squeeze(axis, batch):
    x = np.arange(batch * 3, dtype=np.float32).reshape(batch, 1, 3, 1)
    expected = np.squeeze(x, axis=(1, 3) if axis is None else tuple(a + 1 if a >= 0 else a for a in axis))
    check(V.get_squeeze_vmap_rule(ops.squeeze, batch)((t(x), 0), (axis, None)), expected)


def test_slice_and_grad_and_strided_slice():
    x = np.arange(12, dtype=np.float32).reshape(2, 6)
    check(V.get_slice_vmap_rule('Slice', 2)((t(x), 0), ((1,), None), ((3,), None)), x[:, 1:4])
    grad = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    expected = np.pad(grad, ((0, 0), (1, 2)))
    check(V.get_slice_grad_vmap_rule('SliceGrad', 2)((t(grad), 0), (t(x), 0), ((1,), None), ((3,), None)), expected)
    check(V.get_stridedslice_vmap_rule(P.StridedSlice(), 2)((t(x), 0), ((1,), None), ((6,), None), ((2,), None),
          *((0, None),) * 5), x[:, 1:6:2])
    check(V.get_stridedslice_grad_vmap_rule(G.StridedSliceGrad(), 2)((t(grad), 0), ((6,), None),
          ((1,), None), ((6,), None), ((2,), None)), np.array([[0, 1, 0, 2, 0, 3], [0, 4, 0, 5, 0, 6]]))


@pytest.mark.parametrize('shape', [(2, 3), ms.Tensor([2, 3], ms.int64)])
def test_fill_variants(shape):
    values = t([2, 5])
    check(V.get_fill_v2_vmap_rule('FillV2', 2)((shape, None), (values, 0)), np.broadcast_to(np.array([2, 5])[:, None, None], (2, 2, 3)))
    check(V.get_fill_vmap_rule('Fill', 2)((ms.int32, None), ((2, 3), None), (values, 0)), np.broadcast_to(np.array([2, 5])[:, None, None], (2, 2, 3)))


@pytest.mark.parametrize('xd,vd', [(0, None), (None, 0), (1, 0)])
def test_fills(xd, vd):
    x = np.zeros((3,), np.float32) if xd is None else np.zeros((2, 3), np.float32)
    if xd == 1:
        x = x.T
    v = t(7) if vd is None else t([2, 5])
    expected = np.full(x.shape, 7) if vd is None else np.broadcast_to(np.array([2, 5])[:, None], (2, 3))
    check(V.get_fills_vmap_rule('Fills', 2)((t(x), xd), (v, vd)), expected, xd if vd is None else 0)


@pytest.mark.parametrize('xd,idim', [(0, None), (None, 0), (0, 0)])
@pytest.mark.parametrize('axis', [0, -1])
def test_gather(xd, idim, axis):
    x = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    indices = np.array([[0, 1], [1, 0]], np.int32)
    xx = x[0] if xd is None else x
    ii = indices[0] if idim is None else indices
    result, dim = V.get_gather_vmap_rule(Primitive('Gather'), 2)((t(xx), xd), (t(ii, np.int32), idim), (axis, None), (0, None))
    expected = np.stack([np.take(x[0] if xd is None else x[i], indices[0] if idim is None else indices[i], axis=axis) for i in range(2)])
    assert dim == (axis % 2 if xd is None or idim is not None else 0)
    np.testing.assert_array_equal(np.moveaxis(result.asnumpy(), dim, 0), expected)


def test_gather_nd_and_scatter_nd():
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    index = np.array([[[1], [3]], [[0], [2]]], np.int32)
    check(V.get_gather_nd_vmap_rule('GatherNd', 2)((t(x), 0), (t(index, np.int32), 0)), [[1, 3], [4, 6]])
    updates = t([[10, 20], [30, 40]])
    expected = [[0, 10, 0, 20], [30, 0, 40, 0]]
    check(V.get_scatter_nd_vmap_rule('ScatterNd', 2)((t(index, np.int32), 0), (updates, 0), ((4,), None)), expected)
    check(V.get_tensor_scatter_op_vmap_rule('TensorScatterAdd', 2)((t(x), 0), (t(index, np.int32), 0), (updates, 0)), x + expected)


@pytest.mark.parametrize('dims', [(0, None, None), (None, 0, None), (None, None, 0),
                                 (None, 0, 0), (0, None, 0), (0, 0, None), (0, 0, 0)])
def test_tensor_scatter_elements(dims):
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    index = np.array([[1, 3], [0, 2]], np.int32)
    update = np.array([[10, 20], [30, 40]], np.float32)
    args = tuple((t(a[0] if dim is None else a, dtype), dim) for a, dim, dtype in zip((x, index, update), dims, (np.float32, np.int32, np.float32)))
    expected = np.stack([x[0].copy() if dims[0] is None else x[i].copy() for i in range(2)])
    for i in range(2):
        expected[i, index[0] if dims[1] is None else index[i]] = update[0] if dims[2] is None else update[i]
    check(V.get_tensor_scatter_elements_vmap_rule('TensorScatterElements', 2)(*args), expected)


@pytest.mark.parametrize('indexing', [0, 1])
def test_meshgrid(indexing):
    x = np.array([[1, 2], [3, 4]], np.float32)
    y = np.array([[5, 6, 7], [8, 9, 10]], np.float32)
    result = V.get_meshgrid_vmap_rule('Meshgrid', 2)(((t(x), 0), (t(y.T), 1)), (indexing, None))
    expected = [np.meshgrid(x[i], y[i], indexing='ij' if indexing == 0 else 'xy') for i in range(2)]
    for j in range(2):
        check(result[j], np.stack([e[j] for e in expected]))


def test_masked_select_and_gradient():
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    mask = np.array([True, False, True, False])
    check(V.get_masked_select_vmap_rule(P.MaskedSelect(), 2)((t(x), 0), (t(mask, bool), None)), x[:, mask])
    check(V.get_masked_select_grad_vmap_rule('MaskedSelectGrad', 2)((t(x), 0), (t(mask, bool), None), (t([[1, 2], [3, 4]]), 0)),
          [[1, 0, 2, 0], [3, 0, 4, 0]])


@pytest.mark.parametrize('xd', [0, 1])
def test_topk(xd):
    x = np.array([[1, 4, 2], [6, 3, 5]], np.float32)
    value, index = V.get_topk_vmap_rule(P.TopK(sorted=True), 2)((t(np.moveaxis(x, 0, xd)), xd), (2, None))
    check(value, [[4, 2], [6, 5]])
    check(index, [[1, 2], [0, 2]])


def test_flatten_split_tensor_shape():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    check(V.get_flatten_vmap_rule(P.Flatten(), 2)((t(x), 0)), x.reshape(2, 3, 4))
    check(V.get_tensor_shape_vmap_rule(P.TensorShape(), 2)((t(x), 0)), [3, 4], None)
    result = V.get_split_vmap_rule(P.Split(), 2)((t(x), 0), (-1, None), (2, None))
    check(result[0], x[:, :, :2])
    check(result[1], x[:, :, 2:])


def test_base_stack_unstack_output_and_unary():
    x = t([[1, 2], [3, 4]])
    vals = B.vmap_unstack(1, x)
    np.testing.assert_array_equal(B.vmap_stack(vals).asnumpy(), [[1, 3], [2, 4]])
    check(B.vmap_general_output_process(vals), [[1, 3], [2, 4]])
    check(B.get_unop_vmap_rule('Sin', 2)((x, 1)), np.sin([[1, 2], [3, 4]]), 1)
    with pytest.raises(TypeError, match='mapping'):
        B.vmap_general_output_process(((x, x), (x, x)))


def test_base_dynamic_and_monad_errors():
    x = t([1, 2])
    with pytest.raises(ValueError, match='all axis should be none'):
        B.get_unsupported_dynamic_vmap_rule('Unique', 2)((x, 0))
    for arg in [x, (x, 0)]:
        with pytest.raises(ValueError, match='side effect'):
            B.vmap_monad_rule('Assign', 2)(arg)
    with pytest.raises(ValueError, match='Unsupported number'):
        B.get_assign_vmap_rule('Assign', 2)((x, 0))
    with pytest.raises(ValueError, match='execution order'):
        B.get_assign_vmap_rule(P.Assign(), 2)((x, None), (x, 0), None)
    with pytest.raises(ValueError, match='Failed to get'):
        B._vmap_clone_prim(Primitive('NotRegisteredVmapTest'))


@pytest.mark.parametrize('name', [
    pytest.param('UnsortedSegmentSum', marks=pytest.mark.xfail(
        strict=True, reason='UnsortedSegmentSum wrapper lacks PrimitiveFunction label API')),
    'UnsortedSegmentMin', 'UnsortedSegmentMax', 'UnsortedSegmentProd'])
def test_unsorted_segments(name):
    x = np.array([[1, 3, 2, 4], [5, 7, 6, 8]], np.float32)
    ids = np.array([0, 1, 0, 1], np.int32)
    prim = getattr(P, name)()
    factory = V.get_unsorted_segment_sum_vmap_rule if name.endswith('Sum') else V.get_unsorted_segment_arithmetic_vmap_rule
    result = factory(prim, 2)((t(x), 0), (t(ids, np.int32), None), (2, None))
    reduce = {'Sum': np.sum, 'Min': np.min, 'Max': np.max, 'Prod': np.prod}[name.removeprefix('UnsortedSegment')]
    expected = np.stack([reduce(x[:, ids == j], axis=1) for j in range(2)], axis=1)
    check(result, expected)
    with pytest.raises(ValueError, match='num_segment'):
        factory(prim, 2)((t(x), 0), (t(ids, np.int32), None), (2, 0))


@pytest.mark.parametrize('name', ['CSRSparseMatrixToSparseTensor', 'DenseToCSRSparseMatrix', 'CSRSparseMatrixToDense'])
def test_sparse_conversions(name):
    factory = {'CSRSparseMatrixToSparseTensor': S.get_csr_sparse_matrix_to_sparse_tensor_vmap_rule,
               'DenseToCSRSparseMatrix': S.get_dense_to_csr_sparse_matrix_vmap_rule,
               'CSRSparseMatrixToDense': S.get_csr_sparse_matrix_to_dense_vmap_rule}[name]
    shape = t([2, 3], np.int32)
    batch = t([0, 3], np.int32)
    row = t([0, 2, 3], np.int32)
    col = t([0, 2, 1], np.int32)
    val = t([1, 2, 3])
    dense = np.array([[1, 0, 2], [0, 3, 0]], np.float32)
    indices = np.array([[0, 0], [0, 2], [1, 1]], np.int32)
    args = (t(dense), t(indices, np.int32)) if name == 'DenseToCSRSparseMatrix' else (shape, batch, row, col, val)
    rule = factory(name, 2)
    result = rule(*((a, None) for a in args))
    if name == 'CSRSparseMatrixToDense':
        check(result, dense, None)
    else:
        expected = (indices, [1, 2, 3], [2, 3]) if name == 'CSRSparseMatrixToSparseTensor' else ([2, 3], [0, 3], [0, 2, 3], [0, 2, 1], [1, 2, 3])
        for out, exp in zip(result, expected):
            check(out, exp, None)
    for i in range(len(args)):
        bdims = [(a, 0 if j == i else None) for j, a in enumerate(args)]
        with pytest.raises(ValueError, match='all axes'):
            rule(*bdims)


@pytest.mark.parametrize('batch_axis', [0, 1])
def test_diagonal_matrix_family(batch_axis):
    diagonal = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    expected = np.stack([np.diag(d) for d in diagonal])
    check(V.get_matrix_diag_v3_vmap_rule('MatrixDiagV3', 2)((t(np.moveaxis(diagonal, 0, batch_axis)), batch_axis),
          (t(0, np.int32), None), (t(-1, np.int32), None), (t(-1, np.int32), None), (t(0), None)), expected)
    check(V.get_matrix_diag_part_v3_vmap_rule('MatrixDiagPartV3', 2)((t(expected), 0), (t(0, np.int32), None), (t(0), None)), diagonal)
    check(V.get_matrix_set_diag_v3_vmap_rule('MatrixSetDiagV3', 2)((t(np.zeros((2, 3, 3))), 0),
          (t(diagonal), 0), (t(0, np.int32), None)), expected)


@pytest.mark.parametrize('xd', [0, 1])
def test_padding(xd):
    logical = np.array([[[1], [2]], [[3], [4]]], np.float32)
    x = np.moveaxis(logical, 0, xd)
    result, dim = V.get_padding_vmap_rule(P.Padding(3), 2)((t(x), xd))
    assert dim == xd
    np.testing.assert_array_equal(np.moveaxis(result.asnumpy(), dim, 0), np.pad(logical, ((0, 0), (0, 0), (0, 2))))


def test_padding_last_batch_axis():
    logical = np.array([[[1], [2]], [[3], [4]]], np.float32)
    check(V.get_padding_vmap_rule(P.Padding(3), 2)((t(np.moveaxis(logical, 0, 2)), 2)),
          np.pad(logical, ((0, 0), (0, 0), (0, 2))))


@pytest.mark.parametrize('dim,idx_dim,grad_dim', [(0, 0, 0), (None, 1, 1), (None, None, 0)])
def test_gatherd_gradient(dim, idx_dim, grad_dim):
    x = np.zeros((2, 4), np.float32)
    idx = np.array([[1, 3], [0, 2]], np.int32)
    grad = np.array([[1, 2], [3, 4]], np.float32)
    vals = [a[0] if d is None else np.moveaxis(a, 0, d) for a, d in zip((x, idx, grad), (dim, idx_dim, grad_dim))]
    result, outdim = V.get_gatherd_grad_v2_vmap_rule('GatherDGradV2', 2)((t(vals[0]), dim), (-1, None),
        (t(vals[1], np.int32), idx_dim), (t(vals[2]), grad_dim))
    expected = np.zeros_like(x)
    for i in range(2):
        np.add.at(expected[i], idx[0] if idx_dim is None else idx[i], grad[0] if grad_dim is None else grad[i])
    assert outdim == next(d for d in (dim, idx_dim, grad_dim) if d is not None)
    np.testing.assert_array_equal(np.moveaxis(result.asnumpy(), outdim, 0), expected)


@pytest.mark.parametrize('xd,idim', [(0, 0), (0, None)])
def test_gatherd(xd, idim):
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    index = np.array([[1, 3], [0, 2]], np.int32)
    ii = index[0] if idim is None else index
    check(V.get_gatherd_vmap_rule('GatherD', 2)((t(x), xd), (-1, None), (t(ii, np.int32), idim)),
          np.stack([x[i, index[0] if idim is None else index[i]] for i in range(2)]))


def test_ger_band_and_data_format():
    a = np.array([[1, 2], [3, 4]], np.float32)
    b = np.array([5, 6, 7], np.float32)
    check(V.get_ger_vmap_rule(P.Ger(), 2)((t(a), 0), (t(b), None)), a[:, :, None] * b)
    x = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    check(V.get_matrix_band_part_vmap_rule('MatrixBandPart', 2)((t(x), 0), (t(0, np.int64), None), (t(0, np.int64), None)),
          x * np.eye(3))
    check(V.get_data_format_dim_map_vmap_rule(P.DataFormatDimMap(), 2)((t([[0, 1, 2, 3], [-4, -3, -2, -1]], np.int32), 0)),
          [[0, 3, 1, 2], [0, 3, 1, 2]])


@pytest.mark.xfail(strict=True, reason='MaskedFill wrapper lacks the required PrimitiveFunction label API')
def test_masked_fill_preserves_values():
    result = V.get_masked_fill_vmap_rule(P.MaskedFill(), 2)(
        (t([[1, 2], [3, 4]]), 0), (t([[True, False], [False, True]], np.bool_), 0), (-1., None))
    check(result, [[-1, 2], [3, -1]])


@pytest.mark.parametrize('xd', [0, 2])
def test_no_repeat_ngram(xd):
    state = np.array([[[0, 1, 0]], [[1, 2, 1]]], np.int32)
    probs = np.zeros((2, 1, 4), np.float32)
    expected = probs.copy()
    expected[0, 0, 1] = np.finfo(np.float32).min
    expected[1, 0, 2] = np.finfo(np.float32).min
    check(V.get_no_repeat_ngram_vmap_rule(P.NoRepeatNGram(2), 2)((t(np.moveaxis(state, 0, xd), np.int32), xd),
          (t(probs), 0)), expected)


def test_searchsorted_and_sorter():
    sequence = t([[1, 3, 5], [2, 4, 6]])
    values = t([[0, 3, 7], [1, 4, 8]])
    for sorter in [None, t([[0, 1, 2], [0, 1, 2]], np.int64)]:
        rule = V.get_searchsorted_vmap_rule(Primitive('SearchSorted'), 2)
        if sorter is None:
            result = rule((sequence, 0), (values, 0), (None, None), (35, None), (False, None))
            check(result, [[0, 1, 3], [0, 1, 3]])
        else:
            with pytest.raises(RuntimeError, match='CPU Device is not supported sorter'):
                # A synchronous launch reports the same capability error at the call boundary.
                result = rule((sequence, 0), (values, 0), (sorter, 0), (35, None), (False, None))
                assert result[1] == 0
                result[0].asnumpy()


@pytest.mark.parametrize('xd', [0, 1])
def test_im2col(xd):
    x = np.arange(32, dtype=np.float32).reshape(2, 1, 1, 4, 4)
    expected = np.stack([x[:, :, :, i:i + 2, j:j + 2].reshape(2, 1, 4) for i in range(3) for j in range(3)], axis=-1)
    check(V.get_im2col_vmap_rule(P.Im2Col(ksizes=2), 2)((t(np.moveaxis(x, 0, xd)), xd)), expected[:, :, None])


@pytest.mark.parametrize('xd', [0, 1])
def test_space_batch_inverse(xd):
    x = np.arange(32, dtype=np.float32).reshape(2, 1, 1, 4, 4)
    space = P.SpaceToBatchND([2, 2], [[0, 0], [0, 0]])
    batch = P.BatchToSpaceND([2, 2], [[0, 0], [0, 0]])
    value, dim = V.get_space_to_batch_nd_vmap_rule(space, 2)((t(np.moveaxis(x, 0, xd)), xd))
    assert dim == 1
    expected = np.stack([x[:, :, :, i::2, j::2] for i in range(2) for j in range(2)], axis=0).reshape(4, 2, 1, 2, 2)
    np.testing.assert_array_equal(value.asnumpy(), expected)
    restored, outdim = V.get_batch_to_space_nd_vmap_rule(batch, 2)((value, dim))
    assert outdim == 1
    np.testing.assert_array_equal(np.moveaxis(restored.asnumpy(), 1, 0), x)


@pytest.mark.parametrize('compute_uv', [False, True])
def test_svd(compute_uv):
    x = np.array([[[3, 0], [0, 2]], [[5, 0], [0, 4]]], np.float32)
    s, u, v = W.get_svd_vmap_rule(Primitive('Svd'), 2)((t(x), 0), (False, None), (compute_uv, None))
    check(s, [[3, 2], [5, 4]])
    assert u[1] == v[1] == (0 if compute_uv else None)
    if compute_uv:
        reconstructed = (u[0].asnumpy() * s[0].asnumpy()[:, None, :]) @ v[0].asnumpy().transpose(0, 2, 1)
        np.testing.assert_allclose(reconstructed, x)
    else:
        assert u[0].size == v[0].size == 1


@pytest.mark.parametrize('keep', [False, True])
def test_norm_and_std(keep):
    x = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    result = W.get_reduce_std_vmap_rule(Primitive('ReduceStd'), 2)((t(x), 0), (-1, None), (False, None), (keep, None))
    check(result[0], x.std(axis=-1, keepdims=keep))
    check(result[1], x.mean(axis=-1, keepdims=keep))
    check(W.get_lp_norm_vmap_rule(M.LpNorm(axis=-1, p=2, keep_dims=keep), 2)((t(x), 0)),
          np.linalg.norm(x, axis=-1, keepdims=keep))


@pytest.mark.parametrize('xd', [0, 1, -1])
def test_renorm(xd):
    x = np.array([[[3, 4], [0, 2]], [[6, 8], [1, 0]]], np.float32)
    norms = np.linalg.norm(x, axis=2, keepdims=True)
    expected = x * np.minimum(1, 5 / (norms + 1e-7))
    physical = np.moveaxis(x, 0, xd)
    value, dim = W.get_renorm_rule(P.Renorm(2, 0, 5.), 2)((t(physical), xd))
    assert dim == xd % 3
    np.testing.assert_allclose(np.moveaxis(value.asnumpy(), dim, 0), expected, rtol=1e-5)


@pytest.mark.xfail(strict=True, reason='LinSpace wrapper lacks the required PrimitiveFunction label API')
def test_linspace_retains_each_samples_endpoints():
    result = W.get_linspace_rule(P.LinSpace(), 2)((t([0, 2]), 0), (t([2, 4]), 0), (3, None))
    check(result, [[0, 1, 2], [2, 3, 4]])


def test_sparse_segment_and_square_sum():
    x = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    check(W.get_sparse_segment_mean_vmap_rule(M.SparseSegmentMean(), 1)((t(x), 0), (t([0, 2], np.int32), None),
          (t([0, 0], np.int32), None)), x[:, [0, 2]].mean(axis=1, keepdims=True))
    a, b = W.get_square_sum_all_vmap_rule('SquareSumAll', 1)((t(x), 0), (t(x + 1), 0))
    check(a, (x ** 2).sum(axis=(1, 2)))
    check(b, ((x + 1) ** 2).sum(axis=(1, 2)))


@pytest.mark.parametrize('string', [
    pytest.param(False, marks=pytest.mark.xfail(strict=True, reason='FFTWithSize object rule reads missing oneside attr')),
    True])
def test_fft(string):
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.complex64)
    prim = 'FFTWithSize' if string else M.FFTWithSize(1, False, False)
    check(W.get_fft_with_size_vmap_rule(prim, 2)((t(x, np.complex64), 0)), np.fft.fft(x))
    with pytest.raises(ValueError, match='source axi'):
        W.get_fft_with_size_vmap_rule(prim, 2)((t(x, np.complex64), 1))


def test_round_and_logit_type_errors():
    x = t([[1.25, -1.85], [3.51, 4.49]])
    check(W.get_round_vmap_rule('Round', 2)((x, 0), (0, None)), [[1, -2], [4, 4]])
    with pytest.raises(ValueError, match='decimal'):
        W.get_round_vmap_rule('Round', 2)((x, 0), (1, 0))
    for factory in (W.get_logit_vmap_rule, D.get_logit_grad_vmap_rule):
        with pytest.raises(TypeError, match="can't be str"):
            factory('Logit', 2)


@pytest.mark.parametrize('longest', ['x', 'y', 'g'])
def test_extremum_gradient_rank_broadcast(longest):
    small = np.array([[1, 4, 2], [6, 2, 5]], np.float32)
    large = small[:, None, :]
    x = large if longest == 'x' else small
    y = large + 2 if longest == 'y' else small + 2
    g = np.ones_like(large) if longest == 'g' else np.ones_like(small)
    result = D.get_broadcast_binary_op_grad_vmap_rule(Primitive('MaximumGrad'), 2)(
        (t(x), 0), (t(y), 0), (t(g), 0), (True, None), (True, None))
    xx = x if x.ndim == 3 else x[:, None, :]
    yy = y if y.ndim == 3 else y[:, None, :]
    gg = g if g.ndim == 3 else g[:, None, :]
    dx = np.where(xx > yy, gg, 0)
    dy = np.where(xx > yy, 0, gg)
    if x.ndim == 2:
        dx = dx.sum(axis=1)
    if y.ndim == 2:
        dy = dy.sum(axis=1)
    check(result[0], dx)
    check(result[1], dy)


@pytest.mark.parametrize('maximum', [True])
@pytest.mark.parametrize('broadcast', [False, True])
def test_extremum_second_gradient(maximum, broadcast):
    x = np.array([[1, 4, 2], [6, 2, 5]], np.float32)
    y = np.full_like(x, 3)
    if broadcast:
        y = y.T
    dx = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    dy = np.array([[7, 8, 9], [10, 11, 12]], np.float32)
    args = ((t(x), 0), (t(y), 1 if broadcast else 0), (t(dx), 0), (t(dy), 0))
    factory = D.get_maximum_grad_grad_vmap_rule if maximum else D.get_broadcast_grad_grad_vmap_rule
    prim = Primitive('MaximumGradGrad') if maximum else G.MinimumGradGrad()
    result = factory(prim, 2)(*(args + ((True, None), (True, None)) if maximum else args))
    check(result[0], np.zeros_like(x))
    check(result[1], np.zeros_like(x))
    mask = x > 3 if maximum else x < 3
    check(result[2], np.where(mask, dx, dy))


@pytest.mark.parametrize('axis', [0, -1])
@pytest.mark.parametrize('keep', [False, True])
def test_median_and_gradient(axis, keep):
    x = np.array([[1, 5, 3], [8, 2, 6]], np.float32)
    med = np.median(x, axis=1, keepdims=keep)
    idx = np.array([[2], [2]], np.int64) if keep else np.array([2, 2], np.int64)
    result = W.get_median_vmap_rule(M.Median(False, axis, keep), 2)((t(x), 0))
    check(result[0], med)
    check(result[1], idx)
    sensitivity = np.array([[10], [20]], np.float32) if keep else np.array([10, 20], np.float32)
    check(D.get_median_grad_vmap_rule(G.MedianGrad(False, -1, keep), 2)((t(sensitivity), 0), (t(x), 0),
          (t(med), 0), (t(idx, np.int64), 0)), [[0, 0, 10], [0, 0, 20]])


@pytest.mark.parametrize('xd,vd', [(0, None), (1, 0), (None, None)])
def test_assignment_real_parameter(xd, vd):
    original = np.arange(6, dtype=np.float32).reshape(2, 3)
    var_array = original[0] if xd is None else np.moveaxis(original, 0, xd)
    value = np.array([10, 20, 30], np.float32) if vd is None else np.array([[10, 20, 30], [40, 50, 60]], np.float32)
    var = ms.Parameter(t(var_array), name='vmap_assignment')
    with pytest.raises(RuntimeError, match='inputs number should be 2 but got 3'):
        B.get_assign_vmap_rule(Primitive('Assign'), 2)((var, xd), (t(value), vd), UMonad())
    np.testing.assert_array_equal(var.asnumpy(), var_array)



@pytest.mark.parametrize('name', ['ScatterAdd', 'ScatterNdAdd'])
def test_stateful_scatter(name):
    x = ms.Parameter(t(np.zeros((2, 4))), name='vmap_scatter')
    index = np.array([[1, 3], [0, 2]], np.int32)
    if name == 'ScatterNdAdd':
        index = index[:, :, None]
    with pytest.raises(ValueError, match='inputs number 4.*signature size 3'):
        V.get_scatter_op_vmap_rule(name, 2)((x, 0), (t(index, np.int32), 0),
                                         (t([[10, 20], [30, 40]]), 0), UMonad())
    np.testing.assert_array_equal(x.asnumpy(), np.zeros((2, 4)))


@pytest.mark.parametrize('xd,yd', [(0, None), (None, 0), (0, 0)])
def test_inplace_add(xd, yd):
    x = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    update = np.ones((2, 1, 2), np.float32)
    xx = x[0] if xd is None else x
    yy = update[0] if yd is None else update
    result, dim = W.get_inplace_ops_vmap_rule(P.InplaceAdd((1,)), 2)((t(xx), xd), (t(yy), yd))
    expected = x.copy() if xd is not None else np.broadcast_to(xx, x.shape).copy()
    expected[:, 1, :] += 1
    assert dim == 2
    np.testing.assert_array_equal(np.moveaxis(result.asnumpy(), dim, 0), expected)


def test_reverse_sequence_mapped_and_unmapped_x():
    x = np.arange(24, dtype=np.float32).reshape(2, 2, 3, 2)
    lengths = np.array([[2, 2], [1, 2]], np.int32)
    for xd in [0, None]:
        xx = x[0] if xd is None else x
        lengths = np.ones((2, 2), np.int32) if xd is None else lengths
        result = V.get_reverse_sequence_vmap_rule(P.ReverseSequence(seq_dim=1, batch_dim=0), 2)(
            (t(xx), xd), (t(lengths, np.int32), 0))
        expected = np.broadcast_to(xx, x.shape).copy() if xd is None else x.copy()
        for i in range(2):
            for j in range(2):
                expected[i, j, :lengths[i, j]] = expected[i, j, :lengths[i, j]][::-1]
        check(result, expected)


@pytest.mark.parametrize('axis,batchdim,expected', [(0, 0, (1, 0)), (0, 1, (0, 2)),
                                                  (-1, 1, (1, 2)), (-1, 0, (-1, 0))])
def test_onehot_axis_helper(axis, batchdim, expected):
    assert V._get_one_hot_vmap_axis(axis, 2, batchdim) == expected


@pytest.mark.parametrize('axis', [0, -1])
def test_onehot(axis):
    indices = np.array([[0, 2], [1, 0]], np.int32)
    expected = np.eye(3, dtype=np.float32)[indices]
    result, dim = V.get_one_hot_vmap_rule(Primitive('OneHot'), 2)((t(indices, np.int32), 0), (3, None),
                     (t(1), None), (t(0), None), (axis, None))
    if axis == 0:
        expected = expected.transpose(0, 2, 1)
    assert dim == 0
    np.testing.assert_array_equal(result.asnumpy(), expected)


def test_prefix_empty_and_vector():
    with pytest.raises(ValueError, match='indices_shape is empty'):
        V._get_prefix((), 2, ms.int32)
    np.testing.assert_array_equal(V._get_prefix((2,), 2, ms.int64).asnumpy(), [0, 1])


@pytest.mark.xfail(strict=True, reason='Installed primitive wrappers lack the required PrimitiveFunction label API')
@pytest.mark.parametrize('name', ['Cdist', 'Diag', 'UnsortedSegmentSum', 'MaskedFill', 'LinSpace'])
def test_primitive_function_label_api_boundary(name):
    factory = {'Cdist': W.get_cdist_vmap_rule, 'Diag': V.get_diag_vmap_rule,
               'UnsortedSegmentSum': V.get_unsorted_segment_sum_vmap_rule,
               'MaskedFill': V.get_masked_fill_vmap_rule, 'LinSpace': W.get_linspace_rule}[name]
    assert callable(factory(getattr(P, name)(), 2))


@pytest.mark.xfail(strict=True, reason='STFT vmap factory uses _vmap_clone_prim without importing it')
@pytest.mark.parametrize('existing_batch_rank', [False, True])
def test_stft_factory_resolves_clone(existing_batch_rank):
    prim = M.STFT(4, 2, 4, False, True, False)
    if existing_batch_rank:
        prim.add_prim_attr('batch_rank', 0)
    assert callable(W.get_stft_vmap_rule(prim, 2))


def _bad_axis_cases():
    x = t(np.ones((2, 3)))
    i = t([[0], [1]], np.int32)
    return [
        (V.get_cast_vmap_rule, 'Cast', ((x, 0), (ms.int32, 0)), 'type'),
        (V.get_transpose_vmap_rule, 'Transpose', ((x, 0), ((0,), 0)), 'perm'),
        (V.get_transpose_ext_vmap_rule, 'TransposeExtView', ((x, 0), (0, 0), (0, None)), 'dim1_dim'),
        (V.get_tile_vmap_rule, P.Tile(), ((x, 0), ((1,), 0)), 'shape'),
        (V.get_concat_vmap_rule, Primitive('Concat'), (((x, 0), (x, 0)), (0, 0)), 'axis'),
        (V.get_reshape_vmap_rule, P.Reshape(), ((x, 0), ((3,), 0)), 'shape'),
        (V.get_scatter_nd_vmap_rule, 'ScatterNd', ((i, 0), (x, 0), ((3,), 0)), 'shape'),
        (V.get_slice_grad_vmap_rule, 'SliceGrad', ((x, 0), (x, 0), ((0,), 0), ((3,), None)), 'begin'),
        (V.get_slice_grad_vmap_rule, 'SliceGrad', ((x, 0), (x, 0), ((0,), None), ((3,), 0)), 'size'),
        (V.get_fill_vmap_rule, 'Fill', ((ms.float32, 0), ((3,), None), (x, 0)), 'type'),
        (V.get_fill_vmap_rule, 'Fill', ((ms.float32, None), ((3,), 0), (x, 0)), 'shape'),
        (V.get_fill_vmap_rule, 'Fill', ((ms.float32, None), ((3,), None), (x, 0)), 'value'),
        (V.get_fill_v2_vmap_rule, 'FillV2', (((3,), 0), (x, 0)), 'shape'),
        (V.get_fill_v2_vmap_rule, 'FillV2', (((3,), None), (x, 0)), 'value'),
        (V.get_fills_vmap_rule, 'Fills', ((x, 0), (x, 0)), 'value'),
        (V.get_range_vmap_rule, 'Range', ((0, 0), (3, None), (1, None)), 'all axis'),
        (V.get_matrix_diag_v3_vmap_rule, 'MatrixDiagV3', ((x, 0), (0, 0), (-1, None), (-1, None), (0, None)), '`k`'),
        (V.get_matrix_diag_v3_vmap_rule, 'MatrixDiagV3', ((x, 0), (0, None), (-1, 0), (-1, None), (0, None)), 'num_rows'),
        (V.get_matrix_diag_v3_vmap_rule, 'MatrixDiagV3', ((x, 0), (0, None), (-1, None), (-1, 0), (0, None)), 'num_cols'),
        (V.get_matrix_diag_v3_vmap_rule, 'MatrixDiagV3', ((x, 0), (0, None), (-1, None), (-1, None), (0, 0)), 'padding_value'),
        (V.get_one_hot_vmap_rule, Primitive('OneHot'), ((i, 0), (3, 0), (1, None), (0, None), (-1, None)), 'depth'),
        (V.get_one_hot_vmap_rule, Primitive('OneHot'), ((i, 0), (3, None), (1, 0), (0, None), (-1, None)), 'on_value'),
        (V.get_one_hot_vmap_rule, Primitive('OneHot'), ((i, 0), (3, None), (1, None), (0, 0), (-1, None)), 'off_value'),
        (V.get_masked_select_vmap_rule, P.MaskedSelect(), ((x, 0), (x, 0)), 'mask'),
        (V.get_masked_select_grad_vmap_rule, 'MaskedSelectGrad', ((x, 0), (x, 0), (x, 0)), 'mask'),
        (V.get_matrix_diag_part_v3_vmap_rule, 'MatrixDiagPartV3', ((x, 0), (0, 0), (0, None)), '`k`'),
        (V.get_matrix_diag_part_v3_vmap_rule, 'MatrixDiagPartV3', ((x, 0), (0, None), (0, 0)), 'padding_value'),
        (V.get_matrix_set_diag_v3_vmap_rule, 'MatrixSetDiagV3', ((x, 0), (x, 0), (0, 0)), '`k`'),
        (V.get_gatherd_vmap_rule, 'GatherD', ((x, 0), (0, 0), (i, 0)), 'dim'),
        (V.get_gatherd_vmap_rule, 'GatherD', ((x, 0), (t(0, np.int32), None), (i, 0)), 'must be a int'),
        (V.get_gatherd_grad_v2_vmap_rule, 'GatherDGradV2', ((x, 0), (0, 0), (i, 0), (x, 0)), 'dim'),
        (V.get_gatherd_grad_v2_vmap_rule, 'GatherDGradV2', ((x, 0), (3, None), (i, 0), (x, 0)), 'range'),
        (V.get_gather_vmap_rule, 'Gather', ((x, 0), (i, 0), (0, 0), (0, None)), 'axis'),
        (V.get_gather_vmap_rule, 'Gather', ((x, 0), (i, 0), (0, None), (0, 0)), 'batch_dims'),
        (V.get_expand_dims_vmap_rule, P.ExpandDims(), ((x, 0), (0, 0)), 'shape'),
        (V.get_slice_vmap_rule, 'Slice', ((x, 0), ((0,), 0), ((3,), None)), 'begin'),
        (V.get_slice_vmap_rule, 'Slice', ((x, 0), ((0,), None), ((3,), 0)), 'size'),
        (V.get_stridedslice_vmap_rule, P.StridedSlice(), ((x, 0), ((0,), 0), ((3,), None), ((1,), None)) + ((0, None),) * 5, 'batch dimension'),
        (V.get_stridedslice_grad_vmap_rule, G.StridedSliceGrad(), ((x, 0), ((3,), 0), ((0,), None), ((3,), None), ((1,), None)), 'batch dimension'),
        (V.get_topk_vmap_rule, 'TopK', ((x, 0), (2, 0)), '`k`'),
        (W.get_broadcast_to_vmap_rule, P.BroadcastTo((3,)), ((x, 0), ((3,), 0)), 'shape'),
        (W.get_sparse_segment_mean_vmap_rule, M.SparseSegmentMean(), ((x, 0), (i, 0), (i, 0)), 'segment_ids'),
    ]


@pytest.mark.parametrize('factory,prim,args,reason', _bad_axis_cases())
def test_constant_arguments_reject_mapped_axes(factory, prim, args, reason):
    with pytest.raises(ValueError, match=reason):
        factory(prim, 2)(*args)


@pytest.mark.parametrize('shape', [t([[2, 3]], np.int64), [2, 3]])
def test_fillv2_invalid_shape(shape):
    reason = '1-D tensor' if isinstance(shape, ms.Tensor) else 'should be Tuple or Tensor'
    with pytest.raises(ValueError, match=reason):
        V.get_fill_v2_vmap_rule('FillV2', 2)((shape, None), (t([1, 2]), 0))


def test_reshape_multiple_negative_dimensions():
    with pytest.raises(ValueError, match='one -1 at most'):
        V.get_reshape_vmap_rule(P.Reshape(), 2)((t(np.ones((3, 2, 4))), 1), ((-1, -1), None))


@pytest.mark.parametrize('refdim,indexdim,updatedim', [(None, 0, None), (1, None, None)])
def test_scatter_ref_axis_errors(refdim, indexdim, updatedim):
    with pytest.raises(ValueError, match='execution order|must be 0 or None'):
        V.get_scatter_op_vmap_rule(P.ScatterAdd(), 2)((t(np.zeros((2, 4))), refdim),
                (t([0, 1], np.int32), indexdim), (t([2, 3]), updatedim), None)


@pytest.mark.parametrize('inputs,reason', [([((1, 0))], 'not tuple'),
                                         (((t([1, 2]), 0),), 'greater than 1'),
                                         (((t([1, 2]), None), (t([1, 2]), 0)), 'must be same'),
                                         (((t([1, 2]), 0), (t([[1, 2], [3, 4]]), 0)), 'must be 1D')])
def test_meshgrid_invalid_inputs(inputs, reason):
    with pytest.raises(ValueError, match=reason):
        V.get_meshgrid_vmap_rule('Meshgrid', 2)(inputs, (0, None))


def _unmapped_cases():
    x = t([[1, 4, 2], [6, 3, 5]])
    y = t([[2, 3, 4], [3, 4, 3]])
    mask = t([[True, False, True], [False, True, False]], bool)
    i = t([[0], [1]], np.int32)
    d = t([1, 2])
    zero = t(0)
    return [
        (V.get_no_repeat_ngram_vmap_rule, P.NoRepeatNGram(2), (t([[[0, 1, 0]]], np.int32), t([[[0, 0, 0]]])), [[[0, np.finfo(np.float32).min, 0]]]),
        (V.get_argmin_vmap_rule, P.Argmin(), (x, -1, 34), [0, 1]),
        (V.get_arg_min_max_with_value_vmap_rule, P.ArgMaxWithValue(axis=-1), (x, -1, False), ([1, 0], [4, 6])),
        (V.get_transpose_vmap_rule, P.Transpose(), (x, (1, 0)), [[1, 6], [4, 3], [2, 5]]),
        (V.get_transpose_ext_vmap_rule, gen.TransposeExtView(), (x, 0, 1), [[1, 6], [4, 3], [2, 5]]),
        (V.get_tile_vmap_rule, P.Tile(), (d, (2,)), [1, 2, 1, 2]),
        (V.get_concat_vmap_rule, Primitive('Concat'), ((d, d), 0), [1, 2, 1, 2]),
        (V.get_stack_vmap_rule, P.Stack(), ((d, d),), [[1, 2], [1, 2]]),
        (V.get_unstack_vmap_rule, P.Unstack(), (x,), ([1, 4, 2], [6, 3, 5])),
        (V.get_reshape_vmap_rule, P.Reshape(), (x, (6,)), [1, 4, 2, 6, 3, 5]),
        (V.get_reverse_sequence_vmap_rule, P.ReverseSequence(1, 0), (x, t([2, 3], np.int32)), [[4, 1, 2], [5, 3, 6]]),
        (V.get_flatten_vmap_rule, P.Flatten(), (x,), [[1, 4, 2], [6, 3, 5]]),
        (V.get_select_vmap_rule, P.Select(), (mask, x, y), [[1, 3, 2], [3, 3, 3]]),
        (V.get_scatter_nd_vmap_rule, P.ScatterNd(), (i, d, (3,)), [1, 2, 0]),
        (V.get_slice_grad_vmap_rule, G.SliceGrad(), (d, t([0, 0, 0]), (1,), (2,)), [0, 1, 2]),
        (V.get_tensor_scatter_op_vmap_rule, P.TensorScatterAdd(), (t([10, 20, 30]), i, d), [11, 22, 30]),
        (V.get_unsorted_segment_arithmetic_vmap_rule, P.UnsortedSegmentMax(), (d, t([0, 0], np.int32), 1), [2]),
        (V.get_fill_vmap_rule, P.Fill(), (ms.float32, (2,), 3), [3, 3]),
        (V.get_fill_v2_vmap_rule, P.FillV2(), ((2,), t(3)), [3, 3]),
        (V.get_range_vmap_rule, P.Range(), (0, 3, 1), [0, 1, 2]),
        (V.get_matrix_diag_v3_vmap_rule, A.MatrixDiagV3(), (d, t(0, np.int32), t(-1, np.int32), t(-1, np.int32), zero), [[1, 0], [0, 2]]),
        (V.get_tensor_shape_vmap_rule, P.TensorShape(), (x,), [2, 3]),
        (V.get_one_hot_vmap_rule, Primitive('OneHot'), (t([0, 1], np.int32), 2, t(1), zero, -1), [[1, 0], [0, 1]]),
        (V.get_masked_select_vmap_rule, P.MaskedSelect(), (x, mask), [1, 2, 3]),
        (V.get_masked_select_grad_vmap_rule, G.MaskedSelectGrad(), (x, mask, t([1, 2, 3])), [[1, 0, 2], [0, 3, 0]]),
        (V.get_matrix_band_part_vmap_rule, A.MatrixBandPart(), (x, t(0, np.int64), t(0, np.int64)), [[1, 0, 0], [0, 3, 0]]),
        (V.get_matrix_diag_part_v3_vmap_rule, A.MatrixDiagPartV3(), (x, t(0, np.int32), zero), [1, 3]),
        (V.get_matrix_set_diag_v3_vmap_rule, A.MatrixSetDiagV3(), (x, d, t(0, np.int32)), [[1, 4, 2], [6, 2, 5]]),
        (V.get_padding_vmap_rule, P.Padding(2), (t([[1], [2]]),), [[1, 0], [2, 0]]),
        (V.get_ger_vmap_rule, P.Ger(), (d, d), [[1, 2], [2, 4]]),
        (V.get_gatherd_vmap_rule, P.GatherD(), (x, 1, t([[2, 0], [0, 2]], np.int32)), [[2, 1], [6, 5]]),
        (V.get_gatherd_grad_v2_vmap_rule, G.GatherDGradV2(), (x, 1, t([[2], [0]], np.int32), t([[2], [3]])), [[0, 0, 2], [3, 0, 0]]),
        (V.get_gather_nd_vmap_rule, P.GatherNd(), (x, t([[0, 1], [1, 2]], np.int32)), [4, 5]),
        (V.get_meshgrid_vmap_rule, Primitive('Meshgrid'), ((d, d), 0), ([[1, 1], [2, 2]], [[1, 2], [1, 2]])),
        (V.get_gather_vmap_rule, P.Gather(), (x, t([1, 0], np.int32), 0, 0), [[6, 3, 5], [1, 4, 2]]),
        (V.get_tensor_scatter_elements_vmap_rule, A.TensorScatterElements(), (t([10, 20, 30]), t([1, 0], np.int32), d), [2, 1, 30]),
        (V.get_expand_dims_vmap_rule, P.ExpandDims(), (d, -1), [[1], [2]]),
        (V.get_slice_vmap_rule, P.Slice(), (d, (1,), (1,)), [2]),
        (V.get_squeeze_vmap_rule, ops.squeeze, (t([[1, 2]]), (0,)), [1, 2]),
        (V.get_stridedslice_vmap_rule, P.StridedSlice(), (d, (0,), (2,), (1,), 0, 0, 0, 0, 0), [1, 2]),
        (V.get_stridedslice_grad_vmap_rule, G.StridedSliceGrad(), (d, (3,), (0,), (2,), (1,)), [1, 2, 0]),
        (V.get_topk_vmap_rule, P.TopK(), (x, 2), ([[4, 2], [6, 5]], [[1, 2], [0, 2]])),
        (V.get_split_vmap_rule, Primitive('Split'), (d, 0, 2), ([1], [2])),
        (V.get_searchsorted_vmap_rule, Primitive('SearchSorted'), (t([1, 3, 5]), d, None, 35, False), [0, 1]),
        (W.get_addcxxx_vmap_rule, P.Addcmul(), (d, d, d, t(2)), [3, 10]),
        (W.get_lerp_vamp_rule, M.Lerp(), (d, d, .5), [1, 2]),
        (W.get_bernoulli_op_vmap_rule, M.Bernoulli(), (d, 0.), [0, 0]),
        (W.get_add_n_vmap_rule, P.AddN(), ((d, d),), [2, 4]),
        (W.get_matmul_vmap_rule, Primitive('MatMul'), (t([[1, 0], [0, 2]]), t([[3], [4]]), False, False), [[3], [8]]),
        (W.get_matmul_ext_vmap_rule, gen.MatMulExt(), (t([[1, 0], [0, 2]]), t([[3], [4]])), [[3], [8]]),
        (W.get_matrix_solve_vmap_rule, M.MatrixSolve(), (t([[1, 0], [0, 2]]), t([[3], [4]])), [[3], [2]]),
        (W.get_broadcast_to_vmap_rule, Primitive('BroadcastTo'), (d, (2, 2)), [[1, 2], [1, 2]]),
        (W.get_inplace_ops_vmap_rule, P.InplaceAdd((0,)), (x, t([[10, 20, 30]])), [[11, 24, 32], [6, 3, 5]]),
        (W.get_reduce_vmap_rule, Primitive('ReduceMean'), (x, -1, False), [7 / 3, 14 / 3]),
        (W.get_reduce_sum_vmap_rule, Primitive('ReduceSum'), (x, -1, False, False), [7, 14]),
        (W.get_reduce_std_vmap_rule, Primitive('ReduceStd'), (x, -1, False, False), (np.std([[1, 4, 2], [6, 3, 5]], axis=1), [7 / 3, 14 / 3])),
        (W.get_lp_norm_vmap_rule, M.LpNorm(axis=-1, p=2), (x,), np.sqrt([21, 70])),
        (W.get_renorm_rule, P.Renorm(2, 0, 100.), (x,), [[1, 4, 2], [6, 3, 5]]),
        (W.get_matrix_determinant_vmap_rule, M.MatrixDeterminant(), (t([[1, 0], [0, 2]]),), 2),
        (W.get_log_matrix_determinant_vmap_rule, M.LogMatrixDeterminant(), (t([[1, 0], [0, 2]]),), (1, np.log(2))),
        (W.get_cum_min_max_vmap_rule, Primitive('Cummax'), (x, 1), ([[1, 4, 4], [6, 6, 6]], [[0, 1, 1], [0, 0, 0]])),
        (W.get_sparse_segment_mean_vmap_rule, M.SparseSegmentMean(), (x, t([0, 1], np.int32), t([0, 0], np.int32)), [[3.5, 3.5, 3.5]]),
        (W.get_square_sum_all_vmap_rule, M.SquareSumAll(), (d, d), (5, 5)),
        (W.get_isclose_vmap_rule, Primitive('IsClose'), (d, d, 1e-5, 1e-8, False), [True, True]),
        (D.get_broadcast_binary_op_grad_vmap_rule, Primitive('MaximumGrad'), (x, y, t(np.ones((2, 3))), True, True),
         ([[0, 1, 0], [1, 0, 1]], [[1, 0, 1], [0, 1, 0]])),
        (D.get_maximum_grad_grad_vmap_rule, Primitive('MaximumGradGrad'), (x, y, x, y, True, True),
         (np.zeros((2, 3)), np.zeros((2, 3)), [[2, 4, 4], [6, 4, 5]])),
    ]


def _bind_args(args):
    return tuple((arg, None) if not isinstance(arg, tuple) or all(isinstance(x, int) for x in arg)
                 else tuple((x, None) for x in arg) for arg in args)


def _check_tree(result, expected):
    if isinstance(result[0], ms.Tensor):
        check(result, expected, None)
    else:
        assert len(result) == len(expected)
        for out, exp in zip(result, expected):
            _check_tree(out, exp)


@pytest.mark.parametrize('factory,prim,args,expected', _unmapped_cases())
def test_all_unmapped_short_circuit(factory, prim, args, expected):
    _check_tree(factory(prim, 2)(*_bind_args(args)), expected)


def test_select_rank_alignment():
    cond = np.array([[True, False, True], [False, True, False]])
    x = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    y = np.full((2, 3), -1, np.float32)
    check(V.get_select_vmap_rule('Select', 2)((t(cond, bool), 0), (t(x), 0), (t(y), 0)),
          np.where(cond[:, None, :], x, y[:, None, :]))
    check(V.get_select_vmap_rule(P.Select(), 2)((t(cond, bool), 0), (t(y), 0), (t(x), 0)),
          np.where(cond[:, None, :], y[:, None, :], x))


def test_transpose_ext_mapped():
    x = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    check(V.get_transpose_ext_vmap_rule(gen.TransposeExtView(), 2)((t(x), 1), (0, None), (1, None)),
          x.transpose(2, 1, 0), 1)


def test_tile_trailing_batch():
    x = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    check(V.get_tile_vmap_rule(P.Tile(), 2)((t(x), 2), ((2, 1), None)), np.tile(x, (2, 1, 1)), 2)


@pytest.mark.parametrize('target', [(3, 4), (1, 12)])
def test_reshape_preserved_interior_axis(target):
    x = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    value, dim = V.get_reshape_vmap_rule(P.Reshape(), 2)((t(x), 1), (target, None))
    expected = np.moveaxis(x, 1, 0).reshape((2,) + target)
    np.testing.assert_array_equal(np.moveaxis(value.asnumpy(), dim, 0), expected)
    assert dim == (1 if target == (3, 4) else 0)


@pytest.mark.parametrize('axis', [0, -1])
def test_index_fill(axis):
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    args = ((t(x), 0), (t(axis, np.int32), None), (t([1, 3], np.int32), None), (t(-2), None))
    check(V.get_index_fill_rule(A.IndexFill(), 2)(*args), [[0, -2, 2, -2], [4, -2, 6, -2]])
    check(V.get_index_fill_rule(A.IndexFill(), 2)((t([0, 1, 2, 3]), None), *args[1:]), [0, -2, 2, -2], None)


def test_unique_with_pad_real_batch():
    outputs = V.get_unique_with_pad_vmap_rule(P.UniqueWithPad(), 2)((t([[1, 1, 2], [3, 4, 3]], np.int32), 0),
                                                                  (t(-1, np.int32), None))
    check(outputs[0], [[1, 2, -1], [3, 4, -1]])
    check(outputs[1], [[0, 0, 1], [0, 1, 0]])


def test_broadcast_to_and_matmul_ext():
    x = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    check(W.get_broadcast_to_vmap_rule(P.BroadcastTo((4, 3)), 2)((t(x), 0), ((4, 3), None)), np.tile(x[:, None, :], (1, 4, 1)))
    a = np.array([[[1, 0], [0, 2]], [[3, 0], [0, 4]]], np.float32)
    b = np.array([[[2], [3]], [[4], [5]]], np.float32)
    check(W.get_matmul_ext_vmap_rule('MatMulExt', 2)((t(a), 0), (t(b), 0)), np.matmul(a, b))


@pytest.mark.xfail(strict=True, reason='ReduceMean string rule uses a wrapper with an incompatible call signature')
def test_reduce_string_factory_values():
    result = W.get_reduce_vmap_rule('ReduceMean', 2)((t([[1, 2], [3, 4]]), 0), (0, None), (False, None))
    check(result, [1.5, 3.5])


@pytest.mark.parametrize('x_dim,y_dim,axis,reason', [(None, None, 0, 'indices'),
                                                  (None, 0, 0, 'must be None'),
                                                  (3, 0, 0, 'batch dimension'),
                                                  (0, 0, 3, 'axis')])
def test_index_add_rejected_axes(x_dim, y_dim, axis, reason):
    with pytest.raises(ValueError, match=reason):
        W.get_index_add_vmap_rule(P.IndexAdd(axis), 2)((t(np.zeros((2, 3))), x_dim),
            (t([0], np.int32), 0 if reason == 'indices' else None), (t([[1], [2]]), y_dim), None)


@pytest.mark.parametrize('x_dim,y_dim,axis', [(None, None, 0), (0, 0, 0), (1, None, -1), (-1, 0, 0)])
def test_index_add_monad_eager_boundary(x_dim, y_dim, axis):
    x = ms.Parameter(t(np.zeros((2, 3))), name='index_add_vmap')
    with pytest.raises(ValueError, match='inputs number.*signature size'):
        W.get_index_add_vmap_rule(P.IndexAdd(axis), 2)((x, x_dim), (t([0], np.int32), None),
            (t([[1], [2]]), y_dim), UMonad())
    np.testing.assert_array_equal(x.asnumpy(), np.zeros((2, 3)))


def test_svd_unmapped():
    result = W.get_svd_vmap_rule(P.Svd(), 2)((t([[3, 0], [0, 2]]), None), (False, None), (True, None))
    check(result[0], [3, 2], None)
    check(result[1], np.eye(2), None)
    check(result[2], np.eye(2), None)


@pytest.mark.parametrize('broadcast', [False, True])
def test_minimum_second_gradient_sensitivity(broadcast):
    # The CPU kernel clears only one byte of its first two outputs (see
    # minimum_grad_grad_cpu_kernel.cc:83-87). Their metadata remains testable;
    # the fully written sensitivity output provides the numerical oracle.
    x = np.array([[1, 4, 2], [6, 2, 5]], np.float32)
    y = np.full_like(x, 3)
    a = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    b = a + 10
    args = ((t(x), 0), (t(y.T if broadcast else y), 1 if broadcast else 0), (t(a), 0), (t(b), 0))
    result = D.get_broadcast_grad_grad_vmap_rule(G.MinimumGradGrad(), 2)(*args)
    for tensor, dim in result[:2]:
        assert dim == 0
        assert tensor.shape == (2, 3)
        assert tensor.dtype == ms.float32
    check(result[2], np.where(x <= 3, a, b))


@pytest.mark.xfail(strict=True, reason='MinimumGrad string rule calls a wrapper with incompatible gradient flags')
def test_gradient_string_factory_values():
    result = D.get_broadcast_binary_op_grad_vmap_rule('MinimumGrad', 2)(
        (t([1, 2]), 0), (t([2, 1]), 0), (t([1, 1]), 0), (True, None), (True, None))
    check(result[0], [1, 0])
    check(result[1], [0, 1])


def test_primitive_clone_and_attribute_update():
    from mindspore.ops.primitive import _PrimitiveC
    source = _PrimitiveC('VmapAttributeTest', {'value': 7})
    clone = B._vmap_clone_prim(source)
    assert clone is not source
    assert clone.name == source.name
    assert clone.attrs['value'] == 7
    B._vmap_update_prim_attr(clone, 'value', 9)
    assert clone.attrs['value'] == 9
    assert source.attrs['value'] == 7
    with pytest.raises(ValueError, match='bad value 3'):
        B._raise_value_error('bad value ', 3)


def test_general_monad_print_and_dynamic_unmapped(capfd):
    x = t([1, 2])
    result = B.vmap_monad_rule(P.Print(), 2)((x, None))
    assert result == (None, None)
    assert '1' in capfd.readouterr().out
    # General preprocessing preserves the tuple input of a tuple-taking op.
    check(B.get_unsupported_dynamic_vmap_rule(P.AddN(), 2)((x, None), (x, None)), [2, 4], None)


def test_four_argument_assignment_monad_boundary():
    x = ms.Parameter(t([1, 2]), name='inplace_alpha_vmap')
    with pytest.raises(RuntimeError, match='inputs number should be 3 but got 4'):
        B.get_assign_vmap_rule(Primitive('InplaceAddExt'), 2)((x, 0), (t([2, 3]), 0), (2, None), UMonad())
    np.testing.assert_array_equal(x.asnumpy(), [1, 2])


def test_space_batch_and_im2col_unmapped():
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    space = P.SpaceToBatchND([2, 2], [[0, 0], [0, 0]])
    batch = P.BatchToSpaceND([2, 2], [[0, 0], [0, 0]])
    value, dim = V.get_space_to_batch_nd_vmap_rule(space, 2)((t(x), None))
    assert dim is None
    expected = np.stack([x[0, :, i::2, j::2] for i in range(2) for j in range(2)])
    np.testing.assert_array_equal(value.asnumpy(), expected)
    check(V.get_batch_to_space_nd_vmap_rule(batch, 2)((value, None)), x, None)
    check(V.get_im2col_vmap_rule(P.Im2Col(ksizes=1), 2)((t(x), None)), x.reshape(1, 1, 1, 16), None)
    check(V.get_split_vmap_rule(P.Split(), 2)((t([[1, 2], [3, 4]]), 0), (0, None), (2, None))[0], [[1], [3]])


@pytest.mark.parametrize('batchdim', [0, 2])
def test_reverse_sequence_interior_batch(batchdim):
    logical = np.arange(24, dtype=np.float32).reshape(2, 2, 3, 2)
    lengths = np.array([[2, 3], [1, 2]], np.int32)
    expected = logical.copy()
    for i in range(2):
        for j in range(2):
            if batchdim == 0:
                expected[i, j, :lengths[i, j]] = logical[i, j, :lengths[i, j]][::-1]
            else:
                expected[i, :, :lengths[i, j], j] = logical[i, :, :lengths[i, j], j][:, ::-1]
    physical = np.moveaxis(logical, 0, 1)
    check(V.get_reverse_sequence_vmap_rule(P.ReverseSequence(1, batchdim), 2)(
          (t(physical), 1), (t(lengths, np.int32), 0)), expected)


def test_empty_reshape_falls_back_to_front_axis():
    check(V.get_reshape_vmap_rule(P.Reshape(), 2)((t(np.empty((3, 2, 0))), 1), ((0,), None)), np.empty((2, 0)))


def test_nested_general_rule_boundaries():
    x = t([[1, 2], [3, 4]])
    with pytest.raises(TypeError, match="_VmapGeneralRule.*not callable"):
        W.get_median_vmap_rule(M.Median(global_median=True), 2)((x, 0))
    with pytest.raises(TypeError, match="_VmapGeneralRule.*not callable"):
        D.get_median_grad_vmap_rule(G.MedianGrad(global_median=True), 2)(*((x, 0),) * 4)
    with pytest.raises(TypeError, match="_VmapGeneralRule.*not callable"):
        V.get_index_fill_rule(A.IndexFill(), 2)((x, 0), (t([0, 0], np.int32), 0),
                                               (t([1], np.int32), None), (t(3), None))


def test_real_existing_batch_rank_attributes():
    x = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    prim = P.UnsortedSegmentMax().add_prim_attr('batch_rank', 0)
    check(V.get_unsorted_segment_arithmetic_vmap_rule(prim, 2)((t(x), 0), (t([0, 1, 0], np.int32), None), (2, None)),
          [[3, 2], [6, 5]])
    prim = A.MatrixBandPart().add_prim_attr('batch_rank', 0)
    a = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    check(V.get_matrix_band_part_vmap_rule(prim, 2)((t(a), 0), (t(0, np.int64), None), (t(0, np.int64), None)), a * np.eye(2))
    prim = P.DataFormatDimMap().add_prim_attr('batch_rank', 0)
    check(V.get_data_format_dim_map_vmap_rule(prim, 2)((t([[0, 1, 2, 3], [0, 1, 2, 3]], np.int32), 0)),
          [[0, 3, 1, 2], [0, 3, 1, 2]])
    check(V.get_data_format_dim_map_vmap_rule(P.DataFormatDimMap(), 2)((t([0, 1, 2, 3], np.int32), None)), [0, 3, 1, 2], None)
    prim = M.SparseSegmentMean().add_prim_attr('batch_rank', 0)
    check(W.get_sparse_segment_mean_vmap_rule(prim, 1)((t(x[:1]), 0), (t([0, 2], np.int32), None),
              (t([0, 0], np.int32), None)), [[2]])
    prim = M.SquareSumAll().add_prim_attr('batch_rank', 0)
    result = W.get_square_sum_all_vmap_rule(prim, 2)((t(x), 0), (t(x), 0))
    check(result[0], (x ** 2).sum(axis=1))
    check(result[1], (x ** 2).sum(axis=1))


@pytest.mark.skip(reason='The unmapped Fills primitive has no CPU kernel')
def test_unmapped_fills_values():
    check(V.get_fills_vmap_rule(A.Fills(), 2)((t([1, 2]), None), (3., None)), [3, 3], None)


@pytest.mark.xfail(strict=True, reason='FFTWithSize string factory does not supply required primitive inputs')
def test_unmapped_fft_string_factory_values():
    result = W.get_fft_with_size_vmap_rule('FFTWithSize', 2)((t([1, 2], np.complex64), None))
    check(result, [3, -1], None)


def test_im2col_and_padding_string_protocol_boundaries():
    for factory, name, x in [(V.get_im2col_vmap_rule, 'Im2Col', t(np.zeros((2, 1, 1, 3, 3)))),
                              (V.get_padding_vmap_rule, 'Padding', t(np.zeros((2, 1))))]:
        with pytest.raises((RuntimeError, ValueError), match=r'The pointer\[value\] is null|attr|Attr|ksizes|pad_dim_size'):
            result = factory(name, 2)((x, 0))
            result[0].asnumpy()


def test_gatherd_unmapped_source_and_positive_gradient_axis():
    # Singleton sample axes permit both source/index layouts to coincide.
    x = np.array([[1, 2], [3, 4]], np.float32)
    index = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], np.int32)
    expected = np.zeros((2, 2, 3), np.float32)
    expected[:, 0, 1] = [2, 4]
    expected[:, 1, 0] = [3, 5]
    idx = np.array([[[1], [0]], [[1], [0]]], np.int32)
    grad = np.array([[[2], [3]], [[4], [5]]], np.float32)
    # Batch is after the logical reduction axis, exercising positive unchanged dim.
    out, dim = V.get_gatherd_grad_v2_vmap_rule('GatherDGradV2', 2)((t(np.zeros((2, 3, 2))), 2),
                 (1, None), (t(np.moveaxis(idx, 0, 2), np.int32), 2), (t(np.moveaxis(grad, 0, 2)), 2))
    assert dim == 2
    np.testing.assert_array_equal(np.moveaxis(out.asnumpy(), dim, 0), expected)


def test_minimum_second_gradient_unmapped_sensitivity():
    x = t([[1, 4], [5, 2]])
    y = t([[3, 3], [3, 3]])
    result = D.get_broadcast_grad_grad_vmap_rule(G.MinimumGradGrad(), 2)(
        (x, None), (y, None), (t([[10, 20], [30, 40]]), None), (t([[1, 2], [3, 4]]), None))
    for tensor, dim in result[:2]:
        assert tensor.shape == (2, 2)
        assert tensor.dtype == ms.float32
        assert dim is None
    check(result[2], [[10, 2], [3, 40]], None)


@pytest.mark.xfail(strict=True, reason='MinimumGradGrad CPU clears only one byte of its first two output buffers')
def test_minimum_second_gradient_zero_outputs_regression():
    """Correct complete oracle exposes the CPU kernel's unwritten zero outputs."""
    x = t(np.arange(32).reshape(2, 16))
    y = t(np.full((2, 16), 10))
    a = t(np.full((2, 16), 3))
    b = t(np.full((2, 16), 7))
    # Reuse real device allocations containing nonzero values, exposing the
    # native memset byte-count defect instead of relying on fresh zero pages.
    buffers = [ops.ones_like(x) * 17 for _ in range(128)]
    for value in buffers:
        np.testing.assert_array_equal(value.asnumpy(), np.full((2, 16), 17))
    del value, buffers
    result = D.get_broadcast_grad_grad_vmap_rule(G.MinimumGradGrad(), 2)((x, 0), (y, 0), (a, 0), (b, 0))
    check(result[2], np.where(np.arange(32).reshape(2, 16) <= 10, 3, 7))
    check(result[0], np.zeros((2, 16)))
    check(result[1], np.zeros((2, 16)))


@pytest.mark.xfail(strict=True, reason='FFTWithSize object rule passes a norm enum where a string is required')
def test_fft_explicit_legacy_axis_attribute():
    # The object rule reads the legacy spelling; preserve the real primitive's
    # current onesided configuration and explicitly supply the expected alias.
    prim = M.FFTWithSize(1, False, False)
    prim.add_prim_attr('oneside', prim.onesided)
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.complex64)
    check(W.get_fft_with_size_vmap_rule(prim, 2)((t(data, np.complex64), 0)), np.fft.fft(data))


@pytest.mark.xfail(strict=True, reason='Tile pads repeats but computes output batch metadata from the unpadded rank')
def test_tile_short_multiples_batch_axis_regression():
    logical = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    out, dim = V.get_tile_vmap_rule(P.Tile(), 2)((t(logical), 0), ((2,), None))
    assert dim == 0
    np.testing.assert_array_equal(out.asnumpy(), np.tile(logical, (1, 1, 2)))


@pytest.mark.xfail(strict=True, reason='GatherD discards the return value when moving mismatched batch axes')
def test_gatherd_mismatched_batch_axes_regression():
    x = np.array([[1, 2], [3, 4]], np.float32)
    index = np.array([[0, 1], [1, 0]], np.int32)
    check(V.get_gatherd_vmap_rule('GatherD', 2)((t(x.T), 1), (0, None), (t(index, np.int32), 0)),
          np.take_along_axis(x, index, axis=1))


@pytest.mark.xfail(strict=True, reason='ReverseSequence leaves the logical batch axis ahead of the sequence axis')
def test_reverse_sequence_axis_order_regression():
    logical = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    lengths = np.array([[2, 3], [1, 2]], np.int32)
    expected = logical.copy()
    for i in range(2):
        for j in range(2):
            expected[i, :lengths[i, j], j] = logical[i, :lengths[i, j], j][::-1]
    result = V.get_reverse_sequence_vmap_rule(P.ReverseSequence(0, 1), 2)(
        (t(np.moveaxis(logical, 0, 2)), 2), (t(lengths, np.int32), 0))
    check(result, expected)


def test_reverse_sequence_string_attribute_boundary():
    with pytest.raises(AttributeError, match='batch_dim_'):
        V.get_reverse_sequence_vmap_rule('ReverseSequence', 2)


def test_stateful_scatter_unmapped_monad_boundary():
    ref = ms.Parameter(t([1, 2, 3]), name='scatter_unmapped_vmap')
    result = V.get_scatter_op_vmap_rule('ScatterAdd', 2)((ref, None), (t([0, 2], np.int32), None),
            (t([10, 20]), None), UMonad())
    check(result, [11, 2, 23], None)
    np.testing.assert_array_equal(ref.asnumpy(), [11, 2, 23])


@pytest.mark.xfail(strict=True, reason='MedianGrad recreates the primitive with logical axis 0 after moving batch to front')
def test_median_gradient_positive_axis_regression():
    x = t([[1, 5, 3], [8, 2, 6]])
    check(D.get_median_grad_vmap_rule(G.MedianGrad(False, 0, False), 2)((t([10, 20]), 0), (x, 0),
          (t([3, 6]), 0), (t([2, 2], np.int64), 0)), [[0, 0, 10], [0, 0, 20]])


def test_eager_preprocess_requires_primitive_argument():
    with pytest.raises(TypeError, match='callable primitive'):
        _eager_preprocess((t([1, 2]), 0), (t([1, 2]), None))


@pytest.mark.xfail(strict=True, reason='GatherD vmap compares an unmapped source batch axis None to an integer')
def test_gatherd_broadcasts_unmapped_source():
    data = np.array([[1, 2], [3, 4]], np.float32)
    indices = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], np.int32)
    result = V.get_gatherd_vmap_rule('GatherD', 2)(
        (t(data), None), (0, None), (t(indices, np.int32), 0))
    check(result, np.stack([np.take_along_axis(data, batch, axis=0) for batch in indices]))
