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

"""Behavioral coverage of public function wrappers on the CPU backend.

Numeric checks use NumPy or analytic derivatives. Backend boundary spies, where
present, test Python argument forwarding only and do not claim native coverage.
"""
# Unit coverage intentionally exercises internal Python inference and wrapper helpers.
# pylint: disable=protected-access
import importlib
import random
import numpy as np
import pytest
import mindspore as ms
from mindspore import dataset
from mindspore import Tensor, Parameter, ops
from mindspore.ops.function import array_func as a
from mindspore.ops.function import random_func as r
from mindspore.ops.function import clip_func as c
from mindspore.ops.function import linalg_func as l
from mindspore.ops.function import spectral_func as s
from mindspore.ops.function import debug_func as d
from mindspore.ops.function import other_func as o

g = importlib.import_module('mindspore.ops.function.grad.grad_func')


def t(value, dtype=ms.float32):
    return Tensor(value, dtype)


def eq(actual, expected, **kwargs):
    np.testing.assert_allclose(actual.asnumpy(), expected, rtol=2e-5, atol=2e-5, **kwargs)


@pytest.fixture(autouse=True, name='preserve_context')
def fixture_preserve_context():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    dataset_seed = dataset.config.get_seed()
    generator_manual = r.default_generator._generator.attrs["manual_seed"]
    mode = ms.get_context('mode')
    device = ms.get_context('device_target')
    state = ms.get_rng_state()
    seed = ms.get_seed()
    # _get_graph_seed updates this dictionary even for explicit per-op seeds.
    seed_module = importlib.import_module('mindspore.common.seed')
    op_seeds = seed_module._KERNEL_SEED.copy()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    yield
    ms.set_rng_state(state)
    r.default_generator._generator.add_prim_attr("manual_seed", generator_manual)
    dataset.config.set_seed(dataset_seed)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    seed_module._GLOBAL_SEED = seed
    seed_module._KERNEL_SEED.clear()
    seed_module._KERNEL_SEED.update(op_seeds)
    ms.set_context(mode=mode, device_target=device)


@pytest.mark.parametrize('call,exception', [
    (lambda: a.arange(t([1, 2]), 5), ValueError),
    (lambda: a._get_max_type('bad', 2, 1), TypeError),
    (lambda: a.full((2,), 'bad'), TypeError),
    (lambda: a.full((2,), 1, dtype='bad'), TypeError),
    (lambda: a.full_like([1], 2), TypeError),
    (lambda: a.chunk([1], 2), TypeError),
    (lambda: a.chunk(t([1]), '2'), TypeError),
    (lambda: a.chunk(t([1]), 0), ValueError),
    (lambda: a.fills(t([1]), t([1])), ValueError),
    (lambda: a.fills(t([1]), 'bad'), TypeError),
    (lambda: a.searchsorted(t([1, 2]), t([1]), side='left', right=True), ValueError),
    (lambda: a.flatten([1]), TypeError),
    (lambda: a.flatten(t([1]), start_dim=True), TypeError),
    (lambda: a.flatten(t([[1]]), start_dim=1, end_dim=0), ValueError),
    (lambda: a.index_select(t([1]), 'bad', t([0], ms.int32)), TypeError),
    (lambda: a.index_select(t([1]), 2, t([0], ms.int32)), ValueError),
    (lambda: a.is_nonzero(1), TypeError),
    (lambda: a.is_nonzero(t([1, 2])), ValueError),
    (lambda: a.diagflat([1]), TypeError),
    (lambda: a.diagflat(t([1]), 'bad'), TypeError),
    (lambda: a.split([1], 1), TypeError),
    (lambda: a.split(t([1]), 1, axis=[]), TypeError),
    (lambda: a.split(t([1]), 0), ValueError),
    (lambda: a.split(t([1]), ['1']), TypeError),
    (lambda: a.split(t([1]), [-1]), TypeError),
    (lambda: a.split(t([1]), [2]), ValueError),
    (lambda: a.split(t([1]), 'bad'), TypeError),
    (lambda: a.split_ext(t([1]), 'bad'), TypeError),
    (lambda: a._canonicalize_axis(['x'], 2), TypeError),
    (lambda: a._canonicalize_axis(2, 2), ValueError),
    (lambda: a._canonicalize_axis((0, 0), 2), ValueError),
    (lambda: a.tensor_split([1], 1), TypeError),
    (lambda: a.tensor_split(t([1]), 1, axis=[]), TypeError),
    (lambda: a.tensor_split(t([1]), 0), ValueError),
    (lambda: a.vsplit([1], 1), TypeError),
    (lambda: a.vsplit(t(1), 1), ValueError),
    (lambda: a.hsplit([1], 1), TypeError),
    (lambda: a.dsplit([1], 1), TypeError),
    (lambda: a.dsplit(t([[1]]), 1), ValueError),
    (lambda: a.max(t([1]), axis=0, initial="bad"), TypeError),
    (lambda: a.max(t([1]), axis='bad'), TypeError),
    (lambda: a.min(t([1]), axis=0, initial="bad"), TypeError),
    (lambda: a.min(t([1]), axis='bad'), TypeError),
    (lambda: a.max(t([1]), axis=0, where=t([True], ms.bool_)), ValueError),
    (lambda: a.column_stack(t([1])), TypeError),
    (lambda: a.column_stack([1]), TypeError),
    (lambda: a.column_stack([]), ValueError),
    (lambda: a.hstack(t([1])), TypeError),
    (lambda: a.hstack([1]), TypeError),
    (lambda: a.hstack([]), ValueError),
    (lambda: a.movedim(t([[1]]), (0,), (0, 1)), ValueError),
    (lambda: a.swapaxes([1], 0, 0), TypeError),
    (lambda: a.repeat_interleave(t([1]), 2, axis='bad'), TypeError),
    (lambda: a.repeat_interleave(t([1]), [1.5]), TypeError),
    (lambda: a.repeat_interleave(t([1, 2]), [1, 2, 3]), ValueError),
    (lambda: a.max_ext(t([1]), dim=None, keepdim=True), ValueError),
    (lambda: a.min_ext(t([1]), dim=None, keepdim=True), ValueError),
])
def test_array_validation(call, exception):
    """
    Feature: Array validation.
    Description: Check array validation.
    Expectation: Invalid array arguments raise the parameterized exception type.
    """
    with pytest.raises(exception):
        call()


@pytest.mark.parametrize('axis,flags,exception', [
    ((0, 'x'), (False, True, False), TypeError),
    (['x'], (False, False, True), TypeError),
    ('x', (True, True, True), TypeError),
])
def test_axis_type_validation(axis, flags, exception):
    """
    Feature: Axis type validation.
    Description: Check axis type validation.
    Expectation: Invalid axis type arguments raise the parameterized exception type.
    """
    with pytest.raises(exception):
        a._check_axis_type(axis, *flags, 'coverage')


def test_axis_helpers_and_creation():
    """
    Feature: Axis helpers and creation.
    Description: Check axis helpers and creation.
    Expectation: Axis normalization and array creation return the explicit shapes, dtypes, and literal tensor
        values.
    """
    assert a._check_axis_type((0, 1), False, True, False, 'coverage') is True
    assert a._check_axis_type([0, 1], False, False, True, 'coverage') is True
    assert a._check_axis_valid(None, 3) == (0, 1, 2)
    assert a._check_diagonal_axes(0, -1, 3) == (0, 2)
    assert a._get_type(t(1, ms.int32)) == ms.int32
    eq(a._cast_type(t([1], ms.int32), ms.float32), [1])
    eq(a.full([2], 4), [4, 4])
    assert a.full([2], 4).dtype == ms.int64
    eq(a.full_like(t([1, 2]), 5), [5, 5])
    eq(a.tuple_to_array((1., 2.)), [1, 2])
    assert a.is_tensor(t(1)) and not a.is_tensor(1)
    assert a.is_nonzero(t([[2]])) is True
    assert a.is_nonzero(t(0)) is False
    assert a.size(t([[1, 2]])) == 2
    eq(a.dyn_shape(t([[1, 2]])), [1, 2])


@pytest.mark.parametrize('n,chunks', [(0, 2), (2, 4), (8, 3), (6, 4), (6, 3)])
def test_chunk_numerical(n, chunks):
    """
    Feature: Chunk numerical.
    Description: Check chunk numerical.
    Expectation: Chunking preserves values and matches the independently specified section sizes, including
        empty inputs and excess chunk counts.
    """
    x = np.arange(n, dtype=np.float32)
    result = a.chunk(t(x), chunks)
    if n == 0:
        assert all(v.size == 0 for v in result)
    else:
        width = int(np.ceil(n / chunks))
        expected = [x[i:i + width] for i in range(0, n, width)]
        assert len(result) == len(expected)
        for actual, value in zip(result, expected):
            eq(actual, value)


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason='Fills has no CPU kernel in MS 2.12')
@pytest.mark.parametrize('value', [2, 2., t(2)])
def test_fills_numeric(value):
    """
    Feature: Fills numeric.
    Description: Check fills numeric.
    Expectation: Fills preserves the input shape and fills every element with the converted value two;
        unavailable CPU execution retains its strict expected failure.
    """
    eq(a.fills(t([[1, 3], [4, 5]]), value), np.full((2, 2), 2))


def test_array_numeric_wrappers():
    """
    Feature: Array numeric wrappers.
    Description: Check array numeric wrappers.
    Expectation: Array wrappers preserve independently specified flattening, stacking, diagonal, scatter-free
        reduction, unique, and split values.
    """
    for order in ('C', 'F'):
        eq(a.flatten(t(4), order=order), [4])
        eq(a.flatten(t([1, 2]), start_dim=0, order=order), [1, 2])
    eq(a.column_stack([t(1), t(2)]), [[1, 2]])
    eq(a.column_stack([t([1, 2]), t([3, 4])]), [[1, 3], [2, 4]])
    eq(a.swapdims(t([[1, 2], [3, 4]]), 0, 1), [[1, 3], [2, 4]])
    eq(a.argmax(t(True, ms.bool_)), 0)
    eq(a.argmax(t([False, True], ms.bool_)), 1)
    eq(a.matrix_diag(t([1, 2]), num_rows=2, num_cols=2, padding_value=0.), [[1, 0], [0, 2]])
    eq(a.matrix_set_diag(t([[1, 2], [3, 4]]), t([5, 6])), [[5, 2], [3, 6]])
    eq(a.population_count(t([0, 3, 7], ms.int16)), [0, 2, 3])
    vals, indices = a.top_k(t([2, 5, 1]), 2)
    eq(vals, [5, 2])
    eq(indices, [1, 0])
    eq(a.sequence_mask(t([1, 3], ms.int32)), [[1, 0, 0], [1, 1, 1]])
    eq(a.sequence_mask(t([1, 3], ms.int32), 4), [[1, 0, 0, 0], [1, 1, 1, 0]])
    eq(a.unsorted_segment_min(t([3, 2, 6]), t([0, 0, 1], ms.int32), 2), [2, 6])
    eq(a.unsorted_segment_max(t([3, 2, 6]), t([0, 0, 1], ms.int32), 2), [3, 6])
    eq(a.unsorted_segment_prod(t([3, 2, 6]), t([0, 0, 1], ms.int32), 2), [6, 6])
    out, idx = a.unique_with_pad(t([1, 1, 2], ms.int32), 9)
    eq(out, [1, 2, 9])
    eq(idx, [0, 0, 1])
    for n, sections in [(0, 2), (2, 4), (2, 1)]:
        for got, expected in zip(a.tensor_split(t(np.arange(n)), sections), np.array_split(np.arange(n), sections)):
            eq(got, expected)
    for n, width in [(0, 2), (2, 5)]:
        result = a.split(t(np.arange(n)), width)
        eq(ops.concat(result), np.arange(n))


@pytest.mark.parametrize('name,expected', [('mul', [6, 3]), ('max', [3, 3]),
                         ('min', [2, 3]), ('div', [2/3, 3]), ('update', [3, 3])])
def test_scatter_parameter_wrappers(name, expected):
    """
    Feature: Scatter parameter wrappers.
    Description: Check scatter parameter wrappers.
    Expectation: Scatter updates return the expected arithmetic values and apply the same mutation to the
        original Parameter.
    """
    x = Parameter(t([2, 3]))
    eq(getattr(a, 'scatter_' + name)(x, t([0], ms.int32), t([3])), expected)
    eq(x, expected)


@pytest.mark.parametrize('name,expected', [('add', [5, 3]), ('sub', [-1, 3]),
                         ('mul', [6, 3]), ('div', [2/3, 3]), ('max', [3, 3]), ('min', [2, 3])])
def test_scatter_nd_parameter_wrappers(name, expected):
    """
    Feature: Scatter nd parameter wrappers.
    Description: Check scatter nd parameter wrappers.
    Expectation: ND scatter updates produce the explicit expected values in both the returned result and
        original Parameter.
    """
    x = Parameter(t([2, 3]))
    eq(getattr(a, 'scatter_nd_' + name)(x, t([[0]], ms.int32), t([3])), expected)
    eq(x, expected)


@pytest.mark.parametrize('name,expected', [('mul', [6, 3]), ('max', [3, 3]), ('min', [2, 3])])
def test_tensor_scatter_numeric(name, expected):
    """
    Feature: Tensor scatter numeric.
    Description: Check tensor scatter numeric.
    Expectation: Tensor scatter returns the independently specified updated values while preserving the original
        tensor.
    """
    x = t([2, 3])
    eq(getattr(a, 'tensor_scatter_' + name)(x, t([[0]], ms.int32), t([3])), expected)
    eq(x, [2, 3])


@pytest.mark.parametrize('kwargs,error', [({'window_length': 1.5}, TypeError), ({'window_length': -1}, ValueError),
    ({'window_length': 3, 'periodic': 1}, TypeError), ({'window_length': 3, 'alpha': 1}, TypeError),
    ({'window_length': 3, 'beta': 1}, TypeError), ({'window_length': 3, 'dtype': ms.int32}, TypeError)])
def test_hamming_validation(kwargs, error):
    """
    Feature: Hamming validation.
    Description: Check hamming validation.
    Expectation: Invalid hamming arguments raise the parameterized exception type.
    """
    with pytest.raises(error):
        a.hamming_window(**kwargs)


def test_windows_numeric():
    """
    Feature: Windows numeric.
    Description: Check windows numeric.
    Expectation: Symmetric and periodic Hamming, Blackman, and Bartlett windows match the independent NumPy
        references.
    """
    for n in (0, 1, 5):
        eq(a.hamming_window(n, periodic=False), np.hamming(n))
    eq(a.hamming_window(5), np.hamming(6)[:-1])
    eq(s.blackman_window(t(5, ms.int32), periodic=False), np.blackman(5))
    eq(s.bartlett_window(t(5, ms.int32), periodic=False), np.bartlett(5))


@pytest.mark.parametrize('call,error', [
    (lambda: c.clip_by_value(t([1])), ValueError), (lambda: c.clip_by_value(1, 0, 2), TypeError),
    (lambda: c.clip_by_value([1], 0, 2), TypeError), (lambda: c.clip_by_value(t([1]), 'bad', 2), TypeError),
    (lambda: c.clip_by_value(t([1]), 0, 'bad'), TypeError),
    (lambda: c.clip_by_norm(t([float('nan')]), 1, error_if_nonfinite=True), RuntimeError),
    (lambda: c.clip_by_global_norm((t([1]),), use_norm=1), ValueError),
])
def test_clip_validation(call, error):
    """
    Feature: Clip validation.
    Description: Check clip validation.
    Expectation: Invalid clip arguments raise the parameterized exception type.
    """
    with pytest.raises(error):
        call()


def test_clip_numeric():
    """
    Feature: Clip numeric.
    Description: Check clip numeric.
    Expectation: Clipping and norm clipping match the explicit independently calculated bounds and scale
        factors.
    """
    x = t([-2, 1, 4])
    eq(c.clip_by_value(x, -1, 2), [-1, 1, 2])
    eq(c.clip_by_value(x, None, 2), [-2, 1, 2])
    eq(c.clip(x, -1, None), [-1, 1, 4])
    for values in ([x, x], (x, x)):
        result = c.clip_by_value(values, -1., t(2))
        assert isinstance(result, tuple)
        for item in result:
            eq(item, [-1, 1, 2])
    eq(c.clip_by_norm(t([3, 4]), 10), [3, 4])
    eq(c.clip_by_norm(t([3, 4]), 2), [1.2, 1.6])
    result = c.clip_by_norm((t([3]), t([4])), 2, norm_type=float('inf'))
    eq(result[0], [1.5])
    eq(result[1], [2.])
    result = c.clip_by_global_norm((t([3]), t([4])), 2)
    eq(result[0], [1.2])
    eq(result[1], [1.6])


@pytest.mark.parametrize('call,error', [(lambda:l.pinv([1]), TypeError), (lambda:l.pinv(t(1)), TypeError),
    (lambda:l.pinv(t([1])), ValueError), (lambda:l.qr(t([[1]]), 'invalid'), TypeError),
    (lambda:l._nd_transpose(t([1])), TypeError)])
def test_linalg_validation(call, error):
    """
    Feature: Linalg validation.
    Description: Check linalg validation.
    Expectation: Invalid linalg arguments raise the parameterized exception type.
    """
    with pytest.raises(error):
        call()


def test_linalg_numeric():
    """
    Feature: Linalg numeric.
    Description: Check linalg numeric.
    Expectation: Linear-algebra wrappers agree with independent matrix formulas and NumPy results for each
        supported case.
    """
    x = np.array([[3., 1.], [1., 2.]], np.float32)
    eq(l.cond(t(x)), np.linalg.cond(x))
    values, vectors = l.eig(t(x))
    np.testing.assert_allclose(x @ vectors.asnumpy(), vectors.asnumpy() * values.asnumpy(), atol=1e-5)
    for hermitian in (False, True):
        eq(l.pinv(t(x), atol=t(0., ms.float64), rtol=t(1e-6, ms.float64), hermitian=hermitian), np.linalg.pinv(x))
        eq(l.pinv(t(x), atol=0., hermitian=hermitian), np.linalg.pinv(x))
    for mode in ('reduced', 'complete'):
        q, rr = l.qr(t(x), mode)
        eq(q @ rr, x)
        eq(q.T @ q, np.eye(2))


@pytest.mark.parametrize('args,error', [((1, t(1)), TypeError), (('dump', 1), TypeError),
    (('dump', t(1), 1), TypeError), (('dump', t(1), 'all'), ValueError), (('dump', t(1), 'bad'), ValueError)])
def test_tensordump_validation(args, error):
    """
    Feature: Tensordump validation.
    Description: Check tensordump validation.
    Expectation: Invalid tensordump arguments raise the parameterized exception type.
    """
    with pytest.raises(error):
        d.tensordump(*args)


def test_other_functions_numeric():
    """
    Feature: Other functions numeric.
    Description: Check other functions numeric.
    Expectation: Dependency preserves the input tensor values and partial application forwards the bound operand
        to addition.
    """
    x = t([1, 2])
    eq(o.depend(x, t(3)), [1, 2])
    add = o.partial(ops.add, t(2))
    eq(add(t(3)), 5)


@pytest.mark.parametrize('call,error', [
    (lambda:r.uniform((2,), t(0), t(1), dtype=ms.int64), TypeError),
    (lambda:r.rand(2, dtype=ms.int32), ValueError), (lambda:r.randn(2, dtype=ms.int32), ValueError),
    (lambda:r.rand_like([1]), TypeError), (lambda:r.rand_like(t([1]), dtype=ms.int32), ValueError),
    (lambda:r.randn_like([1]), TypeError), (lambda:r.randn_like(t([1]), dtype=ms.int32), ValueError),
    (lambda:r.randint(0, 2, (2,), dtype=ms.float32), ValueError),
    (lambda:r.randint(0, 2, [2]), ValueError), (lambda:r.randint(True, 2, (2,)), TypeError),
    (lambda:r.randint(0, False, (2,)), TypeError),
    (lambda:r.randint_like([1], 0, 2), TypeError), (lambda:r.randint_like(t([1]), 0, 2), ValueError),
    (lambda:r.randint_like(t([1], ms.int32), True, 2), TypeError),
    (lambda:r.randint_like(t([1], ms.int32), 0, False), TypeError),
    (lambda:r._generate_shapes(('x',)), TypeError),
    (lambda:r._generate_shapes((1, 'x')), TypeError),
    (lambda:r._check_shape([1]), TypeError), (lambda:r._check_shape(('x',)), TypeError),
    (lambda:r._check_shape((0,)), ValueError), (lambda:r._check_param('normal', 'mean', 'x'), TypeError),
    (lambda:r.multinomial_with_replacement(t([1, 2]), seed='x', offset=0, numsamples=1), TypeError),
    (lambda:r.multinomial_with_replacement(t([1, 2]), seed=1, offset='x', numsamples=1), TypeError),
])
def test_random_validation(call, error):
    """
    Feature: Random validation.
    Description: Check random validation.
    Expectation: Invalid random arguments raise the parameterized exception type.
    """
    with pytest.raises(error):
        call()


def test_random_shapes_and_support():
    """
    Feature: Random shapes and support.
    Description: Check random shapes and support.
    Expectation: Random wrappers return the expected shapes/dtypes and values remain within their documented
        distribution support.
    """
    assert r._generate_shapes(()) == (1,)
    assert r._check_shape((1, 2)) is True
    assert r._check_param('normal', 'mean', np.array(1.)) is True
    assert r.is_cpu_backend() is True
    for fn in (r.rand_like, r.randn_like):
        out = fn(t(np.zeros((2, 3))), seed=71)
        assert out.shape == (2, 3) and out.dtype == ms.float32
        assert np.isfinite(out.asnumpy()).all()
        if fn is r.rand_like:
            assert np.all((out.asnumpy() >= 0) & (out.asnumpy() < 1))
    for dtype in (None, ms.int32):
        out = r.randint(-2, 5, (3, 4), seed=71, dtype=dtype)
        assert out.shape == (3, 4)
        assert np.all((out.asnumpy() >= -2) & (out.asnumpy() < 5))
        assert out.dtype == (dtype or ms.int64)
    out = r.randint_like(t(np.zeros((2, 3)), ms.int32), 2, 7, seed=71)
    assert out.shape == (2, 3) and out.dtype == ms.int32
    assert np.all((out.asnumpy() >= 2) & (out.asnumpy() < 7))
    out = r.uniform((3, 4), t(2, ms.int32), t(5, ms.int32), seed=71, dtype=ms.int32)
    assert np.all((out.asnumpy() >= 2) & (out.asnumpy() < 5))
    out = r.normal((2, 3), mean=4., stddev=0., seed=71)
    eq(out, np.full((2, 3), 4.))
    out = r.normal((2, 3), mean=t([1, 2, 3]), stddev=t(0), seed=71)
    eq(out, [[1, 2, 3], [1, 2, 3]])
    out = r.laplace((2, 3), t(2), t(0), seed=71)
    eq(out, np.full((2, 3), 2.))
    shuffled = r.shuffle(t([[1, 2], [3, 4], [5, 6]]), seed=71)
    assert sorted(map(tuple, shuffled.asnumpy())) == [(1., 2.), (3., 4.), (5., 6.)]
    indices, mask = r.choice_with_mask(t([[True, False], [False, True]], ms.bool_), count=3, seed=71)
    chosen = indices.asnumpy()[mask.asnumpy()]
    assert sorted(map(tuple, chosen)) == [(0, 0), (1, 1)]
    categories = r.random_categorical(t([[0., -1000.], [-1000., 0.]]), 4, seed=71)
    eq(categories, [[0, 0, 0, 0], [1, 1, 1, 1]])
    out = r.multinomial_with_replacement(t([0., 1.]), numsamples=4, seed=71, offset=0, replacement=True)
    eq(out, [1, 1, 1, 1])
    for data in ([0., 1.], [[0., 1.], [1., 0.]]):
        out = r.multinomial(t(data), 1, replacement=False, seed=71)
        eq(out, [1] if np.ndim(data) == 1 else [[1], [0]])


@pytest.mark.parametrize('call,error', [
    (lambda:g._check_duplicate_grad_position((0, 0)), ValueError),
    (lambda:g._convert_grad_position_type(('x',)), TypeError),
    (lambda:g._convert_grad_position_type((-1,)), ValueError),
    (lambda:g._convert_grad_position_type(-1), ValueError),
    (lambda:g._check_grad_position(2, 2), ValueError),
    (lambda:g._check_jvp_input_v_len(1, 2), ValueError),
    (lambda:g._check_tensor((t(1), 1)), TypeError),
    (lambda:g._check_jet_inputs([t(1)], t(1)), TypeError),
    (lambda:g._check_jet_inputs(t([1, 2]), t([1, 2, 3])), ValueError),
    (lambda:g._check_jet_inputs((t(1),), (t(1), t(2))), ValueError),
    (lambda:g._trans_jet_inputs(t(1, ms.float16), t(1, ms.float16)), TypeError),
])
def test_grad_validation(call, error):
    """
    Feature: Grad validation.
    Description: Check grad validation.
    Expectation: Invalid grad arguments raise the parameterized exception type.
    """
    with pytest.raises(error):
        call()


def square(x):
    return x * x


def polynomial(x, y):
    return x * x + 3 * y


def multiple_outputs(x, y):
    return x * x + y, x * y


def auxiliary(x):
    return x * x, x + 7


def auxiliary_three(x, y):
    return x * x + 3 * y, x + 7, y - 2


def test_grad_transforms_and_derivative():
    """
    Feature: Grad transforms and derivative.
    Description: Check grad transforms and derivative.
    Expectation: Gradient transforms preserve primal values and return the independently calculated analytic
        derivatives.
    """
    assert g._convert_grad_position_type((False, True)) == (0, 1)
    primal, series = g._trans_jet_inputs(t([2], ms.int32), t([[1]], ms.int32))
    assert primal.dtype == series.dtype == ms.float32
    eq(primal, [2])
    eq(series, [[1]])
    eq(g._trans_derivative_inputs(t([3], ms.int64)), [3])
    primal, series = g._check_jet_inputs(t([2]), t([1]))
    eq(primal, [2])
    eq(series, [[1]])
    primal, derivative = g.derivative(square, t([2, 3], ms.int32), order=1)
    eq(primal, [4, 9])
    eq(derivative, [4, 6])


@pytest.mark.parametrize('kind', ['single', 'single_aux', 'multi', 'multi_aux', 'tuple_output'])
def test_jvp_analytic(kind):
    """
    Feature: Jvp analytic.
    Description: Check jvp analytic.
    Expectation: Single/multiple input and output JVPs match analytic directional derivatives and retain the
        expected auxiliary values.
    """
    x, y, v, w = t([2, 3]), t([4, 5]), t([1, 2]), t([3, 4])
    if kind == 'single':
        out, tangent = g.jvp(square, x, v)
        eq(out, [4, 9])
        eq(tangent, [4, 12])
    elif kind == 'single_aux':
        out, tangent, aux = g.jvp(auxiliary, x, v, has_aux=True)
        eq(out, [4, 9])
        eq(tangent, [4, 12])
        eq(aux, [9, 10])
    elif kind == 'multi':
        out, tangent = g.jvp(polynomial, [x, y], [v, w])
        eq(out, [16, 24])
        eq(tangent, [13, 24])
    elif kind == 'multi_aux':
        out, tangent, aux = g.jvp(auxiliary_three, (x, y), (v, w), has_aux=True)
        eq(out, [16, 24])
        eq(tangent, [13, 24])
        eq(aux[0], [9, 10])
        eq(aux[1], [2, 3])
    else:
        out, tangent = g.jvp(multiple_outputs, (x, y), (v, w))
        eq(out[0], [8, 14])
        eq(out[1], [8, 15])
        eq(tangent[0], [7, 16])
        eq(tangent[1], [10, 22])


def test_linearize_vjp_analytic():
    """
    Feature: Linearize vjp analytic.
    Description: Check linearize vjp analytic.
    Expectation: Linearization and VJP closures return the expected primal values, auxiliary outputs, and
        analytic tangent/cotangent products.
    """
    x, y, v = t([2, 3]), t([4, 5]), t([1, 2])
    out, linear = g.linearize(square, x)
    eq(out, [4, 9])
    eq(linear(v), [4, 12])
    out, linear = g.linearize(polynomial, (x, y))
    eq(out, [16, 24])
    eq(linear((v, v)), [7, 18])
    out, backward, aux = g.vjp(auxiliary, x, has_aux=True)
    eq(out, [4, 9])
    eq(aux, [9, 10])
    eq(backward(v)[0], [4, 12])


@pytest.mark.parametrize('fn', [g.jacfwd, g.jacrev])
@pytest.mark.parametrize('case', ['single', 'two_inputs', 'two_outputs', 'aux', 'selected'])
def test_jacobians_analytic(fn, case):
    """
    Feature: Jacobians analytic.
    Description: Check jacobians analytic.
    Expectation: Forward and reverse Jacobians match the independent diagonal/identity matrices for selected
        inputs, outputs, and auxiliary modes.
    """
    x, y = t([2, 3]), t([4, 5])
    diagonal = np.diag([4., 6.])
    if case == 'single':
        eq(fn(square)(x), diagonal)
    elif case == 'two_inputs':
        result = fn(polynomial, grad_position=(0, 1))(x, y)
        eq(result[0], diagonal)
        eq(result[1], 3 * np.eye(2))
    elif case == 'two_outputs':
        result = fn(multiple_outputs, grad_position=(0, 1))(x, y)
        eq(result[0][0], diagonal)
        eq(result[0][1], np.eye(2))
        eq(result[1][0], np.diag([4., 5.]))
        eq(result[1][1], np.diag([2., 3.]))
    elif case == 'aux':
        result, aux = fn(auxiliary_three, grad_position=(0, 1), has_aux=True)(x, y)
        eq(result[0], diagonal)
        eq(result[1], 3 * np.eye(2))
        eq(aux[0], [9, 10])
        eq(aux[1], [2, 3])
    else:
        eq(fn(polynomial, grad_position=1)(x, y), 3 * np.eye(2))


def test_custom_vjp_protocol_and_gradient():
    """
    Feature: Custom vjp protocol and gradient.
    Description: Check custom vjp protocol and gradient.
    Expectation: Custom forward calls preserve square values, registered backward callbacks return analytic
        gradients, and direct bprop preserves sensitivities.
    """
    @g.custom_vjp
    def fwd(x):
        return x * x

    def backward(x, out, dout):
        return (2 * x * dout,)

    fwd.defbwd(backward)
    eq(fwd(t([2, 3])), [4, 9])
    eq(g.grad(fwd)(t([2, 3])), [4, 6])
    # Direct Cell bprop invocation verifies the Python delegate as graph
    # compilation otherwise executes the equivalent compiled bprop graph.
    eq(fwd.bprop(t([2, 3]), t([4, 9]), t([1, 2]))[0], [4, 12])
    wrapped = g.custom_vjp()(square)
    wrapped.defbwd(backward)
    eq(wrapped(t([2, 3])), [4, 9])


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason='ScalarToArray lacks __infer__ in MS 2.12 eager mode')
def test_scalar_to_array_expected_numeric():
    """
    Feature: Scalar to array expected numeric.
    Description: Check scalar to array expected numeric.
    Expectation: Scalar conversion retains the numeric value 3.0; the unavailable eager inference path retains
        its strict expected failure.
    """
    eq(a.scalar_to_array(3.), 3.)


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason='MoveTo has no CPU kernel in MS 2.12')
def test_move_to_expected_numeric():
    """
    Feature: Move to expected numeric.
    Description: Check move to expected numeric.
    Expectation: Moving to CPU preserves [1, 2]; the unavailable CPU native kernel retains its strict expected
        failure.
    """
    eq(o.move_to(t([1, 2]), 'CPU', True), [1, 2])


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason='Gamma has no CPU kernel in MS 2.12')
def test_gamma_expected_numeric():
    """
    Feature: Gamma expected numeric.
    Description: Check gamma expected numeric.
    Expectation: Gamma samples have shape (2, 3) and are nonnegative, with unsupported CPU execution retaining
        its strict expected failure.
    """
    out = r.gamma((2, 3), t(2), t(1), seed=71)
    assert out.shape == (2, 3) and np.all(out.asnumpy() >= 0)


@pytest.mark.parametrize('value', [2, 2., t(2, ms.int32)])
def test_fills_native_boundary_protocol(monkeypatch, value):
    """
    Feature: Fills native boundary protocol.
    Description: CPU Fills unavailable: verify conversion and full native call only.
    Expectation: CPU Fills unavailable: verify conversion and full native call only.
    """
    x = t([1, 2])
    sentinel = object()
    calls = []
    def native(input_x, fill_value):
        calls.append((input_x, fill_value))
        return sentinel
    monkeypatch.setattr(a, 'fills_', native)
    assert a.fills(x, value) is sentinel
    assert len(calls) == 1 and calls[0][0] is x
    actual = calls[0][1]
    if isinstance(value, Tensor):
        assert actual.dtype == ms.float32 and actual.shape == ()
        eq(actual, 2.)
    else:
        assert isinstance(actual, float) and actual == 2.


def test_move_to_native_boundary_protocol(monkeypatch):
    """
    Feature: Move to native boundary protocol.
    Description: CPU MoveTo unavailable: check full forwarding and return identity.
    Expectation: CPU MoveTo unavailable: check full forwarding and return identity.
    """
    x = t([1, 2])
    sentinel, calls = object(), []
    def native(*args):
        calls.append(args)
        return sentinel
    monkeypatch.setattr(o, 'move_to_', native)
    assert o.move_to(x, 'CPU', False) is sentinel
    assert len(calls) == 1 and calls[0][0] is x
    assert calls[0][1:] == ('CPU', False)


def test_gamma_native_boundary_protocol(monkeypatch):
    """
    Feature: Gamma native boundary protocol.
    Description: CPU Gamma unavailable: check constructor seeds and parameter identity.
    Expectation: CPU Gamma unavailable: check constructor seeds and parameter identity.
    """
    ms.set_seed(37)
    alpha, beta = t(2), t(1)
    sentinel, calls = object(), []
    class NativeGamma:
        def __init__(self, seed1, seed2):
            calls.append(('init', seed1, seed2))
        def add_prim_attr(self, key, value):
            calls.append(('attr', key, value))
            return self
        def add_prim_user_data(self, key, value):
            calls.append(('data', key, value))
            return self
        def __call__(self, *args):
            calls.append(('call', args))
            return sentinel
    monkeypatch.setattr(r.P, 'Gamma', NativeGamma)
    assert r.gamma((2, 3), alpha, beta, seed=71) is sentinel
    assert calls[0][0] == 'init' and all(isinstance(seed, int) for seed in calls[0][1:])
    assert calls[0][1:] == (37, 71)
    assert calls[1] == ('attr', 'random_cache', False)
    assert calls[2][0] == 'call'
    shape, passed_alpha, passed_beta = calls[2][1]
    assert shape == (2, 3) and passed_alpha is alpha and passed_beta is beta


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason='repeat_interleave zero repeats constructs Tensor with reversed arguments in MS 2.12')
def test_repeat_interleave_zero_expected_numeric():
    """
    Feature: Repeat interleave zero expected numeric.
    Description: Check repeat interleave zero expected numeric.
    Expectation: Zero repeats return an empty float32 tensor with shape (0,), subject to the preserved runtime
        expected-failure marker.
    """
    out = a.repeat_interleave(t([1, 2]), 0)
    assert out.shape == (0,) and out.dtype == ms.float32


@pytest.mark.parametrize('fn', [a.unique_ext, a.unique_consecutive])
def test_mutable_unique_flags_rejected(fn, monkeypatch):
    """
    Feature: Mutable unique flags rejected.
    Description: Graph metadata protocol: isconstant rejects a nonconstant graph flag.
    Expectation: Graph metadata protocol: isconstant rejects a nonconstant graph flag.
    """
    flag = ms.mutable(True)
    seen = []
    def graph_isconstant(value):
        seen.append(value)
        assert value is flag
        return False
    monkeypatch.setattr(a.F, 'isconstant', graph_isconstant)
    with pytest.raises(ValueError, match='cannot be mutable'):
        fn(t([1, 1], ms.int32), return_inverse=flag)
    assert len(seen) == 1


@pytest.fixture(name='python_grad_wrappers')
def fixture_python_grad_wrappers(monkeypatch):
    """Run graph wrapper Python bodies directly; all tensor/autograd ops stay real."""
    def direct_jit(fn=None, **kwargs):
        if fn is None:
            return lambda function: function
        return fn
    monkeypatch.setattr(g, 'jit', direct_jit)


@pytest.mark.parametrize('kind', ['single', 'single_aux', 'multi', 'multi_aux'])
def test_jvp_python_wrapper_numeric(python_grad_wrappers, kind):
    """
    Feature: Jvp python wrapper numeric.
    Description: Check jvp python wrapper numeric.
    Expectation: The direct Python JVP wrapper returns the same analytic tangents and auxiliary outputs as the
        reference JVP cases.
    """
    test_jvp_analytic(kind)


def test_linearize_python_wrapper_numeric(python_grad_wrappers):
    """
    Feature: Linearize python wrapper numeric.
    Description: Check linearize python wrapper numeric.
    Expectation: Direct Python linearization and VJP wrappers preserve the analytic primal values and
        tangent/cotangent products.
    """
    test_linearize_vjp_analytic()


@pytest.mark.parametrize('fn', [g.jacfwd, g.jacrev])
@pytest.mark.parametrize('case', ['single', 'two_inputs', 'two_outputs', 'aux', 'selected'])
def test_jacobian_python_wrapper_numeric(python_grad_wrappers, fn, case):
    """
    Feature: Jacobian python wrapper numeric.
    Description: Check jacobian python wrapper numeric.
    Expectation: Direct Python forward/reverse Jacobian wrappers return the analytic matrices for all
        parameterized input/output cases.
    """
    test_jacobians_analytic(fn, case)


def test_grad_python_wrapper_rejection(python_grad_wrappers):
    """
    Feature: Grad python wrapper rejection.
    Description: Check grad python wrapper rejection.
    Expectation: Invalid grad python wrapper rejection raise ValueError. Diagnostics identify more than one
        outputs.
    """
    with pytest.raises(ValueError, match='more than one outputs'):
        g.jvp(square, t([2]), t([1]), has_aux=True)
    for fn in (g.jacfwd, g.jacrev):
        with pytest.raises(ValueError, match='more than one outputs'):
            fn(square, has_aux=True)(t([2]))


def test_derivative_tuple_analytic():
    """
    Feature: Derivative tuple analytic.
    Description: Check derivative tuple analytic.
    Expectation: Tuple-input differentiation returns primal values [16, 24] and first derivatives [7, 9].
    """
    primal, derivative = g.derivative(polynomial, (t([2, 3]), t([4, 5])), order=1)
    eq(primal, [16, 24])
    eq(derivative, [7, 9])


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason='Poisson has no CPU kernel in MS 2.12')
def test_poisson_expected_numeric():
    """
    Feature: Poisson expected numeric.
    Description: Check poisson expected numeric.
    Expectation: Poisson samples have shape (2, 3) and nonnegative integer values, with unavailable CPU
        execution retaining its strict expected failure.
    """
    out = r.poisson((2, 3), t(2), seed=71)
    assert out.shape == (2, 3) and np.all(out.asnumpy() >= 0)
    assert np.all(out.asnumpy() == np.floor(out.asnumpy()))


def test_poisson_native_boundary_protocol(monkeypatch):
    """
    Feature: Poisson native boundary protocol.
    Description: CPU Poisson unavailable; verify complete constructor/call/return protocol.
    Expectation: CPU Poisson unavailable; verify complete constructor/call/return protocol.
    """
    ms.set_seed(37)
    mean, sentinel, calls = t(2), object(), []
    class NativePoisson:
        def __init__(self, seed1, seed2):
            calls.append(('init', seed1, seed2))
        def add_prim_attr(self, key, value):
            calls.append(('attr', key, value))
        def __call__(self, *args):
            calls.append(('call', args))
            return sentinel
    monkeypatch.setattr(r.P, 'Poisson', NativePoisson)
    assert r.poisson((2, 3), mean, seed=71) is sentinel
    assert calls[0][0] == 'init' and all(isinstance(seed, int) for seed in calls[0][1:])
    assert calls[0][1:] == (37, 71)
    assert calls[1] == ('attr', 'random_cache', False)
    shape, actual_mean = calls[2][1]
    assert shape == (2, 3) and actual_mean is mean


@pytest.mark.parametrize('call,error', [
    (lambda:a.full(2, 1), TypeError),
    (lambda:a.index_select(t([1]), 0, [0]), TypeError),
    (lambda:a.index_select(t([1]), 0, t([[0]], ms.int32)), ValueError),
    (lambda:r.uniform((1,), 0, t(1)), TypeError),
    (lambda:g.linearize(square, 1), TypeError),
    (lambda:g.vjp(square, [t(1)]), TypeError),
])
def test_remaining_validation(call, error):
    """
    Feature: Remaining validation.
    Description: Check remaining validation.
    Expectation: Invalid remaining arguments raise the parameterized exception type.
    """
    with pytest.raises(error):
        call()


def test_remaining_array_numeric():
    """
    Feature: Remaining array numeric.
    Description: Check remaining array numeric.
    Expectation: Affine grids, image reconstruction, and pseudoinverses match independently specified coordinate
        and matrix values.
    """
    theta = t([[[1, 0, 0], [0, 1, 0]]])
    eq(a.affine_grid(theta, (1, 1, 2, 2)), [[[[-.5, -.5], [.5, -.5]], [[-.5, .5], [.5, .5]]]])
    eq(a.col2im(t([[[[1, 2, 3, 4]]]]), t([2, 2], ms.int32), (1, 1), (1, 1), (0, 0), (1, 1)),
       [[[[1, 2], [3, 4]]]])
    eq(l.pinv(t([[2., 0.], [0., 4.]]), atol=0., rtol=1e-6), [[.5, 0.], [0., .25]])


def test_unknown_shape_metadata_protocol(monkeypatch):
    """
    Feature: Unknown shape metadata protocol.
    Description: Graph primitive boundary: verify sentinel shapes for dynamic metadata.
    Expectation: Graph primitive boundary: verify sentinel shapes for dynamic metadata.
    """
    shape = (2, 3)
    calls = []
    def unknown_rank(value):
        calls.append(('rank', value))
        return True
    monkeypatch.setattr(a.ops, 'is_sequence_shape_unknown', unknown_rank)
    assert a.get_x_shape(shape) == (-2,)
    assert calls == [('rank', shape)]
    calls.clear()
    monkeypatch.setattr(a.ops, 'is_sequence_shape_unknown', lambda value: False)
    def unknown_dimension(value):
        calls.append(('dimension', value))
        return True
    monkeypatch.setattr(a.ops, 'is_sequence_value_unknown', unknown_dimension)
    assert a.get_x_shape(shape) == (-1,)
    assert calls == [('dimension', shape)]


@pytest.mark.parametrize('repeats', [2, t([1, 2], ms.int64)])
def test_repeat_interleave_ext_native_boundary(monkeypatch, repeats):
    """
    Feature: Repeat interleave ext native boundary.
    Description: Documented Ascend-only operation; CPU checks branch and full protocol.
    Expectation: Documented Ascend-only operation; CPU checks branch and full protocol.
    """
    x, sentinel, calls = t([1, 2]), object(), []
    def native(*args):
        calls.append(args)
        return sentinel
    attr = 'repeat_interleave_int_' if isinstance(repeats, int) else 'repeat_interleave_tensor_'
    monkeypatch.setattr(a, attr, native)
    assert a.repeat_interleave_ext(x, repeats, dim=0, output_size=4) is sentinel
    assert len(calls) == 1 and calls[0][0] is x
    assert calls[0][1] is repeats and calls[0][2:] == (0, 4)


def test_chunk_ext_native_boundary(monkeypatch):
    """
    Feature: Chunk ext native boundary.
    Description: Ascend Chunk boundary, without claiming its native numerical execution.
    Expectation: Ascend Chunk boundary, without claiming its native numerical execution.
    """
    x, sentinel, calls = t([1, 2]), object(), []
    def native(*args):
        calls.append(args)
        return sentinel
    monkeypatch.setattr(a, 'chunk_op', native)
    assert a.chunk_ext(x, 3, dim=-1) is sentinel
    assert len(calls) == 1 and calls[0][0] is x and calls[0][1:] == (3, -1)


@pytest.mark.parametrize('fn,attr', [(r.uniform_ext, 'uniform_ext_'), (r.uniform_, 'inplace_uniform_op')])
@pytest.mark.parametrize('custom', [False, True])
def test_uniform_ext_native_protocol(monkeypatch, fn, attr, custom):
    """
    Feature: Uniform ext native protocol.
    Description: Ascend-only native boundary; generator steps and all native inputs are real.
    Expectation: Ascend-only native boundary; generator steps and all native inputs are real.
    """
    generator = ms.Generator() if custom else r.default_generator
    generator.manual_seed(19)
    x, sentinel, calls = t([1, 2]), object(), []
    def native(*args):
        calls.append(args)
        return sentinel
    monkeypatch.setattr(r, attr, native)
    kwargs = {'generator': generator} if custom else {}
    assert fn(x, 2., 5., **kwargs) is sentinel
    assert len(calls) == 1
    actual_x, low, high, seed, offset = calls[0]
    assert actual_x is x and (low, high) == (2., 5.)
    assert seed.dtype == offset.dtype == ms.int64
    eq(seed, 19)
    eq(offset, 0)
    assert generator.initial_seed() == 19
    eq(generator._offset, 12)


@pytest.mark.parametrize('like', [False, True])
def test_randint_ext_native_protocol(monkeypatch, like):
    """
    Feature: Randint ext native protocol.
    Description: Ascend-only kernel boundary: shorthand inserts low=0 and forwards RNG.
    Expectation: Ascend-only kernel boundary: shorthand inserts low=0 and forwards RNG.
    """
    r.default_generator.manual_seed(23)
    sentinel, calls = object(), []
    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel
    x = t([1, 2], ms.int32)
    monkeypatch.setattr(r, 'randint_like_' if like else 'randint_', native)
    if like:
        assert r.randint_like_ext(x, 5, dtype=ms.int32, device='CPU') is sentinel
    else:
        assert r.randint_ext(5, (2,), dtype=ms.int32, device='CPU') is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    if like:
        assert args[0] is x and args[1:3] == (0, 5)
    else:
        assert args[:3] == (0, 5, (2,))
    eq(args[-2], 23)
    eq(args[-1], 0)
    assert kwargs == {'dtype': ms.int32, 'device': 'CPU'}
    eq(r.default_generator._offset, 12)


def _closed(function, name):
    """Access an actual nested helper for direct graph-Python behavior testing."""
    return dict(zip(function.__code__.co_freevars,
                    (cell.cell_contents for cell in function.__closure__)))[name]


@pytest.mark.parametrize('fn', [g.jacfwd, g.jacrev])
@pytest.mark.parametrize('aux', [False, True])
@pytest.mark.parametrize('multi', [False, True])
def test_jacobian_graph_helpers_direct_numeric(monkeypatch, python_grad_wrappers, fn, aux, multi):
    """
    Feature: Jacobian graph helpers direct numeric.
    Description: Capture helpers while delegating vmap, then test their real Tensor behavior.
    Expectation: Capture helpers while delegating vmap, then test their real Tensor behavior.
    """
    callbacks = []
    native_vmap = g._vmap
    def observe_vmap(callback):
        callbacks.append(callback)
        return native_vmap(callback)
    monkeypatch.setattr(g, '_vmap', observe_vmap)
    function = auxiliary_three if (aux and multi) else auxiliary if aux else polynomial if multi else square
    positions = (0, 1) if multi else 0
    wrapped = fn(function, grad_position=positions, has_aux=aux)
    inputs = (t([2, 3]), t([4, 5])) if multi else (t([2, 3]),)
    result = wrapped(*inputs)
    jacobian = result[0] if aux else result
    eq(jacobian[0] if multi else jacobian, np.diag([4., 6.]))
    assert len(callbacks) == 1
    callback = callbacks[0]
    if fn is g.jacfwd:
        vectors = (t([1, 2]), t([3, 4])) if multi else t([1, 2])
        tangent = callback(inputs, vectors)
        eq(tangent, [13, 24] if multi else [4, 12])
        grad_helper = _closed(wrapped, 'grad_all' if multi else 'grad_single')
        grad_result = grad_helper(t([1, 2]), inputs)
        if multi:
            eq(grad_result[0], [4, 12])
            eq(grad_result[1], [3, 6])
        else:
            eq(grad_result, [4, 12])
    else:
        reverse = callback(inputs, t([1, 2]))
        if multi:
            eq(reverse[0], [4, 12])
            eq(reverse[1], [3, 6])
        else:
            eq(reverse, [4, 12])
    if aux:
        # aux_fn is closed over by the helper; its direct return is checked
        # independently of the compiled autodiff graph.
        auxiliary_helper = _closed(_closed(wrapped, 'grad_single'),
                                   'aux_fn') if fn is g.jacfwd else _closed(callback, 'aux_fn')
        eq(auxiliary_helper(*inputs), [16, 24] if multi else [4, 9])


def test_vjp_graph_aux_helper_direct_numeric():
    """
    Feature: Vjp graph aux helper direct numeric.
    Description: Check vjp graph aux helper direct numeric.
    Expectation: Graph VJP helpers preserve square primal values, auxiliary values, and the analytic cotangent
        product.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    value, backward, aux = g.vjp(auxiliary, t([2, 3]), has_aux=True)
    eq(value, [4, 9])
    eq(aux, [9, 10])
    eq(_closed(backward, 'aux_fn')(t([2, 3])), [4, 9])
    eq(backward(t([1, 2]))[0], [4, 12])


def test_jacobian_postprocess_mismatch_validation():
    # Three reverse-mode output blocks cannot be distributed over two inputs.
    """
    Feature: Jacobian postprocess mismatch validation.
    Description: Check jacobian postprocess mismatch validation.
    Expectation: Invalid jacobian postprocess mismatch arguments raise ValueError. Diagnostics identify numbers
        of inputs and outputs.
    """
    with pytest.raises(ValueError, match='numbers of inputs and outputs'):
        g._jac_postprocess(t(np.arange(3).reshape(3, 1)),
                           (((), 0), ((1,), 1), ((1,), 2), ((1,), 3)), (0, 1), 'reverse')


def test_linearize_list_vector_direct_wrapper(python_grad_wrappers):
    """
    Feature: Linearize list vector direct wrapper.
    Description: Check linearize list vector direct wrapper.
    Expectation: A list input/vector pair produces primal values [16, 24] and tangent values [13, 24].
    """
    value, linear = g.linearize(polynomial, [t([2, 3]), t([4, 5])])
    eq(value, [16, 24])
    eq(linear([t([1, 2]), t([3, 4])]), [13, 24])


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason='BatchToSpaceNDV2 has no CPU kernel in MS 2.12')
def test_batch_to_space_tensor_arguments_expected_numeric():
    """
    Feature: Batch to space tensor arguments expected numeric.
    Description: Check batch to space tensor arguments expected numeric.
    Expectation: Tensor-valued block/crop arguments produce the expected 2-by-2 spatial layout, with the
        unavailable CPU variant retaining its strict expected failure.
    """
    eq(a.batch_to_space_nd(t([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]), t([2, 2], ms.int32),
                           t([[0, 0], [0, 0]], ms.int32)), [[[[1, 2], [3, 4]]]])


def test_batch_to_space_tensor_native_protocol(monkeypatch):
    """
    Feature: Batch to space tensor native protocol.
    Description: Tensor-argument native variant unavailable on CPU: full call identity.
    Expectation: Tensor-argument native variant unavailable on CPU: full call identity.
    """
    x, blocks, crops = t([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]), t([2, 2], ms.int32), t([[0, 0], [0, 0]], ms.int32)
    sentinel, calls = object(), []
    def native(*args):
        calls.append(args)
        return sentinel
    monkeypatch.setattr(a, 'batch_to_space_nd_v2_', native)
    assert a.batch_to_space_nd(x, blocks, crops) is sentinel
    assert len(calls) == 1
    assert len(calls[0]) == 3
    assert all(actual is expected for actual, expected in zip(calls[0], (x, blocks, crops)))


def test_empty_slice_scatter_numeric():
    """
    Feature: Empty slice scatter numeric.
    Description: Check empty slice scatter numeric.
    Expectation: Scattering an empty slice leaves the original values [1, 2] intact.
    """
    eq(a.slice_scatter(t([1, 2]), t([]), start=0, end=0), [1, 2])


@pytest.mark.parametrize('fn', [g.jacfwd, g.jacrev])
def test_jacobian_aux_graph_helper_rejection_direct(python_grad_wrappers, fn):
    """
    Feature: Jacobian aux graph helper rejection direct.
    Description: Check jacobian aux graph helper rejection direct.
    Expectation: Invalid jacobian aux graph helper rejection direct raise ValueError. Diagnostics identify more
        than one outputs.
    """
    wrapped = fn(square, has_aux=True)
    helper = _closed(_closed(wrapped, 'grad_single'), 'aux_fn') if fn is g.jacfwd else _closed(wrapped, 'aux_fn')
    with pytest.raises(ValueError, match='more than one outputs'):
        helper(t([2]))


def test_seeded_random_stream_reproducibility_and_support():
    """
    Feature: Seeded random stream reproducibility and support.
    Description: Check seeded random stream reproducibility and support.
    Expectation: Resetting the seed reproduces the same int32 samples, with shape (4, 5) and values in [2, 8).
    """
    ms.set_seed(43)
    first = r.randint(2, 8, (4, 5), dtype=ms.int32, seed=17)
    ms.set_seed(43)
    second = r.randint(2, 8, (4, 5), dtype=ms.int32, seed=17)
    np.testing.assert_array_equal(first.asnumpy(), second.asnumpy())
    assert first.shape == (4, 5) and first.dtype == ms.int32
    assert np.all((first.asnumpy() >= 2) & (first.asnumpy() < 8))


def singleton_output(x):
    return (x * x,)


def conditional_auxiliary(x):
    if x.shape == (2,):
        return x * x, x + 7
    return x * x


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='Direct eager tuple-output graph helper loses tangent; compiled API is tested separately')
def test_jvp_singleton_tuple_python_numeric(python_grad_wrappers):
    """
    Feature: Jvp singleton tuple python numeric.
    Description: Check jvp singleton tuple python numeric.
    Expectation: Python JVP preserves singleton tuple structure and returns square values [4, 9] and tangents
        [4, 12].
    """
    output, tangent = g.jvp(singleton_output, t([2, 3]), t([1, 2]))
    assert isinstance(output, tuple) and len(output) == 1
    assert isinstance(tangent, tuple) and len(tangent) == 1
    eq(output[0], [4, 9])
    eq(tangent[0], [4, 12])


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='Direct eager tuple-output graph helper loses tangent; compiled API is tested separately')
def test_jacfwd_singleton_tuple_graph_helper_numeric(monkeypatch, python_grad_wrappers):
    """
    Feature: Jacfwd singleton tuple graph helper numeric.
    Description: Check jacfwd singleton tuple graph helper numeric.
    Expectation: The graph helper preserves a singleton tuple tangent and computes the analytic diagonal
        Jacobian and tangent values.
    """
    callbacks = []
    native_vmap = g._vmap
    def observe(callback):
        callbacks.append(callback)
        return native_vmap(callback)
    monkeypatch.setattr(g, '_vmap', observe)
    eq(g.jacfwd(singleton_output)(t([2, 3])), np.diag([4., 6.]))
    tangent = callbacks[0]((t([2, 3]),), t([1, 2]))
    assert isinstance(tangent, tuple) and len(tangent) == 1
    eq(tangent[0], [4, 12])


def test_vjp_aux_shape_dependent_rejection_graph_helper():
    """
    Feature: Vjp aux shape dependent rejection graph helper.
    Description: Check vjp aux shape dependent rejection graph helper.
    Expectation: Valid graph auxiliary outputs preserve the analytic VJP, while shape-dependent invalid
        auxiliary structure raises ValueError.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    value, backward, aux = g.vjp(conditional_auxiliary, t([2, 3]), has_aux=True)
    eq(value, [4, 9])
    eq(aux, [9, 10])
    eq(backward(t([1, 2]))[0], [4, 12])
    helper = _closed(backward, 'aux_fn')
    eq(helper(t([2, 3])), [4, 9])
    with pytest.raises(ValueError, match='more than one outputs'):
        helper(t([2]))


def test_check_tensor_container_validation():
    """
    Feature: Check tensor container validation.
    Description: Check check tensor container validation.
    Expectation: Invalid check tensor container arguments raise TypeError. Diagnostics identify inputs type must
        be Tensor.
    """
    with pytest.raises(TypeError, match='inputs type must be Tensor'):
        g._check_tensor([t([1])])


def test_jvp_singleton_tuple_compiled_numeric():
    """
    Feature: Jvp singleton tuple compiled numeric.
    Description: Check jvp singleton tuple compiled numeric.
    Expectation: Compiled JVP preserves singleton tuple structure with square values [4, 9] and tangents [4,
        12].
    """
    output, tangent = g.jvp(singleton_output, t([2, 3]), t([1, 2]))
    assert isinstance(output, tuple) and len(output) == 1
    assert isinstance(tangent, tuple) and len(tangent) == 1
    eq(output[0], [4, 9])
    eq(tangent[0], [4, 12])


@pytest.mark.parametrize('fn', [
    g.jacfwd,
    pytest.param(g.jacrev, marks=pytest.mark.xfail(
        strict=True, raises=TypeError,
        reason='jacrev singleton-tuple output fails AbstractTensor/AbstractTuple type join in MS 2.12')),
])
def test_jacobian_singleton_tuple_compiled_numeric(fn):
    """
    Feature: Jacobian singleton tuple compiled numeric.
    Description: Check jacobian singleton tuple compiled numeric.
    Expectation: Compiled Jacobians match diag(4, 6); the recorded reverse-mode singleton-tuple limitation
        retains its strict expected failure.
    """
    eq(fn(singleton_output)(t([2, 3])), np.diag([4., 6.]))
