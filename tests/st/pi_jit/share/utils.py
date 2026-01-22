import numpy as onp
import types
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops
from mindspore._c_expression import get_code_extra, clear_jit_compile_results
from mindspore.graph._utils.pijit_context import PIJitCaptureContext


def get_empty_tensor(dtype=mstype.float32):
    x = Tensor([1], dtype)
    output = ops.slice(x, (0,), (0,))
    return output


def match_array(actual, expected, error=0, err_msg=''):
    if isinstance(actual, (int, tuple, list, bool)):
        actual = onp.asarray(actual)

    if isinstance(actual, Tensor):
        actual = actual.asnumpy()

    if isinstance(expected, (int, tuple, list, bool)):
        expected = onp.asarray(expected)

    if isinstance(expected, Tensor):
        expected = expected.asnumpy()

    if error > 0:
        onp.testing.assert_almost_equal(actual, expected, decimal=error, err_msg=err_msg)
    else:
        onp.testing.assert_equal(actual, expected, err_msg=err_msg)


def match_value(actual, expected, error=0, err_msg=''):
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for idx in range(len(actual)):
            match_value(actual[idx], expected[idx], error, err_msg)
    elif isinstance(actual, dict) and isinstance(expected, dict):
        match_value(tuple(actual.keys()), tuple(expected.keys()), error, err_msg)
        match_value(tuple(actual.values()), tuple(expected.values()), error, err_msg)
    else:
        match_array(actual, expected, error, err_msg)


def assert_equal(expected, actual, decimal=7, err_msg=''):
    if isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual)
        assert len(expected) == len(actual)
        for l, r in zip(expected, actual):
            assert_equal(l, r, decimal=decimal, err_msg=err_msg)
    elif isinstance(expected, dict):
        assert type(expected) is type(actual)
        assert len(expected) == len(actual)
        for k in expected:
            assert k in actual
            assert_equal(expected[k], actual[k], decimal=decimal, err_msg=err_msg)
    elif isinstance(expected, Tensor):
        assert isinstance(actual, Tensor)
        match_array(actual, expected, error=decimal, err_msg=err_msg)
    else:
        assert expected == actual, f'expect: {expected}, actual: {actual}'


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = onp.abs(data_expected - data_me)
    greater = onp.greater(error, atol + onp.abs(data_me) * rtol)
    loss_count = onp.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater]
    )


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if onp.any(onp.isnan(data_expected)) or onp.any(onp.isnan(data_me)):
        assert onp.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not onp.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert onp.array(data_expected).shape == onp.array(data_me).shape


def tensor_to_numpy(data):
    if isinstance(data, Tensor):
        return data.asnumpy()
    elif isinstance(data, tuple):
        if len(data) == 1:
            return (tensor_to_numpy(data[0]),)
        else:
            return (tensor_to_numpy(data[0]), *tensor_to_numpy(data[1:]))
    else:
        assert False, 'unsupported data type'


def nptype_to_mstype(type_):
    """
    Convert MindSpore dtype to torch type.

    Args:
        type_ (:class:`mindspore.dtype`): MindSpore's dtype.

    Returns:
        The data type of torch.
    """

    return {
        onp.bool_: mstype.bool_,
        onp.int8: mstype.int8,
        onp.int16: mstype.int16,
        onp.int32: mstype.int32,
        onp.int64: mstype.int64,
        onp.uint8: mstype.uint8,
        onp.float16: mstype.float16,
        onp.float32: mstype.float32,
        onp.float64: mstype.float64,
        onp.complex64: mstype.complex64,
        onp.complex128: mstype.complex128,
        None: None,
    }[type_]


def is_empty(variable):
    if variable is None:
        return True
    if isinstance(variable, str) and variable == "":
        return True
    if isinstance(variable, (list, tuple, dict, set)) and len(variable) == 0:
        return True
    return False


def has_graph(jcr, *, depth=2):
    if not depth or jcr is None or 'code' not in jcr:
        return False
    if jcr['code'].get('phase_', None):
        return True
    if not jcr['code'].get('compiled_code_', None):
        return False
    for item in jcr['code']['compiled_code_'].co_consts:
        if isinstance(item, types.CodeType) and has_graph(get_code_extra(item), depth=depth - 1):
            return True
    return False


def assert_executed_by_graph_mode(func, *, call_count: int = None):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0, f'break_count expect: 0, actual: {jcr["break_count_"]}'
    assert has_graph(jcr)
    if call_count is not None:
        assert (
            jcr['code']['call_count_'] == call_count
        ), f'call_count expect: {call_count}, actual: {jcr["code"]["call_count_"]}'


def assert_no_graph_break(func, *, call_count: int = None):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0, f'break_count expect: 0, actual: {jcr["break_count_"]}'
    if call_count is not None:
        assert (
            jcr['code']['call_count_'] == call_count
        ), f'call_count expect: {call_count}, actual: {jcr["code"]["call_count_"]}'


def assert_has_graph_break(func, *, break_count: int = 1, call_count: int = None):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == break_count, f'break_count expect: {break_count}, actual: {jcr["break_count_"]}'
    if call_count is not None:
        assert (
            jcr['code']['call_count_'] == call_count
        ), f'call_count expect: {call_count}, actual: {jcr["code"]["call_count_"]}'


def assert_graph_compile_status(func, break_count=None, call_count=None, compile_count=None):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert (
        break_count is None or jcr['break_count_'] == break_count
    ), f'break_count expect: {break_count}, actual: {jcr["break_count_"]}'
    assert (
        call_count is None or jcr['code']['call_count_'] == call_count
    ), f'call_count expect: {call_count}, actual: {jcr["code"]["call_count_"]}'
    assert (
        compile_count is None or jcr['compile_count_'] == compile_count
    ), f'compile_count expect: {compile_count}, actual: {jcr["compile_count_"]}'
    assert has_graph(jcr)


def reset_func(func):
    inner_func = getattr(func, "__wrapped__", func)
    assert clear_jit_compile_results(inner_func)


def pi_jit_with_config(function=None, jit_config=None, *, fullgraph=False):
    wrap_func = PIJitCaptureContext(jit_config=jit_config, fullgraph=fullgraph)
    if function is not None:
        return wrap_func(function)
    return wrap_func


def allclose_nparray_recursive(data_expected, data_me, rtol, atol, equal_nan=True):
    if isinstance(data_me, onp.ndarray):
        allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif isinstance(data_me, tuple):
        allclose_nparray_recursive(data_expected[0], data_me[0], rtol, atol, equal_nan=equal_nan)
        if len(data_me) > 1:
            allclose_nparray_recursive(data_expected[1:], data_me[1:],
                                       rtol, atol, equal_nan=equal_nan)
    else:
        assert False, 'unsupported data type'
