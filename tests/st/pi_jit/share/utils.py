import numpy as onp
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops
from mindspore._c_expression import get_code_extra


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
        onp.testing.assert_almost_equal(
            actual, expected, decimal=error, err_msg=err_msg)
    else:
        onp.testing.assert_equal(actual, expected, err_msg=err_msg)


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = onp.abs(data_expected - data_me)
    greater = onp.greater(error, atol + onp.abs(data_me) * rtol)
    loss_count = onp.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
        format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if onp.any(onp.isnan(data_expected)) or onp.any(onp.isnan(data_me)):
        assert onp.allclose(data_expected, data_me, rtol,
                            atol, equal_nan=equal_nan)
    elif not onp.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert onp.array(data_expected).shape == onp.array(data_me).shape


def tensor_to_numpy(data):
    if isinstance(data, Tensor):
        return data.asnumpy()
    elif isinstance(data, tuple):
        if len(data) == 1:
            return tensor_to_numpy(data[0]),
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
        None: None
    }[type_]

def is_empty(variable):
    if variable is None:
        return True
    if isinstance(variable, str) and variable == "":
        return True
    if isinstance(variable, (list, tuple, dict, set)) and len(variable) == 0:
        return True
    return False

def assert_executed_by_graph_mode(func):
    jcr = get_code_extra(func)
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert len(jcr['code']['phase_']) > 0
