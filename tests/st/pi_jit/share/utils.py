import numpy as onp
from mindspore import Tensor


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
