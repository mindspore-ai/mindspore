import numpy as onp


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
