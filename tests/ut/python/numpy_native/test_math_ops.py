
import pytest
import numpy as onp

import mindspore.context as context
import mindspore.numpy as mnp


def rand_int(*shape):
    """return an random integer array with parameter shape"""
    res = onp.random.randint(low=1, high=5, size=shape)
    if isinstance(res, onp.ndarray):
        res = res.astype(onp.float32)
    return res


class Cases():
    def __init__(self):
        self.device_cpu = context.get_context('device_target') == 'CPU'

        self.arrs = [
            rand_int(2),
            rand_int(2, 3),
            rand_int(2, 3, 4),
            rand_int(2, 3, 4, 5),
        ]

        # scalars expanded across the 0th dimension
        self.scalars = [
            rand_int(),
            rand_int(1),
            rand_int(1, 1),
            rand_int(1, 1, 1),
        ]

        # arrays with last dimension aligned
        self.aligned_arrs = [
            rand_int(2, 3),
            rand_int(1, 4, 3),
            rand_int(5, 1, 2, 3),
            rand_int(4, 2, 1, 1, 3),
        ]


test_case = Cases()


def mnp_inner(a, b):
    return mnp.inner(a, b)


def onp_inner(a, b):
    return onp.inner(a, b)


def test_inner():
    for arr1 in test_case.aligned_arrs:
        for arr2 in test_case.aligned_arrs:
            match_res(mnp_inner, onp_inner, arr1, arr2)

    for scalar1 in test_case.scalars:
        for scalar2 in test_case.scalars:
            match_res(mnp_inner, onp_inner,
                      scalar1, scalar2)


# check if the output from mnp function and onp function applied on the arrays are matched


def match_res(mnp_fn, onp_fn, arr1, arr2):
    actual = mnp_fn(mnp.asarray(arr1, dtype='float32'),
                    mnp.asarray(arr2, dtype='float32')).asnumpy()
    expected = onp_fn(arr1, arr2)
    match_array(actual, expected)


def match_array(actual, expected, error=5):
    if error > 0:
        onp.testing.assert_almost_equal(actual.tolist(), expected.tolist(),
                                        decimal=error)
    else:
        onp.testing.assert_equal(actual.tolist(), expected.tolist())


def test_exception_innner():
    with pytest.raises(ValueError):
        mnp.inner(mnp.asarray(test_case.arrs[0]),
                  mnp.asarray(test_case.arrs[1]))
