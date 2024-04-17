import numpy as onp
from .utils import _count_unequal_element


class CompareBase:
    def __init__(self):
        pass

    # compare the data which is numpy array type
    def compare_nparray(self, data_expected, data_me, rtol, atol, equal_nan=True):
        if onp.any(onp.isnan(data_expected)):
            assert onp.allclose(data_expected, data_me, rtol,
                                atol, equal_nan=equal_nan)
        elif not onp.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
            _count_unequal_element(data_expected, data_me, rtol, atol)
        else:
            assert onp.array(data_expected).shape == onp.array(data_me).shape


comparebase = CompareBase()
