import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import PrimitiveWithInfer
from mindspore.ops import prim_attr_register
from mindspore._checkparam import Validator as validator
from mindspore import Tensor
import numpy as np
import scipy as scp

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Cholesky(PrimitiveWithInfer):
    """
    Inner API for Cholesky base class.
    """

    @prim_attr_register
    def __init__(self, lower=False, clean=True):
        super().__init__(name="Cholesky")
        self.lower = validator.check_value_type("lower", lower, [bool], self.lower)
        self.clean = validator.check_value_type("clean", clean, [bool], self.clean)
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def __infer__(self, x):
        x_shape = x['shape']
        x_dtype = x['dtype']
        return {
            'shape': tuple(x_shape),
            'dtype': x_dtype,
            'value': None
        }


class CholeskySolver(PrimitiveWithInfer):
    """
    Inner API for CholeskySolver class.
    """

    @prim_attr_register
    def __init__(self, lower=False):
        super().__init__(name="CholeskySolver")
        self.lower = validator.check_value_type("lower", lower, [bool], self.lower)
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def __infer__(self, x, b):
        b_shape = b['shape']
        x_dtype = x['dtype']
        return {
            'shape': tuple(b_shape),
            'dtype': x_dtype,
            'value': None
        }


class CholeskyNet(nn.Cell):
    def __init__(self, lower=False, clean=False):
        super(CholeskyNet, self).__init__()
        self.cholesky = Cholesky(lower, clean)

    def construct(self, x):
        return self.cholesky(x)


class CholeskySolverNet(nn.Cell):
    def __init__(self, lower=False):
        super(CholeskySolverNet, self).__init__()
        self.cholesky_solver = CholeskySolver(lower)

    def construct(self, c, b):
        return self.cholesky_solver(c, b)


def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """
    ompute the Cholesky decomposition of a matrix, to use in cho_solve.
    Returns a matrix containing the Cholesky decomposition
    """
    cholesky_net = CholeskyNet(lower=lower, clean=False)
    c = cholesky_net(a)
    return c, lower


def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    """
    Compute the Cholesky decomposition of a matrix.
    Returns the Cholesky decomposition
    """
    cholesky_net = CholeskyNet(lower=lower, clean=True)
    c = cholesky_net(a)
    return c


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """Solve the linear equations A x = b, given the Cholesky factorization of A.

    Parameters
    ----------
    c_and_lower: (c, lower) tuple, (array, bool)
        Cholesky factorization of a, as given by cho_factor
    b : array
        Right-hand side
    overwrite_b : bool, optional
        Whether to overwrite data in b (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : array
        The solution to the system A x = b

    See also
    --------
    cho_factor : Cholesky factorization of a matrix

    """
    (c, lower) = c_and_lower
    cholesky_solver_net = CholeskySolverNet(lower=lower)
    x = cholesky_solver_net(c, b)
    return x


def test_cholesky():
    """
    Feature: ALL TO ALL
    Description:  test cases for cholesky [N,N]
    Expectation: the result match scipy cholesky
    """
    a = np.array([[4, 12, -6], [12, 37, -43], [-16, -43, 98]], dtype=np.float32)
    tensor_a = Tensor(a)
    scp_c_1, _ = scp.linalg.cho_factor(a, lower=True)
    mscp_c_1, _ = cho_factor(tensor_a, lower=True)

    scp_c_2 = scp.linalg.cholesky(a, lower=True)
    mscp_c_2 = cholesky(tensor_a, lower=True)
    assert np.allclose(scp_c_1, mscp_c_1.asnumpy())
    assert np.allclose(scp_c_2, mscp_c_2.asnumpy())


def test_cholesky_solver():
    """
    Feature: ALL TO ALL
    Description:  test cases for cholesky  solver [N,N]
    Expectation: the result match scipy cholesky_solve
    """
    a = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]], dtype=np.float32)
    b = np.array([1, 1, 1, 1], dtype=np.float32)
    tensor_a = Tensor(a)
    tensor_b = Tensor(b)
    scp_c, lower = scp.linalg.cho_factor(a, lower=False)
    mscp_c, mscp_lower = cho_factor(tensor_a, lower=False)
    assert np.allclose(scp_c, mscp_c.asnumpy())

    scp_factor = (scp_c, lower)
    ms_cho_factor = (mscp_c, mscp_lower)
    scp_x = scp.linalg.cho_solve(scp_factor, b)
    mscp_x = cho_solve(ms_cho_factor, tensor_b)
    assert np.allclose(scp_x, mscp_x.asnumpy())
