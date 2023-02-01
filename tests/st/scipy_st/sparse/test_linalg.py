# Copyright 2021 Huawei Technologies Co., Ltd
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
"""st for scipy.sparse.linalg."""
import pytest
import numpy as onp
import scipy as osp
import scipy.sparse.linalg
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.scipy as msp
from mindspore import context
from mindspore.common import Tensor
from tests.st.scipy_st.utils import create_sym_pos_matrix, create_full_rank_matrix, to_tensor, to_ndarray, get_platform


def _fetch_preconditioner(preconditioner, a):
    """
    Returns one of various preconditioning matrices depending on the identifier
    `preconditioner' and the input matrix A whose inverse it supposedly
    approximates.
    """
    if preconditioner == 'identity':
        M = onp.eye(a.shape[0], dtype=a.dtype)
    elif preconditioner == 'random':
        random_matrix = create_sym_pos_matrix(a.shape, a.dtype)
        M = onp.linalg.inv(random_matrix)
    elif preconditioner == 'exact':
        M = onp.linalg.inv(a)
    else:
        M = None
    return M


def _is_valid_platform(tensor_type='Tensor'):
    if tensor_type == "CSRTensor" and get_platform() != "linux":
        return False
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('tensor_type, dtype, tol', [('Tensor', onp.float32, 1e-5), ('Tensor', onp.float64, 1e-12),
                                                     ('CSRTensor', onp.float32, 1e-5)])
@pytest.mark.parametrize('shape', [(7, 7)])
@pytest.mark.parametrize('preconditioner', [None, 'identity', 'exact', 'random'])
@pytest.mark.parametrize('maxiter', [3, None])
def test_cg_against_scipy(tensor_type, dtype, tol, shape, preconditioner, maxiter):
    """
    Feature: ALL TO ALL
    Description: test cases for cg using function way in pynative/graph mode
    Expectation: the result match scipy
    """
    if not _is_valid_platform(tensor_type):
        return
    onp.random.seed(0)
    a = create_sym_pos_matrix(shape, dtype)
    b = onp.random.random(shape[:1]).astype(dtype)
    m = _fetch_preconditioner(preconditioner, a)
    osp_res = scipy.sparse.linalg.cg(a, b, M=m, maxiter=maxiter, atol=tol, tol=tol)

    a = to_tensor((a, tensor_type))
    b = Tensor(b)
    m = to_tensor((m, tensor_type)) if m is not None else m

    # Using PYNATIVE MODE
    context.set_context(mode=context.PYNATIVE_MODE)
    msp_res_dyn = msp.sparse.linalg.cg(a, b, M=m, maxiter=maxiter, atol=tol, tol=tol)

    # Using GRAPH MODE
    context.set_context(mode=context.GRAPH_MODE)
    msp_res_sta = msp.sparse.linalg.cg(a, b, M=m, maxiter=maxiter, atol=tol, tol=tol)

    onp.testing.assert_allclose(osp_res[0], msp_res_dyn[0].asnumpy(), rtol=tol, atol=tol)
    onp.testing.assert_allclose(osp_res[0], msp_res_sta[0].asnumpy(), rtol=tol, atol=tol)
    assert osp_res[1] == msp_res_dyn[1].asnumpy().item()
    assert osp_res[1] == msp_res_sta[1].asnumpy().item()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('shape', [(2, 2)])
def test_cg_against_numpy(dtype, shape):
    """
    Feature: ALL TO ALL
    Description: test cases for cg
    Expectation: the result match numpy
    """
    onp.random.seed(0)
    a = create_sym_pos_matrix(shape, dtype)
    b = onp.random.random(shape[:1]).astype(dtype)
    expected = onp.linalg.solve(a, b)

    # Using PYNATIVE MODE
    context.set_context(mode=context.PYNATIVE_MODE)
    actual_dyn, _ = msp.sparse.linalg.cg(Tensor(a), Tensor(b))

    # Using GRAPH MODE
    context.set_context(mode=context.GRAPH_MODE)
    actual_sta, _ = msp.sparse.linalg.cg(Tensor(a), Tensor(b))

    tol = 1e-5
    onp.testing.assert_allclose(expected, actual_dyn.asnumpy(), rtol=tol, atol=tol)
    onp.testing.assert_allclose(expected, actual_sta.asnumpy(), rtol=tol, atol=tol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('tensor_type, dtype, tol', [('Tensor', onp.float32, 1e-5), ('Tensor', onp.float64, 1e-12),
                                                     ('CSRTensor', onp.float32, 1e-5)])
@pytest.mark.parametrize('shape', [(7, 7)])
@pytest.mark.parametrize('preconditioner', [None, 'identity', 'exact', 'random'])
@pytest.mark.parametrize('maxiter', [3, None])
def test_cg_against_scipy_graph(tensor_type, dtype, tol, shape, preconditioner, maxiter):
    """
    Feature: ALL TO ALL
    Description: test cases for cg within Cell object in pynative/graph mode
    Expectation: the result match scipy
    """
    if tensor_type == "CSRTensor" and get_platform() != "linux":
        return

    class Net(nn.Cell):
        def construct(self, a, b, m, maxiter, tol):
            return msp.sparse.linalg.cg(a, b, M=m, maxiter=maxiter, atol=tol, tol=tol)

    onp.random.seed(0)
    a = create_sym_pos_matrix(shape, dtype)
    b = onp.random.random(shape[:1]).astype(dtype)
    m = _fetch_preconditioner(preconditioner, a)
    osp_res = scipy.sparse.linalg.cg(a, b, M=m, maxiter=maxiter, atol=tol, tol=tol)

    a = to_tensor((a, tensor_type))
    b = Tensor(b)
    m = to_tensor((m, tensor_type)) if m is not None else m

    # Using PYNATIVE MODE
    context.set_context(mode=context.PYNATIVE_MODE)
    msp_res_dyn = Net()(a, b, m, maxiter, tol)

    # Using GRAPH MODE
    context.set_context(mode=context.GRAPH_MODE)
    msp_res_sta = Net()(a, b, m, maxiter, tol)

    onp.testing.assert_allclose(osp_res[0], msp_res_dyn[0].asnumpy(), rtol=tol, atol=tol)
    onp.testing.assert_allclose(osp_res[0], msp_res_sta[0].asnumpy(), rtol=tol, atol=tol)
    assert osp_res[1] == msp_res_dyn[1].asnumpy().item()
    assert osp_res[1] == msp_res_sta[1].asnumpy().item()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('flatten', [True, False])
@pytest.mark.parametrize('tensor_type, dtype, tol', [('Tensor', onp.float32, 1e-5), ('Tensor', onp.float64, 1e-8),
                                                     ('CSRTensor', onp.float32, 1e-5)])
@pytest.mark.parametrize('a, b, grad_a, grad_b', [
    ([[1.96822833, 0.82204467, 1.03749232, 0.88915326, 0.44986806, 1.11167143],
      [0.82204467, 2.25216591, 1.40235719, 0.70838919, 0.81377919, 1.06000368],
      [1.03749232, 1.40235719, 2.90618746, 0.7126087, 0.81029544, 1.28673025],
      [0.88915326, 0.70838919, 0.7126087, 2.17515263, 0.40443765, 1.02082996],
      [0.44986806, 0.81377919, 0.81029544, 0.40443765, 1.60570668, 0.62292701],
      [1.11167143, 1.06000368, 1.28673025, 1.02082996, 0.62292701, 2.30795277]],
     [0.79363745, 0.58000418, 0.1622986, 0.70075235, 0.96455108, 0.50000836],
     [[-0.07867674, -0.01521201, 0.06394698, -0.03854052, -0.13523701, 0.01326866],
      [-0.03508505, -0.00678363, 0.02851647, -0.01718673, -0.06030749, 0.00591702],
      [-0.00586019, -0.00113306, 0.00476305, -0.00287067, -0.01007304, 0.00098831],
      [-0.07704304, -0.01489613, 0.06261914, -0.03774023, -0.13242886, 0.01299314],
      [-0.14497008, -0.02802971, 0.11782896, -0.07101491, -0.24918826, 0.02444888],
      [-0.01868565, -0.00361284, 0.01518735, -0.00915334, -0.03211867, 0.00315129]],
     [0.22853142, 0.10191113, 0.01702201, 0.22378603, 0.42109291, 0.054276]),
    ([[1.85910724, 0.73233206, 0.65960803, 1.03821349, 0.55277616],
      [0.73233206, 1.69548841, 0.59992146, 1.01518264, 0.50824059],
      [0.65960803, 0.59992146, 1.98169091, 1.45565213, 0.47901749],
      [1.03821349, 1.01518264, 1.45565213, 3.3133049, 0.75598147],
      [0.55277616, 0.50824059, 0.47901749, 0.75598147, 1.46831254]],
     [0.59674531, 0.226012, 0.10694568, 0.22030621, 0.34982629],
     [[-0.07498642, 0.00167461, 0.01353184, 0.01008293, -0.03770084],
      [-0.09940184, 0.00221986, 0.01793778, 0.01336592, -0.04997616],
      [-0.09572781, 0.00213781, 0.01727477, 0.01287189, -0.04812897],
      [0.03135044, -0.00070012, -0.00565741, -0.00421549, 0.01576203],
      [-0.14053766, 0.00313851, 0.02536103, 0.01889718, -0.07065797]],
     [0.23398106, 0.31016481, 0.29870068, -0.09782316, 0.43852141]),
])
def test_cg_grad(flatten, tensor_type, dtype, tol, a, b, grad_a, grad_b):
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of cg in graph mode
    Expectation: the result match expectation
    """
    if tensor_type == "CSRTensor" and get_platform() != "linux":
        return
    context.set_context(mode=context.GRAPH_MODE)
    shape = (len(b),) if flatten else (len(b), 1)
    a = to_tensor((a, tensor_type), dtype)
    b = Tensor(onp.array(b, dtype=dtype).reshape(shape))
    expect_grad_a = onp.array(grad_a, dtype=dtype)
    expect_grad_b = onp.array(grad_b, dtype=dtype).reshape(shape)

    # Function
    grad_net = ops.GradOperation(get_all=True)(msp.sparse.linalg.cg)
    grad_a, grad_b = grad_net(a, b)[:2]
    onp.testing.assert_allclose(expect_grad_a, to_ndarray(grad_a), rtol=tol, atol=tol)
    onp.testing.assert_allclose(expect_grad_b, to_ndarray(grad_b), rtol=tol, atol=tol)

    # Cell
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sum = ops.ReduceSum()
            self.cg = msp.sparse.linalg.cg

        def construct(self, a, b):
            x, _ = self.cg(a, b)
            return self.sum(x)

    grad_net = ops.GradOperation(get_all=True)(Net())
    grad_a, grad_b = grad_net(a, b)[:2]
    onp.testing.assert_allclose(expect_grad_a, to_ndarray(grad_a), rtol=tol, atol=tol)
    onp.testing.assert_allclose(expect_grad_b, to_ndarray(grad_b), rtol=tol, atol=tol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('tensor_type, dtype, tol', [('Tensor', onp.float32, 1e-5), ('Tensor', onp.float64, 1e-8)])
@pytest.mark.parametrize('a, b, grad_a, grad_b', [
    ([[1.96822833, 0.82204467, 1.03749232, 0.88915326, 0.44986806, 1.11167143],
      [0.82204467, 2.25216591, 1.40235719, 0.70838919, 0.81377919, 1.06000368],
      [1.03749232, 1.40235719, 2.90618746, 0.7126087, 0.81029544, 1.28673025],
      [0.88915326, 0.70838919, 0.7126087, 2.17515263, 0.40443765, 1.02082996],
      [0.44986806, 0.81377919, 0.81029544, 0.40443765, 1.60570668, 0.62292701],
      [1.11167143, 1.06000368, 1.28673025, 1.02082996, 0.62292701, 2.30795277]],
     [0.79363745, 0.58000418, 0.1622986, 0.70075235, 0.96455108, 0.50000836],
     [[-0.07867674, -0.01521201, 0.06394698, -0.03854052, -0.13523701, 0.01326866],
      [-0.03508505, -0.00678363, 0.02851647, -0.01718673, -0.06030749, 0.00591702],
      [-0.00586019, -0.00113306, 0.00476305, -0.00287067, -0.01007304, 0.00098831],
      [-0.07704304, -0.01489613, 0.06261914, -0.03774023, -0.13242886, 0.01299314],
      [-0.14497008, -0.02802971, 0.11782896, -0.07101491, -0.24918826, 0.02444888],
      [-0.01868565, -0.00361284, 0.01518735, -0.00915334, -0.03211867, 0.00315129]],
     [0.22853142, 0.10191113, 0.01702201, 0.22378603, 0.42109291, 0.054276]),
    ([[1.85910724, 0.73233206, 0.65960803, 1.03821349, 0.55277616],
      [0.73233206, 1.69548841, 0.59992146, 1.01518264, 0.50824059],
      [0.65960803, 0.59992146, 1.98169091, 1.45565213, 0.47901749],
      [1.03821349, 1.01518264, 1.45565213, 3.3133049, 0.75598147],
      [0.55277616, 0.50824059, 0.47901749, 0.75598147, 1.46831254]],
     [0.59674531, 0.226012, 0.10694568, 0.22030621, 0.34982629],
     [[-0.07498642, 0.00167461, 0.01353184, 0.01008293, -0.03770084],
      [-0.09940184, 0.00221986, 0.01793778, 0.01336592, -0.04997616],
      [-0.09572781, 0.00213781, 0.01727477, 0.01287189, -0.04812897],
      [0.03135044, -0.00070012, -0.00565741, -0.00421549, 0.01576203],
      [-0.14053766, 0.00313851, 0.02536103, 0.01889718, -0.07065797]],
     [0.23398106, 0.31016481, 0.29870068, -0.09782316, 0.43852141]),
])
def test_cg_grad_pynative_tensor(tensor_type, dtype, tol, a, b, grad_a, grad_b):
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of cg in pynative mode
    Expectation: the result match expectation
    """
    if tensor_type == "CSRTensor" and get_platform() != "linux":
        return
    context.set_context(mode=context.PYNATIVE_MODE)

    a = to_tensor((a, tensor_type), dtype)
    b = Tensor(onp.array(b, dtype=dtype))
    expect_grad_a = onp.array(grad_a, dtype=dtype)
    expect_grad_b = onp.array(grad_b, dtype=dtype)

    # Function
    grad_net = ops.GradOperation(get_all=True)(msp.sparse.linalg.cg)
    grad_a, grad_b = grad_net(a, b)[:2]
    onp.testing.assert_allclose(expect_grad_a, to_ndarray(grad_a), rtol=tol, atol=tol)
    onp.testing.assert_allclose(expect_grad_b, to_ndarray(grad_b), rtol=tol, atol=tol)

    # Cell
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sum = ops.ReduceSum()
            self.cg = msp.sparse.linalg.cg

        def construct(self, a, b):
            x, _ = self.cg(a, b)
            return self.sum(x)

    grad_net = ops.GradOperation(get_all=True)(Net())
    grad_a, grad_b = grad_net(a, b)[:2]
    onp.testing.assert_allclose(expect_grad_a, to_ndarray(grad_a), rtol=tol, atol=tol)
    onp.testing.assert_allclose(expect_grad_b, to_ndarray(grad_b), rtol=tol, atol=tol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('tensor_type, dtype, tol', [('CSRTensor', onp.float32, 1e-5)])
@pytest.mark.parametrize('a, b, grad_a, grad_b', [
    ([[1.85910724, 0.73233206, 0.65960803, 1.03821349, 0.55277616],
      [0.73233206, 1.69548841, 0.59992146, 1.01518264, 0.50824059],
      [0.65960803, 0.59992146, 1.98169091, 1.45565213, 0.47901749],
      [1.03821349, 1.01518264, 1.45565213, 3.3133049, 0.75598147],
      [0.55277616, 0.50824059, 0.47901749, 0.75598147, 1.46831254]],
     [0.59674531, 0.226012, 0.10694568, 0.22030621, 0.34982629],
     [[-0.07498642, 0.00167461, 0.01353184, 0.01008293, -0.03770084],
      [-0.09940184, 0.00221986, 0.01793778, 0.01336592, -0.04997616],
      [-0.09572781, 0.00213781, 0.01727477, 0.01287189, -0.04812897],
      [0.03135044, -0.00070012, -0.00565741, -0.00421549, 0.01576203],
      [-0.14053766, 0.00313851, 0.02536103, 0.01889718, -0.07065797]],
     [0.23398106, 0.31016481, 0.29870068, -0.09782316, 0.43852141]),
])
def test_cg_grad_pynative_csrtensor_data1(tensor_type, dtype, tol, a, b, grad_a, grad_b):
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of cg in pynative mode
    Expectation: the result match expectation
    """
    if tensor_type == "CSRTensor" and get_platform() != "linux":
        return
    context.set_context(mode=context.PYNATIVE_MODE)

    a = to_tensor((a, tensor_type), dtype)
    b = Tensor(onp.array(b, dtype=dtype))
    expect_grad_a = onp.array(grad_a, dtype=dtype)
    expect_grad_b = onp.array(grad_b, dtype=dtype)

    # Function
    grad_net = ops.GradOperation(get_all=True)(msp.sparse.linalg.cg)
    grad_a, grad_b = grad_net(a, b)[:2]
    onp.testing.assert_allclose(expect_grad_a, to_ndarray(grad_a), rtol=tol, atol=tol)
    onp.testing.assert_allclose(expect_grad_b, to_ndarray(grad_b), rtol=tol, atol=tol)

    # Cell
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sum = ops.ReduceSum()
            self.cg = msp.sparse.linalg.cg

        def construct(self, a, b):
            x, _ = self.cg(a, b)
            return self.sum(x)

    grad_net = ops.GradOperation(get_all=True)(Net())
    grad_a, grad_b = grad_net(a, b)[:2]
    onp.testing.assert_allclose(expect_grad_a, to_ndarray(grad_a), rtol=tol, atol=tol)
    onp.testing.assert_allclose(expect_grad_b, to_ndarray(grad_b), rtol=tol, atol=tol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('tensor_type, dtype, tol', [('CSRTensor', onp.float32, 1e-5)])
@pytest.mark.parametrize('a, b, grad_a, grad_b', [
    ([[1.96822833, 0.82204467, 1.03749232, 0.88915326, 0.44986806, 1.11167143],
      [0.82204467, 2.25216591, 1.40235719, 0.70838919, 0.81377919, 1.06000368],
      [1.03749232, 1.40235719, 2.90618746, 0.7126087, 0.81029544, 1.28673025],
      [0.88915326, 0.70838919, 0.7126087, 2.17515263, 0.40443765, 1.02082996],
      [0.44986806, 0.81377919, 0.81029544, 0.40443765, 1.60570668, 0.62292701],
      [1.11167143, 1.06000368, 1.28673025, 1.02082996, 0.62292701, 2.30795277]],
     [0.79363745, 0.58000418, 0.1622986, 0.70075235, 0.96455108, 0.50000836],
     [[-0.07867674, -0.01521201, 0.06394698, -0.03854052, -0.13523701, 0.01326866],
      [-0.03508505, -0.00678363, 0.02851647, -0.01718673, -0.06030749, 0.00591702],
      [-0.00586019, -0.00113306, 0.00476305, -0.00287067, -0.01007304, 0.00098831],
      [-0.07704304, -0.01489613, 0.06261914, -0.03774023, -0.13242886, 0.01299314],
      [-0.14497008, -0.02802971, 0.11782896, -0.07101491, -0.24918826, 0.02444888],
      [-0.01868565, -0.00361284, 0.01518735, -0.00915334, -0.03211867, 0.00315129]],
     [0.22853142, 0.10191113, 0.01702201, 0.22378603, 0.42109291, 0.054276]),
])
def test_cg_grad_pynative_csrtensor_data2(tensor_type, dtype, tol, a, b, grad_a, grad_b):
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of cg in pynative mode
    Expectation: the result match expectation
    """
    if tensor_type == "CSRTensor" and get_platform() != "linux":
        return
    context.set_context(mode=context.PYNATIVE_MODE)

    a = to_tensor((a, tensor_type), dtype)
    b = Tensor(onp.array(b, dtype=dtype))
    expect_grad_a = onp.array(grad_a, dtype=dtype)
    expect_grad_b = onp.array(grad_b, dtype=dtype)

    # Function
    grad_net = ops.GradOperation(get_all=True)(msp.sparse.linalg.cg)
    grad_a, grad_b = grad_net(a, b)[:2]
    onp.testing.assert_allclose(expect_grad_a, to_ndarray(grad_a), rtol=tol, atol=tol)
    onp.testing.assert_allclose(expect_grad_b, to_ndarray(grad_b), rtol=tol, atol=tol)

    # Cell
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sum = ops.ReduceSum()
            self.cg = msp.sparse.linalg.cg

        def construct(self, a, b):
            x, _ = self.cg(a, b)
            return self.sum(x)

    grad_net = ops.GradOperation(get_all=True)(Net())
    grad_a, grad_b = grad_net(a, b)[:2]
    onp.testing.assert_allclose(expect_grad_a, to_ndarray(grad_a), rtol=tol, atol=tol)
    onp.testing.assert_allclose(expect_grad_b, to_ndarray(grad_b), rtol=tol, atol=tol)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [64])
@pytest.mark.parametrize('dtype,error', [(onp.float32, 5e-3)])
@pytest.mark.parametrize('preconditioner', ['random'])
@pytest.mark.parametrize('solve_method', ['incremental', 'batched'])
def test_gmres_against_scipy_level1(n, dtype, error, preconditioner, solve_method):
    """
    Feature: ALL TO ALL
    Description: level1 test cases for [N x N] X [N X 1]
    Expectation: the result match scipy
    """
    onp.random.seed(0)
    a = create_full_rank_matrix((n, n), dtype)
    b = onp.random.rand(n).astype(dtype)
    x0 = onp.zeros_like(b).astype(dtype)
    M = _fetch_preconditioner(preconditioner, a)
    tol = float(onp.finfo(dtype=dtype).eps)
    atol = tol
    restart = n
    maxiter = None
    scipy_output, _ = osp.sparse.linalg.gmres(a, b, x0, tol=tol, restart=restart, maxiter=maxiter, M=M, atol=atol)
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE)
    M = Tensor(M) if M is not None else M
    ms_output, _ = msp.sparse.linalg.gmres(Tensor(a), Tensor(b), Tensor(x0), tol=tol, restart=restart, maxiter=maxiter,
                                           M=M, atol=atol, solve_method=solve_method)
    assert onp.allclose(scipy_output, ms_output.asnumpy(), rtol=error, atol=error)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [3, 7])
@pytest.mark.parametrize('tensor_type, dtype, error', [('Tensor', onp.float64, 1e-5), ('Tensor', onp.float32, 1e-4),
                                                       ('CSRTensor', onp.float32, 1e-4)])
@pytest.mark.parametrize('restart', [1, 2])
@pytest.mark.parametrize('maxiter', [1, 2])
@pytest.mark.parametrize('preconditioner', ['identity', 'exact', 'random'])
@pytest.mark.parametrize('solve_method', ['incremental', 'batched'])
def test_gmres_against_scipy(n, tensor_type, dtype, error, restart, maxiter, preconditioner, solve_method):
    """
    Feature: ALL TO ALL
    Description:  test cases for [N x N] X [N X 1]
    Expectation: the result match scipy
    """
    if not _is_valid_platform(tensor_type):
        return
    onp.random.seed(0)
    a = create_full_rank_matrix((n, n), dtype)
    b = onp.random.rand(n).astype(dtype)
    x0 = onp.zeros_like(b).astype(dtype)
    m = _fetch_preconditioner(preconditioner, a)
    tol = float(onp.finfo(dtype=dtype).eps)
    atol = tol
    if preconditioner == 'random':
        restart = n
        maxiter = None
    scipy_output, _ = osp.sparse.linalg.gmres(a, b, x0, tol=tol, restart=restart, maxiter=maxiter, M=m, atol=atol)
    # PyNative Mode
    context.set_context(mode=context.PYNATIVE_MODE)
    a = to_tensor((a, tensor_type))
    b = Tensor(b)
    x0 = Tensor(x0)
    m = to_tensor((m, tensor_type)) if m is not None else m
    ms_output, _ = msp.sparse.linalg.gmres(a, b, x0, tol=tol, restart=restart,
                                           maxiter=maxiter, M=m, atol=atol, solve_method=solve_method)
    assert onp.allclose(scipy_output, ms_output.asnumpy(), rtol=error, atol=error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [3])
@pytest.mark.parametrize('tensor_type, dtype, error', [('Tensor', onp.float64, 1e-5), ('Tensor', onp.float32, 1e-4),
                                                       ('CSRTensor', onp.float32, 1e-4)])
@pytest.mark.parametrize('preconditioner', ['random'])
@pytest.mark.parametrize('solve_method', ['incremental', 'batched'])
def test_gmres_against_graph_scipy(n, tensor_type, dtype, error, preconditioner, solve_method):
    """
    Feature: ALL TO ALL
    Description:  test cases for [N x N] X [N X 1]
    Expectation: the result match scipy in graph
    """
    if not _is_valid_platform(tensor_type):
        return

    # Input CSRTensor of gmres in mindspore graph mode is not supported, just ignored it.
    if tensor_type == "CSRTensor":
        return

    class TestNet(nn.Cell):
        def __init__(self, solve_method):
            super(TestNet, self).__init__()
            self.solve_method = solve_method

        def construct(self, a, b, x0, tol, restart, maxiter, m, atol):
            return msp.sparse.linalg.gmres(a, b, x0, tol=tol, restart=restart, maxiter=maxiter, M=m,
                                           atol=atol, solve_method=self.solve_method)

    onp.random.seed(0)
    a = create_full_rank_matrix((n, n), dtype)
    b = onp.random.rand(n).astype(dtype)
    x0 = onp.zeros_like(b).astype(dtype)
    m = _fetch_preconditioner(preconditioner, a)
    tol = float(onp.finfo(dtype=dtype).eps)
    atol = tol
    restart = n
    maxiter = None
    scipy_output, _ = osp.sparse.linalg.gmres(a, b, x0, tol=tol, restart=restart, maxiter=maxiter, M=m, atol=atol)
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE)
    a = to_tensor((a, tensor_type))
    b = Tensor(b)
    x0 = Tensor(x0)
    m = to_tensor((m, tensor_type)) if m is not None else m
    # Not in graph's construct
    ms_output, _ = msp.sparse.linalg.gmres(a, b, x0, tol=tol, restart=restart, maxiter=maxiter,
                                           M=m, atol=atol)
    assert onp.allclose(scipy_output, ms_output.asnumpy(), rtol=error, atol=error)
    # With in graph's construct
    ms_net_output, _ = TestNet(solve_method)(a, b, x0, tol, restart, maxiter, m, atol)
    assert onp.allclose(scipy_output, ms_net_output.asnumpy(), rtol=error, atol=error)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('tensor_type, dtype, error', [('Tensor', onp.float64, 1e-5), ('Tensor', onp.float32, 1e-4),
                                                       ('CSRTensor', onp.float32, 1e-4)])
@pytest.mark.parametrize('preconditioner', ['identity', 'exact', 'random'])
@pytest.mark.parametrize('solve_method', ['incremental', 'batched'])
@pytest.mark.parametrize('a, b, grad_a, grad_b', [
    ([[1.96822833, 0.82204467, 1.03749232, 0.88915326, 0.44986806, 1.11167143],
      [0.82204467, 2.25216591, 1.40235719, 0.70838919, 0.81377919, 1.06000368],
      [1.03749232, 1.40235719, 2.90618746, 0.7126087, 0.81029544, 1.28673025],
      [0.88915326, 0.70838919, 0.7126087, 2.17515263, 0.40443765, 1.02082996],
      [0.44986806, 0.81377919, 0.81029544, 0.40443765, 1.60570668, 0.62292701],
      [1.11167143, 1.06000368, 1.28673025, 1.02082996, 0.62292701, 2.30795277]],
     [0.79363745, 0.58000418, 0.1622986, 0.70075235, 0.96455108, 0.50000836],
     [[-0.07867674, -0.01521201, 0.06394698, -0.03854052, -0.13523701, 0.01326866],
      [-0.03508505, -0.00678363, 0.02851647, -0.01718673, -0.06030749, 0.00591702],
      [-0.00586019, -0.00113306, 0.00476305, -0.00287067, -0.01007304, 0.00098831],
      [-0.07704304, -0.01489613, 0.06261914, -0.03774023, -0.13242886, 0.01299314],
      [-0.14497008, -0.02802971, 0.11782896, -0.07101491, -0.24918826, 0.02444888],
      [-0.01868565, -0.00361284, 0.01518735, -0.00915334, -0.03211867, 0.00315129]],
     [0.22853142, 0.10191113, 0.01702201, 0.22378603, 0.42109291, 0.054276]),
    ([[1.85910724, 0.73233206, 0.65960803, 1.03821349, 0.55277616],
      [0.73233206, 1.69548841, 0.59992146, 1.01518264, 0.50824059],
      [0.65960803, 0.59992146, 1.98169091, 1.45565213, 0.47901749],
      [1.03821349, 1.01518264, 1.45565213, 3.3133049, 0.75598147],
      [0.55277616, 0.50824059, 0.47901749, 0.75598147, 1.46831254]],
     [0.59674531, 0.226012, 0.10694568, 0.22030621, 0.34982629],
     [[-0.07498642, 0.00167461, 0.01353184, 0.01008293, -0.03770084],
      [-0.09940184, 0.00221986, 0.01793778, 0.01336592, -0.04997616],
      [-0.09572781, 0.00213781, 0.01727477, 0.01287189, -0.04812897],
      [0.03135044, -0.00070012, -0.00565741, -0.00421549, 0.01576203],
      [-0.14053766, 0.00313851, 0.02536103, 0.01889718, -0.07065797]],
     [0.23398106, 0.31016481, 0.29870068, -0.09782316, 0.43852141]),
])
def test_gmres_grad(tensor_type, dtype, error, preconditioner, solve_method, a, b, grad_a, grad_b):
    """
    Feature: ALL TO ALL
    Description:  test cases for gmres grad [N x N] X [N X 1]
    Expectation: the result match jax grad
    """
    if not _is_valid_platform(tensor_type):
        return

    # Input CSRTensor of gmres grad in mindspore graph or pynative mode is not supported, just ignored it.
    # Root cause: CSRTensor has no distribute function of T.
    if tensor_type == "CSRTensor":
        return

    # Gmres grad in construct
    class GmresGradNet(nn.Cell):
        def __init__(self, solve_method):
            super(GmresGradNet, self).__init__()
            self.sum = ops.ReduceSum()
            self.gmres = msp.sparse.linalg.gmres
            self.solve_method = solve_method

        def construct(self, a, b, x0, tol, m, atol):
            # For restart && maxiter args, we maintain default values to ensure gmres can coverage.
            x, _ = self.gmres(a, b, x0, tol=tol, M=m, atol=atol,
                              solve_method=self.solve_method)
            return self.sum(x)


    tol = float(onp.finfo(dtype=dtype).eps)
    atol = tol
    a = onp.array(a, dtype=dtype)
    b = onp.array(b, dtype=dtype)
    x0 = onp.zeros_like(b).astype(dtype)
    m = _fetch_preconditioner(preconditioner, a)
    expect_grad_a = grad_a
    expect_grad_b = grad_b
    a = to_tensor((a, tensor_type))
    b = Tensor(b)
    x0 = Tensor(x0)
    m = to_tensor((m, tensor_type)) if m is not None else m

    # PyNative Mode
    context.set_context(mode=context.PYNATIVE_MODE)
    gmres_grad_net = ops.GradOperation(get_all=True)(GmresGradNet(solve_method))
    grad_a, grad_b = gmres_grad_net(a, b, x0, tol, m, atol)[:2]
    assert onp.allclose(expect_grad_a, to_ndarray(grad_a), rtol=error, atol=error)
    assert onp.allclose(expect_grad_b, to_ndarray(grad_b), rtol=error, atol=error)

    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE)
    gmres_grad_net = ops.GradOperation(get_all=True)(GmresGradNet(solve_method))
    grad_a, grad_b = gmres_grad_net(a, b, x0, tol, m, atol)[:2]
    assert onp.allclose(expect_grad_a, to_ndarray(grad_a), rtol=error, atol=error)
    assert onp.allclose(expect_grad_b, to_ndarray(grad_b), rtol=error, atol=error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype_tol', [(onp.float64, 1e-10)])
@pytest.mark.parametrize('shape', [(4, 4), (7, 7)])
@pytest.mark.parametrize('preconditioner', [None, 'identity', 'exact', 'random'])
@pytest.mark.parametrize('maxiter', [1, 3])
def test_bicgstab_against_scipy(dtype_tol, shape, preconditioner, maxiter):
    """
    Feature: ALL TO ALL
    Description: test cases for bicgstab
    Expectation: the result match scipy
    """
    onp.random.seed(0)
    dtype, tol = dtype_tol
    A = create_full_rank_matrix(shape, dtype)
    b = onp.random.random(shape[:1]).astype(dtype)
    M = _fetch_preconditioner(preconditioner, A)
    osp_res = scipy.sparse.linalg.bicgstab(A, b, M=M, maxiter=maxiter, atol=tol, tol=tol)[0]

    A = Tensor(A)
    b = Tensor(b)
    M = Tensor(M) if M is not None else M

    # using PYNATIVE MODE
    context.set_context(mode=context.PYNATIVE_MODE)
    msp_res_dyn = msp.sparse.linalg.bicgstab(A, b, M=M, maxiter=maxiter, atol=tol, tol=tol)[0]

    # using GRAPH MODE
    context.set_context(mode=context.GRAPH_MODE)
    msp_res_sta = msp.sparse.linalg.bicgstab(A, b, M=M, maxiter=maxiter, atol=tol, tol=tol)[0]

    kw = {"atol": tol, "rtol": tol}
    onp.testing.assert_allclose(osp_res, msp_res_dyn.asnumpy(), **kw)
    onp.testing.assert_allclose(osp_res, msp_res_sta.asnumpy(), **kw)
