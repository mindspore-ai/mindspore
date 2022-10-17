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

import mindspore
from mindspore.common.initializer import initializer, Identity, Dirac, Sparse, VarianceScaling, Orthogonal
from mindspore import context
import mindspore.ops as ops
import numpy as np
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_sparse(mode):
    """
    Feature: Test sparse initializer.
    Description: Initialize a 2 dimension sparse matrix to fill the input tensor.
    Expectation: The Tensor is initialized with a 2 dimension sparse matrix.
    """
    context.set_context(mode=mode)
    tensor1 = initializer(Sparse(sparsity=0.2, sigma=0.01), [5, 6], mindspore.float32)
    output = tensor1.init_data()
    assert np.array_equal(np.count_nonzero(output.asnumpy(), axis=0), [4, 4, 4, 4, 4, 4])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_orthogonal(mode):
    """
    Feature: Test orthogonal initializer.
    Description: Initialize a (semi) orthogonal matrix to fill the input tensor.
    Expectation: The Tensor is initialized with values from orthogonal matrix.
    """
    context.set_context(mode=mode)
    identity = np.identity(2)
    tensor1 = initializer(Orthogonal(gain=1.), [2, 2], mindspore.float32)
    t1 = tensor1.init_data()
    transpose = t1.transpose()
    mul = ops.MatMul()
    output = mul(t1, transpose)
    assert np.allclose(output.asnumpy(), identity, atol=1e-6, rtol=1e-7)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_variancescaling(mode):
    """
    Feature: Test varianceScaling initializer.
    Description: Randomly initialize an array with scaling to fill the input tensor.
    Expectation: The Tensor is initialized successfully.
    """
    context.set_context(mode=mode)
    mindspore.set_seed(0)
    tensor1 = initializer('varianceScaling', [2, 3], mindspore.float32)
    tensor2 = initializer(VarianceScaling(scale=1.0, mode='fan_out', distribution='untruncated_normal'), [2, 3],
                          mindspore.float32)
    tensor3 = initializer(VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'), [2, 3],
                          mindspore.float32)
    tensor4 = initializer(VarianceScaling(scale=3.0, mode='fan_avg', distribution='uniform'), [2, 3],
                          mindspore.float32)
    t1 = tensor1.init_data()
    expected_t1 = np.array([[0.49535394, -0.03666719, 0.23151064],
                            [-0.08424897, 0.39260703, -0.26104233]])
    t2 = tensor2.init_data()
    expected_t2 = np.array([[1.2710124e+00, 1.2299923e-03, -1.1589712e+00],
                            [1.1465757e+00, -2.2482322e-01, 9.2637345e-02]])
    t3 = tensor3.init_data()
    expected_t3 = np.array([[1.2023407, -0.9182362, 0.20436235],
                            [0.8581208, 1.0288558, 1.0927733]])
    t4 = tensor4.init_data()
    expected_t4 = np.array([[1.2470493, -1.0861205, -1.1339132],
                            [-0.07604776, -1.8196303, 0.5115674]])
    assert np.allclose(t1.asnumpy(), expected_t1)
    assert np.allclose(t2.asnumpy(), expected_t2)
    assert np.allclose(t3.asnumpy(), expected_t3)
    assert np.allclose(t4.asnumpy(), expected_t4)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_identity(mode):
    """
    Feature: Test identity initializer.
    Description: Initialize an identity matrix to fill a Tensor.
    Expectation: The Tensor is initialized with identity matrix.
    """
    context.set_context(mode=mode)
    tensor1 = initializer(Identity(), [3, 3], mindspore.float32)
    tensor2 = initializer('identity', [3, 4], mindspore.float32)
    tensor3 = initializer('identity', [4, 3], mindspore.float32)
    expect1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    expect2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    expect3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
    assert (tensor1.asnumpy() == expect1).all()
    assert (tensor2.asnumpy() == expect2).all()
    assert (tensor3.asnumpy() == expect3).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dirac(mode):
    """
    Feature: Test dirac initializer.
    Description: Initialize input tensor with the Dirac delta function.
    Expectation: The Tensor is correctly initialized.
    """
    context.set_context(mode=mode)
    tensor3_1 = initializer(Dirac(groups=1), [6, 2, 3], mindspore.float32)
    tensor3_2 = initializer(Dirac(groups=2), [6, 2, 3], mindspore.float32)
    tensor3_3 = initializer(Dirac(groups=3), [6, 2, 3], mindspore.float32)

    tensor4_1 = initializer(Dirac(groups=1), [6, 4, 3, 3], mindspore.float32)
    tensor4_2 = initializer(Dirac(groups=2), [6, 4, 3, 3], mindspore.float32)
    tensor4_3 = initializer(Dirac(groups=3), [6, 4, 3, 3], mindspore.float32)

    tensor5_1 = initializer(Dirac(groups=1), [6, 2, 3, 3, 3], mindspore.float32)
    tensor5_2 = initializer(Dirac(groups=2), [6, 2, 3, 3, 3], mindspore.float32)
    tensor5_3 = initializer(Dirac(groups=3), [6, 2, 3, 3, 3], mindspore.float32)

    expectation3_1 = np.array([[[0., 1., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 1., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]]], dtype=np.float32)

    expectation3_2 = np.array([[[0., 1., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 1., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]],
                               [[0., 1., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 1., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]]], dtype=np.float32)

    expectation3_3 = np.array([[[0., 1., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 1., 0.]],
                               [[0., 1., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 1., 0.]],
                               [[0., 1., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 1., 0.]]], dtype=np.float32)

    expectation4_1 = np.array([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]], dtype=np.float32)

    expectation4_2 = np.array([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]], dtype=np.float32)

    expectation4_3 = np.array([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]], dtype=np.float32)

    expectation5_1 = np.array([[[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]]], dtype=np.float32)

    expectation5_2 = np.array([[[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]]], dtype=np.float32)

    expectation5_3 = np.array([[[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]]], dtype=np.float32)

    assert (tensor3_1.asnumpy() == expectation3_1).all()
    assert (tensor3_2.asnumpy() == expectation3_2).all()
    assert (tensor3_3.asnumpy() == expectation3_3).all()

    assert (tensor4_1.asnumpy() == expectation4_1).all()
    assert (tensor4_2.asnumpy() == expectation4_2).all()
    assert (tensor4_3.asnumpy() == expectation4_3).all()

    assert (tensor5_1.asnumpy() == expectation5_1).all()
    assert (tensor5_2.asnumpy() == expectation5_2).all()
    assert (tensor5_3.asnumpy() == expectation5_3).all()
