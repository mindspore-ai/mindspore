# Copyright 2022 Huawei Technologies Co., Ltd
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

import numpy as np
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from tests.mindspore_test_framework.utils.check_gradient import (
    check_jacobian, Tensor, OperationGradChecker, check_gradient, NNGradChecker)
from tests.mark_utils import arg_mark


def test_operation_grad_checker():
    """
    Feature: Auto diff.
    Description: Check the result for GradOperation.
    Expectation: The result is expected.
    """
    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

    check_gradient(Net(), Tensor(np.array([[0.65, 0.8, 0.8]], np.float32)),
                   Tensor(np.array([[0.1], [0.2], [-.1]], np.float32)), grad_checker_class=OperationGradChecker,
                   input_selector=[1], sampling_times=2)


def test_grad_checker_primitive():
    """
    Feature: Auto diff.
    Description: Check the result for GradOperation.
    Expectation: The result is expected.
    """
    matmul = P.MatMul()

    def prim_f(x, y):
        return matmul(x, y)

    check_gradient(prim_f, Tensor(np.array([[0.65, 0.8, 0.8]], np.float32)),
                   Tensor(np.array([[0.1], [0.2], [-.1]], np.float32)),
                   grad_checker_class=OperationGradChecker, sampling_times=2)


def test_nn_jacobian_checker():
    """
    Feature: Auto diff.
    Description: Check the result for GradOperation.
    Expectation: The result is expected.
    """
    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.dense = nn.Dense(10, 10)

        def construct(self, x):
            out = self.dense(x)
            return out, x

    check_jacobian(Net(), Tensor(np.random.rand(1, 10).astype(np.float32)),
                   delta=1e-3,
                   max_error=1e-7,
                   grad_checker_class=NNGradChecker,
                   input_selector=[1],
                   output_selector=[0])


def test_nn_grad_checker():
    """
    Feature: Auto diff.
    Description: Check the result for GradOperation.
    Expectation: The result is expected.
    """
    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.dense = nn.Dense(10, 10)

        def construct(self, x):
            out = self.dense(x)
            return out

    check_gradient(Net(), Tensor(np.random.rand(1, 10).astype(np.float32)),
                   delta=1e-3,
                   max_error=1e-3,
                   grad_checker_class=NNGradChecker, sampling_times=3)


def test_operation_jacobian_checker():
    """
    Feature: Auto diff.
    Description: Check the result for GradOperation.
    Expectation: The result is expected.
    """
    class Net(nn.Cell):
        """ Net definition """

        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return x, out

    check_jacobian(Net(), Tensor(np.array([[0.65, 0.8, 0.8], [0.1, 0.2, 0.3]], np.float32)),
                   Tensor(np.array([[0.1, 0.3], [0.2, 0.2], [-.1, 0.4]], np.float32)),
                   grad_checker_class=OperationGradChecker, input_selector=[0],
                   output_selector=[0])


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_framstruct_all():
    """
    Feature: Auto diff.
    Description: Check the result for GradOperation.
    Expectation: The result is expected.
    """
    test_operation_grad_checker()
    test_grad_checker_primitive()
    test_nn_jacobian_checker()
    test_nn_grad_checker()
    test_operation_jacobian_checker()
