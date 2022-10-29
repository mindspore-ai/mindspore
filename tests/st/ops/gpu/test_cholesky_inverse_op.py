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

import torch
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.math_ops as P
from mindspore import Tensor, jit


class CholeskyInverseNet(nn.Cell):
    def __init__(self, upper):
        super(CholeskyInverseNet, self).__init__()
        self.choleskyinverse = P.CholeskyInverse(upper=upper)

    @jit
    def construct(self, x_ms):
        return self.choleskyinverse(x_ms)


def choleskyinverse(upper, loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x_np = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]).astype(np.float32)
    x_np_zd = x_np.dot(x_np.T) + 1e-5 * np.eye(4)
    x_torch = torch.tensor(x_np_zd)
    torch_u = torch.cholesky(x_torch, upper)
    ms_u = Tensor(torch_u.numpy())
    choleskyinverse_ = CholeskyInverseNet(upper)
    choleskyinverse_output = choleskyinverse_(ms_u)
    choleskyinverse_expect = torch.cholesky_inverse(torch_u, upper)
    assert np.allclose(choleskyinverse_output.asnumpy(), choleskyinverse_expect.numpy(), loss, loss)


def choleskyinverse_pynative(upper, loss):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x_np = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]).astype(np.float64)
    x_np_zd = x_np.dot(x_np.T) + 1e-5 * np.eye(4)
    x_torch = torch.tensor(x_np_zd)
    torch_u = torch.cholesky(x_torch, upper)
    ms_u = Tensor(torch_u.numpy())
    choleskyinverse_ = CholeskyInverseNet(upper)
    choleskyinverse_output = choleskyinverse_(ms_u)
    choleskyinverse_expect = torch.cholesky_inverse(torch_u, upper)
    assert np.allclose(choleskyinverse_output.asnumpy(), choleskyinverse_expect.numpy(), loss, loss)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_choleskyinverse_graph_float():
    """
    Feature: ALL To ALL
    Description: test cases for CholeskyInverse
    Expectation: the result match to numpy
    """
    choleskyinverse(True, loss=1.0e-4)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_choleskyinverse_pynative_double():
    """
    Feature: ALL To ALL
    Description: test cases for CholeskyInverse
    Expectation: the result match to numpy
    """
    choleskyinverse_pynative(False, loss=1.0e-5)
