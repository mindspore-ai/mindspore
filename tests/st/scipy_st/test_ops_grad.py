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
"""st for scipy.ops_grad."""
import pytest
import numpy as onp
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.scipy.ops import Eigh
from tests.st.scipy_st.utils import create_random_rank_matrix, gradient_check


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('compute_eigenvectors', [True, False])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('shape', [(8, 8)])
@pytest.mark.parametrize('data_type', [(onp.float32, 1e-3, 1e-3), (onp.float64, 1e-4, 1e-7)])
def test_eigh_grad(compute_eigenvectors, lower, shape, data_type):
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of eigh operator
    Expectation: the result match gradient checking.
    """
    onp.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE)
    dtype, epsilon, error = data_type

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.mean = ops.ReduceMean()
            self.sum = ops.ReduceSum()
            self.compute_eigenvectors = compute_eigenvectors
            self.lower = lower
            self.eigh = Eigh(compute_eigenvectors, lower)

        def construct(self, a):
            w, v = self.eigh(a)
            res = None
            if self.compute_eigenvectors:
                res = self.sum(w) + self.mean(v)
            else:
                res = self.mean(w)
            return res

    net = Net()
    a = create_random_rank_matrix(shape, dtype)
    assert gradient_check(Tensor(a), net, epsilon) < error
