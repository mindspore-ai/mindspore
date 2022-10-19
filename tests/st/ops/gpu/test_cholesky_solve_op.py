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
import pytest
import mindspore
from mindspore import nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops.operations.math_ops import CholeskySolve


class Net(nn.Cell):
    """a class used to test CholeskySolve gpu operator."""

    def __init__(self, upper=False):
        super(Net, self).__init__()
        self.cholesky_solve = CholeskySolve(upper=upper)

    def construct(self, x1, x2):
        """construct."""
        return self.cholesky_solve(x1, x2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cholesky_solve():
    """
    Feature: CholeskySolve gpu TEST.
    Description: test CholeskySolve operator
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    x1 = Tensor(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), mindspore.float32)
    x2 = Tensor(np.array([[2, 0, 0], [4, 1, 0], [-1, 1, 2]]), mindspore.float32)
    expect = np.array([[5.8125, -2.625, 0.625], [-2.625, 1.25, -0.25], [0.625, -0.25, 0.25]])
    net = Net()
    mindspore_output = net(x1, x2)
    diff = mindspore_output.asnumpy() - expect
    error = np.ones(shape=expect.shape)
    assert np.all(diff < error)
