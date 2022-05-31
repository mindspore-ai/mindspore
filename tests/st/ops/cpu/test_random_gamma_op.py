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

import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore import Tensor


class RandomGammaTEST(nn.Cell):
    def __init__(self, seed=0):
        super(RandomGammaTEST, self).__init__()
        self.seed = seed

    def construct(self, shape, alpha, beta):
        return C.gamma(shape, alpha, beta, self.seed)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_gamma_op(dtype):
    """
    Feature: Gamma cpu kernel
    Description: test the gamma beta is a tensor.
    Expectation: match to tensorflow benchmark.
    """

    shape = (3, 1, 2)
    alpha = Tensor(np.array([[3, 4], [5, 6]]), ms.float32)
    beta = Tensor(np.array([3.0, 2.0]), ms.float32)
    gamma_test = RandomGammaTEST(seed=3)
    expect = np.array([3, 1, 2, 2, 2])

    ms.set_context(mode=ms.GRAPH_MODE, device_target='CPU')
    output = gamma_test(shape, alpha, beta)
    assert (output.shape == expect).all()

    ms.set_context(mode=ms.PYNATIVE_MODE)
    output = ms.ops.gamma(shape, alpha, beta)
    assert (output.shape == expect).all()

    ms.set_context(mode=ms.GRAPH_MODE, device_target='CPU')
    output = gamma_test(shape, alpha, None)
    assert (output.shape == expect).all()

    ms.set_context(mode=ms.PYNATIVE_MODE)
    output = ms.ops.gamma(shape, alpha, None)
    assert (output.shape == expect).all()
