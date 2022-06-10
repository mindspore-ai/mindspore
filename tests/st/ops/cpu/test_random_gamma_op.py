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
from mindspore.ops import operations as P
from mindspore import Tensor


class RandomGammaTEST(nn.Cell):
    def __init__(self, shape, seed=0):
        super(RandomGammaTEST, self).__init__()
        self.shape = shape
        self.random_gamma = P.RandomGamma(seed)

    def construct(self, alpha):
        return self.random_gamma(self.shape, alpha)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_random_gamma_op(dtype):
    """
    Feature: RandomGamma cpu kernel
    Description: test the random gamma alpha is a tensor.
    Expectation: match to tensorflow benchmark.
    """

    shape = Tensor(np.array([3, 1, 2]), ms.int32)
    alpha = Tensor(np.array([[3, 4], [5, 6]]), ms.float32)
    gamma_test = RandomGammaTEST(shape=shape, seed=3)
    expect = np.array([3, 1, 2, 2, 2])

    ms.set_context(mode=ms.GRAPH_MODE, device_target='CPU')
    output = gamma_test(alpha)
    print(output)
    assert (output.shape == expect).all()


if __name__ == '__main__':
    test_random_gamma_op(np.float32)
