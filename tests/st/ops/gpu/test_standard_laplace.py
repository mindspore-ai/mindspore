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
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class StandardLaplaceMock(nn.Cell):
    def __init__(self, seed=0, seed2=0):
        super(StandardLaplaceMock, self).__init__()
        self.seed = seed
        self.seed2 = seed2
        self.stdlaplace = P.StandardLaplace(seed, seed2)

    def construct(self, shape):
        return self.stdlaplace(shape)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_standard_laplace_net():
    """
        Feature: Standard Laplace gpu kernel
        Description: Generates random numbers according to the Laplace random number distribution (mean=0, lambda=1).
        Expectation: The shape of the input matches the output
    """
    seed = 10
    seed2 = 10
    shape = (3, 2, 4)
    expect_shape = (3, 2, 4)
    net = StandardLaplaceMock(seed, seed2)
    output = net(shape)
    assert output.shape == expect_shape
