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
from mindspore import nn
from mindspore import context
from mindspore.ops.operations import Cauchy


class Net(nn.Cell):
    """a class used to test CholeskySolve gpu operator."""

    def __init__(self, size, median=0.0, sigma=1.0):
        super(Net, self).__init__()
        self.cauchy = Cauchy(size=size, median=median, sigma=sigma)

    def construct(self):
        """construct."""
        return self.cauchy()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cauchy():
    """
    Feature: CholeskySolve gpu TEST.
    Description: test CholeskySolve operator
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    size = [1000]
    net = Net(size)
    result = net()
    assert list(result.shape) == size

if __name__ == '__main__':
    test_cauchy()
