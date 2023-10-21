# Copyright 2023 Huawei Technologies Co., Ltd
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


from mindspore import Tensor, nn
import mindspore as mstype
import numpy as np
import pytest


class LUSolveNet(nn.Cell):
    def construct(self, x, LU, pivots):
        return x.lu_solve(LU, pivots)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_lu_solve():
    """
    Feature: ALL To ALL
    Description: test cases for lu_solve
    Expectation: the result matches with torch
    """
    net = LUSolveNet()
    input_lu = Tensor([[1.6253573, 1.4034185, -0.9425243],
                       [0.19567108, 1.6314834, -0.96950316],
                       [0.26517096, 0.45115343, -0.5865267]])
    pivots = Tensor([2, 3, 3], dtype=mstype.int32)
    input_x = Tensor([[0.8851084, -0.8193832], [0.15885238, -1.0667698], [0.25483948, -0.3558243]])
    output_x = net(input_x, input_lu, pivots)
    out = np.array([[-0.10517889, -0.5249513], [-0.6146541, 0.38617927], [-1.2651373, 0.80157894]])
    assert np.allclose(out, output_x.asnumpy())
