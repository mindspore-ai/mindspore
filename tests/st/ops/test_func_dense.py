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
import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn


class Net(nn.Cell):
    def construct(self, x, weight, bias):
        output = ms.ops.dense(x, weight, bias)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dense(mode):
    """
    Feature: ops.dense
    Description: Verify the result of dense
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[[195, 41, 17],
                    [-15, 26, 160],
                    [-182, -95, 104]],
                   [[-236, -29, 98],
                    [121, -205, 66],
                    [107, -157, -38]]], ms.float32)
    weight = ms.Tensor([[25, 50, 73],
                        [-2, -11, -90],
                        [81, -25, 41],
                        [-12, 20, 23]], ms.float32)
    bias = ms.Tensor([90, 114, -91, -102], ms.float32)
    net = Net()
    output = net(x, weight, bias)
    expect_output = [[[8.25600000e+03, -2.25700000e+03, 1.53760000e+04, -1.23100000e+03],
                      [1.26950000e+04, -1.45420000e+04, 4.60400000e+03, 4.27800000e+03],
                      [-1.61800000e+03, -7.83700000e+03, -8.19400000e+03, 2.57400000e+03]],
                     [[-1.06000000e+02, -7.91500000e+03, -1.44640000e+04, 4.40400000e+03],
                      [-2.31700000e+03, -3.81300000e+03, 1.75410000e+04, -4.13600000e+03],
                      [-7.85900000e+03, 5.04700000e+03, 1.09430000e+04, -5.40000000e+03]]]
    assert np.allclose(output.asnumpy(), expect_output)
