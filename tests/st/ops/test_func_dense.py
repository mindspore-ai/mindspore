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

"""
Dense op test case
"""

import numpy as np
import pytest
import mindspore as ms
from mindspore import nn
from mindspore.graph.api import _pynative_executor
from tests.device_utils import set_device, get_device
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def construct(self, x, weight, bias):
        output = ms.ops.dense(x, weight, bias)
        return output


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.02)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dense(mode):
    """
    Feature: ops.dense
    Description: Verify the result of dense
    Expectation: success
    """
    ms.set_context(mode=mode)
    set_device()
    if mode == ms.GRAPH_MODE and get_device() == "Ascend":
        ms.device_context.ascend.op_precision.precision_mode("force_fp32")
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


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dense_invalid_shape(mode):
    """
    Feature: ops.dense
    Description: input shape is invalid
    Expectation: raise error
    """
    ms.set_context(mode=mode)
    x = ms.Tensor(np.random.uniform(-10, 10, size=())).astype(ms.float32)
    weight = ms.Tensor(np.random.uniform(-10, 10, size=[2, 3])).astype(ms.float32)
    bias = ms.Tensor(np.random.uniform(-10, 10, size=[2])).astype(ms.float32)
    with pytest.raises(ValueError):
        net = Net()
        _ = net(x, weight, bias)
        if mode == ms.PYNATIVE_MODE:
            _pynative_executor.sync()
