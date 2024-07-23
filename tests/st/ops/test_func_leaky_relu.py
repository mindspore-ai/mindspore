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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops


class Net(nn.Cell):
    def construct(self, x):
        return ops.leaky_relu(x, alpha=10.0)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_leaky_relu(mode):
    """
    Feature: ops.leaky_relu
    Description: Verify the result of leaky_relu
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[-1.51791394, -1.49894616, 1.13866319, 1.05513196, -1.14626506],
                [1.18550548, -0.04312532, -2.73655943, -0.36267431, -1.40520197],
                [-0.13018957, 0.15305919, 0.09166156, 1.67514884, -1.79096267]], ms.float32)
    net = Net()
    output = net(x)
    expect_output = [[-1.51791391e+01, -1.49894619e+01, 1.13866317e+00, 1.05513191e+00, -1.14626503e+01],
                     [1.18550551e+00, -4.31253195e-01, -2.73655930e+01, -3.62674284e+00, -1.40520191e+01],
                     [-1.30189562e+00, 1.53059185e-01, 9.16615576e-02, 1.67514884e+00, -1.79096260e+01]]
    assert np.allclose(output.asnumpy(), expect_output)
