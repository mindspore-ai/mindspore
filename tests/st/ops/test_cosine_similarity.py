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
from mindspore import ops


class Net(nn.Cell):
    def construct(self, x1, x2):
        return ops.cosine_similarity(x1, x2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cosine_similarity(mode):
    """
    Feature: cosine_similarity
    Description: Verify the result of cosine_similarity
    Expectation: success
    """
    ms.set_context(mode=mode)
    x1 = ms.Tensor([[-0.0256, 0.0127, -0.2475, 0.2316, 0.8037],
                    [0.5809, -1.2712, -0.7038, -0.2558, 0.7494]], dtype=ms.float32)
    x2 = ms.Tensor([[-0.6115, -0.1965, -0.8484, 0.2389, 0.2409],
                    [1.8940, -2.1997, 0.1915, 0.0856, 0.7542]], dtype=ms.float32)
    net = Net()
    output = net(x1, x2)
    expect_output = np.array([0.4843164, 0.81647635])
    assert np.allclose(output.asnumpy(), expect_output)
