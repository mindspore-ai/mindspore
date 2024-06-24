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
from tests.mark_utils import arg_mark
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, ops, nn


class Net(nn.Cell):
    def construct(self, x, output_size):
        return ops.adaptive_max_pool1d(x, output_size)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_adaptive_max_pool1d_3d(mode):
    """
    Feature: ops.adaptive_max_pool1d
    Description: Verify the result of adaptive_max_pool1d of 3d
    Expectation: success
    """
    ms.set_context(mode=mode)
    a = np.arange(16).reshape(2, 2, 4).astype(np.float32)
    x = Tensor(a)
    except_out_val = np.array([[[1., 3.],
                                [5., 7.]],
                               [[9., 11.],
                                [13., 15.]]], dtype=np.float32)
    net = Net()
    out = net(x, 2)
    assert np.allclose(out.asnumpy(), except_out_val)
