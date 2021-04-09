# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE,
                    device_target='CPU', save_graphs=True)


class UpdateCacheNet(nn.Cell):
    def __init__(self, x):
        super().__init__()
        self.ops = P.UpdateCache()
        self.max_num = 9999
        self.x = Parameter(Tensor(x), name='x')

    def construct(self, indices, update):
        return self.ops(self.x, indices, update, self.max_num)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_update_cache():
    x_np = np.array([[2, 3, 4, 5],
                     [6, 7, 8, 9],
                     [11, 12, 13, 14],
                     [1, 2, 3, 4],
                     [5, 6, 7, 8]], np.int32)

    indices_np = np.array([[-1, 3, 4]], np.int32)
    update_np = np.array([[0, 0, 0, 0],
                          [23, 34, 56, 78],
                          [44, 55, 66, 77]], np.int32)

    indices = Tensor(indices_np)
    update = Tensor(update_np)

    expect = np.array([[2, 3, 4, 5],
                       [6, 7, 8, 9],
                       [11, 12, 13, 14],
                       [23, 34, 56, 78],
                       [44, 55, 66, 77]], np.int32)
    net = UpdateCacheNet(x_np)
    out = net(indices, update)
    assert np.allclose(net.x.data.asnumpy(), expect)
    assert np.allclose(out.asnumpy(), np.array([0], np.int32))
