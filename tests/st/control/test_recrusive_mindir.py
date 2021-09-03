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
import os
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export, load

ZERO = Tensor([0], mstype.int32)
ONE = Tensor([1], mstype.int32)


class RecrusiveNet(nn.Cell):
    def construct(self, x, z):
        def f(x, z):
            y = ZERO
            if x < 0:
                y = ONE
            elif x < 3:
                y = x * f(x - 1, z)
            elif x < 5:
                y = x * f(x - 2, z)
            else:
                y = f(x - 4, z)
            z = y + 1 + z
            return z

        return f(x, z)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_recrusive():
    context.set_context(mode=context.GRAPH_MODE)
    network = RecrusiveNet()

    x = Tensor(np.array([1]).astype(np.float32))
    y = Tensor(np.array([2]).astype(np.float32))
    origin_out = network(x, y)

    file_name = "recrusive_net"
    export(network, x, y, file_name=file_name, file_format='MINDIR')
    mindir_name = file_name + ".mindir"
    assert os.path.exists(mindir_name)

    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(x, y)
    assert origin_out == outputs_after_load
