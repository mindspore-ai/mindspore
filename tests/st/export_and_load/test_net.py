# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Export net test."""
import os
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import export
from tests.mark_utils import arg_mark


class SliceNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x, y):
        x = self.relu(x)
        x[2,] = y
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_export_slice_net():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_x = Tensor(np.random.rand(4, 4, 4), ms.float32)
    input_y = Tensor(np.array([1]), ms.float32)
    net = SliceNet()
    file_name = "slice_net"
    export(net, input_x, input_y, file_name=file_name, file_format='AIR')
    verify_name = file_name + ".air"
    assert os.path.exists(verify_name)
    os.remove(verify_name)
    export(net, input_x, input_y, file_name=file_name, file_format='MINDIR')

    verify_name = file_name + ".mindir"
    assert os.path.exists(verify_name)
    os.remove(verify_name)
