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
"""Export const tuple net test."""
import os
import numpy as np

import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import export
from tests.mark_utils import arg_mark


class ConstTupleNet(nn.Cell):
    def __init__(self, t1):
        super().__init__()
        self.tuple1 = t1

    def construct(self):
        return self.tuple1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_export_const_tuple_net():
    """
    Feature: export AIR.
    Description: test net which return const tuple can be exported in AIR format.
    Expectation: air file can be exported successfully.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_np1 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np1)
    t1 = (input_me_x,)
    net = ConstTupleNet(t1)

    file_name = "const_tuple_net"
    export(net, file_name=file_name, file_format='AIR')
    verify_name = file_name + ".air"
    assert os.path.exists(verify_name)
    os.remove(verify_name)
