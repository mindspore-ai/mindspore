# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import ops, nn, Tensor
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_dynamic_broadcast_to():
    """
    Feature: KernelPacket
    Description: DynamicBroadcastTo's input is a TensorShape, the occurred in bprop of UnsortedSegmentMax
    Expectation: success
    """
    from mindspore.ops.operations._inner_ops import DynamicBroadcastTo
    ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE, jit_config={"jit_level": "O1"})
    os.environ["GLOG_v"] = "1"

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.shape = ops.TensorShape()
            self.bt = DynamicBroadcastTo()

        def construct(self, x, y):
            outshape = self.shape(y)
            return self.bt(x, outshape)

    x_dyn = Tensor(shape=None, dtype=ms.float32)
    y_dyn = Tensor(shape=None, dtype=ms.float32)
    net = Net()
    net.set_inputs(x_dyn, y_dyn)

    os.environ["MS_SIMULATION_LEVEL"] = "1"
    x = Tensor(np.ones([2]), dtype=ms.float32)
    y = Tensor(np.ones([2, 2]), dtype=ms.float32)
    output = net(x, y)
    assert output.shape == y.shape

    y = Tensor(np.ones([5, 4, 2]), dtype=ms.float32)
    output = net(x, y)
    assert output.shape == y.shape

    x = Tensor(np.ones([1, 3]), dtype=ms.float32)
    y = Tensor(np.ones([5, 3]), dtype=ms.float32)
    output = net(x, y)
    assert output.shape == y.shape
    del os.environ["MS_SIMULATION_LEVEL"]
    del os.environ["GLOG_v"]
