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

import numpy as np
import argparse
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from tests.device_utils import set_device


class Net(nn.Cell):
    def construct(self, x, y):
        out = ops.add(x, y)
        return out

def test_add():
    """
    Feature: op debug option
    Description: test op debug option with add ops
    Expectation: success
    """
    parser = argparse.ArgumentParser(description="test_op_debug_option")
    parser.add_argument("--run_mode", type=int, default=0, help="GRAPH_MODE is 0, PYNATIVE_MODE is 1")
    args_opt = parser.parse_args()
    if args_opt.run_mode == 0:
        ms.set_context(mode=ms.GRAPH_MODE)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_context(device_target="Ascend")
    set_device()
    ms.device_context.ascend.op_debug.debug_option("oom")
    net = Net()
    x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    y = Tensor(np.array([4, 5, 6]).astype(np.float32))
    net(x, y)

if __name__ == "__main__":
    test_add()
