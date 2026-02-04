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
"""
test lccl barrier with 8p
"""

import time

import mindspore as ms
from mindspore import runtime, nn
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.ops import operations as P

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0", "infer_boost": "on"})

init()
rank = get_rank()
size = get_group_size()

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.barrier = P.Barrier()

    def construct(self):
        self.barrier()


def test_Barrier():
    """
    Feature: lccl operator test.
    Description: msrun lccl barrier 8P case.
    Expectation: success
    """
    barrier_net = Net()
    if get_rank() in [0, 1, 2, 3]:
        time.sleep(3)
    barrier_net()
    runtime.synchronize()
    print("Time is ", time.time())
