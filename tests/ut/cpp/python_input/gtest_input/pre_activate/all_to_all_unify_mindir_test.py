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
import mindspore as ms
from mindspore import context
from mindspore.ops.operations.comm_ops import AlltoAll, NeighborExchange
from mindspore.communication.management import GlobalComm, init

context.set_context(device_target="Ascend")
GlobalComm.CHECK_ENVS = False
init("hccl")
GlobalComm.CHECK_ENVS = True

class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]

def test_neighbor_exchange(tag):
    fns = FnDict()
    neighbor = NeighborExchange(send_rank_ids=[0], recv_rank_ids=[1], recv_shapes=([2, 2],), send_shapes=([2, 2],),
                                recv_type=ms.float32)
    @fns
    def before(x):
        return neighbor(x)

    return fns[tag]

def test_all_to_all(tag):
    context.set_auto_parallel_context(device_num=2, global_rank=0)
    fns = FnDict()
    altoall = AlltoAll(split_count=2, split_dim=2, concat_dim=3)
    @fns
    def before(x):
        return altoall(x)

    return fns[tag]
