# Copyright 2025 Huawei Technologies Co., Ltd
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
from datetime import timedelta
import mindspore.multiprocessing as mp
from mindspore import Tensor, nn
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore import context
from mindspore.mint.distributed import init_process_group


class AllGatherNet(nn.Cell):
    def __init__(self):
        super(AllGatherNet, self).__init__()
        self.all_gather = P.AllGather()

    def construct(self, x):
        return self.all_gather(x)

def run(world_size, rank):
    context.set_context(jit_level='O0')
    context.set_context(device_target="Ascend")

    init_process_group("hccl", "tcp://127.0.0.1:8228", timedelta(seconds=300), world_size, rank)

    x = np.ones([3, 4]).astype(np.float32)
    net = AllGatherNet()
    expect_output = np.ones([24, 4]).astype(np.float32)
    output = net(Tensor(x, mstype.float32))
    assert np.allclose(output.asnumpy(), expect_output)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    rank_size = 8
    process_list = []
    for i in range(rank_size):
        p = mp.Process(target=run, args=(rank_size, i))
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()
    for p in process_list:
        if p.exitcode != 0:
            raise RuntimeError(f"Process {p.pid} exits with exception!")
