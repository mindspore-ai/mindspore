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
This test is for msrun launch only call mindspore.communication.init().
"""
import os
import numpy as np

import mindspore as ms
from mindspore.ops import cat
from mindspore.communication import init
from mindspore.mint.distributed.distributed import (
    init_process_group,
    TCPStore,
    all_gather_into_tensor,
)


ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})


if __name__ == "__main__":

    init_approach = os.environ.get('INIT_APPROACH')
    rank = int(os.environ.get('RANK_ID'))
    world_size = int(os.environ.get('MS_WORKER_NUM'))
    ip = os.environ.get('MS_SCHED_HOST')
    port = int(os.environ.get('MS_SCHED_PORT'))

    if init_approach == "INIT_METHOD":
        print("Entry calling init_process_group with provided init_method!", flush=True)
        init_method = f"tcp://{ip}:{port}"
        init_process_group(init_method=init_method, rank=rank, world_size=world_size)
    elif init_approach == "TCPSTORE":
        print("Entry calling init_process_group with provided TcpStore!", flush=True)
        is_master = rank == 0
        store = TCPStore(ip, port, world_size, is_master)
        init_process_group(rank=rank, world_size=world_size, store=store)
    else:
        init()

    input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    output_tensor = ms.Tensor(np.zeros([12, 3]).astype(np.float32))
    output_handle = all_gather_into_tensor(output_tensor, input_tensor)
    except_output_tensor = cat(
        [input_tensor, input_tensor, input_tensor, input_tensor]
    )
    assert output_handle is None
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())

    print("==== End initialize Communication group ====")
