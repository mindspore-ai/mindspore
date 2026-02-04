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

"""test hccl BatchISendIRecv with 8p"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops.operations.comm_ops import BatchISendIRecv
from mindspore.communication.comm_func import batch_isend_irecv, P2POp
from mindspore.communication.management import init, get_rank, get_group_size

# 'BatchISendIRecv' operator only supports KernelByKernel mode by now.
np.random.seed(1)
ms.set_context(jit_level='O0')
init()
this_rank = get_rank()
world_size = get_group_size()

class BatchISendIRecvNet(nn.Cell):
    def __init__(self, op_types, remote_ranks, receive_shapes, receive_types, group=None):
        super(BatchISendIRecvNet, self).__init__()
        self.batch_isend_irecv = BatchISendIRecv(op_types, remote_ranks, receive_shapes, receive_types, group=group)
    def construct(self, x):
        return self.batch_isend_irecv(x)

class BatchISendIRecvFuncNet(nn.Cell):
    def construct(self, p2p_op_list):
        return batch_isend_irecv(p2p_op_list)


def test_hccl_batchisendirecv_8p():
    """
    Feature: test 'BatchISendIRecv' communication operator.
    Description: test 'BatchISendIRecv' communication operator.
    Expectation: expect correct result.
    """
    next_rank = (this_rank + 1) % world_size
    prev_rank = (this_rank + world_size - 1) % world_size
    net = BatchISendIRecvNet(("isend", "irecv"), [next_rank, prev_rank], [()], (ms.float32,))
    send_tensor = ms.Tensor(this_rank + 1, dtype=ms.float32)
    output = net((send_tensor,))

    expect_output = np.array(prev_rank + 1).astype(np.float32)

    assert np.allclose(output[1].asnumpy(), expect_output)

def test_hccl_batchisendirecv_func_8p():
    """
    Feature: test 'BatchISendIRecv' communication operator.
    Description: test 'BatchISendIRecv' communication operator.
    Expectation: expect correct result.
    """
    next_rank = (this_rank + 1) % world_size
    prev_rank = (this_rank + world_size - 1) % world_size
    net = BatchISendIRecvFuncNet()
    send_tensor = ms.Tensor(this_rank, dtype=ms.float32)
    recv_tensor = ms.Tensor(11., dtype=ms.float32)

    send_op = P2POp('isend', send_tensor, next_rank)
    recv_op = P2POp('irecv', recv_tensor, prev_rank)
    expect_output = np.array(prev_rank).astype(np.float32)

    p2p_op_list = [send_op, recv_op]
    output = net(p2p_op_list)
    assert np.allclose(output[1].asnumpy(), expect_output)
