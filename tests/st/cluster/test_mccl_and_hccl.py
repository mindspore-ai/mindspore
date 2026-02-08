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
"""
This test is for mccl and hccl utilities.
"""
import numpy as np
import mindspore as ms
from mindspore import context, nn, Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.communication import init, get_rank, get_group_size, create_group
from mindspore.mint.distributed.distributed import send, recv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test MCCL and HCCL hybrid communication')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16'],
                        help='Data type for testing (default: float32)')
    return parser.parse_args()

def run_test(dtype_str):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    init()

    rank = get_rank()
    size = get_group_size()

    mccl_rank_list = list(range(0, size))
    print(f"Start create inter-node group mccl_inter_node with {mccl_rank_list}")
    create_group("mccl_inter_node", mccl_rank_list)

    hccl_rank_list1 = list(range(0, size // 2))
    hccl_rank_list2 = list(range(size // 2, size))
    if rank in hccl_rank_list1:
        print(f"Start create intar-node group hccl_intra_node1 with {hccl_rank_list1}")
        create_group("hccl_intra_node1", hccl_rank_list1)
    if rank in hccl_rank_list2:
        print(f"Start create intar-node group hccl_intra_node2 with {hccl_rank_list2}")
        create_group("hccl_intra_node2", hccl_rank_list2)

    class AllReduceNet(nn.Cell):
        def __init__(self, dtype=np.float32):
            super().__init__()
            self.mul = P.Mul()
            if rank in hccl_rank_list1:
                self.all_reduce = P.AllReduce(group="hccl_intra_node1")
            if rank in hccl_rank_list2:
                self.all_reduce = P.AllReduce(group="hccl_intra_node2")
            self.add = P.Add()
            self.y1 = Tensor(np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])).astype(dtype)
            self.y2 = Tensor(np.array([[-16, -16, -16, -16], [-16, -16, -16, -16], [-16, -16, -16, -16]])).astype(dtype)

        def construct(self, x):
            x = self.mul(x, 2)
            z = self.add(x, self.y1)
            z = self.add(z, self.y1)
            z = self.all_reduce(z)
            out = self.add(z, self.y2)
            out = self.all_reduce(out)
            out = self.mul(out, 2)
            return out

    # Convert string to numpy dtype and mindspore dtype
    if dtype_str == 'float32':
        np_dtype = np.float32
        ms_dtype = mstype.float32
    elif dtype_str == 'float16':
        np_dtype = np.float16
        ms_dtype = mstype.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    # Test with specified dtype
    print(f"=== Testing with {dtype_str} ===")
    net = AllReduceNet(dtype=np_dtype)
    input_x = np.ones([3, 4]).astype(np_dtype)
    output = net(Tensor(input_x, ms_dtype))

    input_tensor = output
    recv_output = ms.Tensor(np.zeros([3, 4]).astype(np_dtype))
    if rank // (size // 2) == 0:
        print(f"send rank is {rank} to {rank + size // 2} {input_tensor * rank}", flush=True)
        send(input_tensor * rank, rank + size // 2, group="mccl_inter_node")
    else:
        recv(recv_output, src=rank - size // 2, group="mccl_inter_node")
        print(f"send rank is {rank} from {rank - size // 2} {recv_output.asnumpy()}", flush=True)

if __name__ == "__main__":
    args = parse_args()
    run_test(args.dtype)
