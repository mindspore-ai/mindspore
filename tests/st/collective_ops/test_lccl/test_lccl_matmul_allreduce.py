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
test lccl matmul_allreduce with 8p
"""

import os
import numpy as np

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.communication.management import init, HCCL_WORLD_COMM_GROUP, get_rank, get_group_size
from mindspore.ops import operations as P

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0", "infer_boost": "on"})

init()
rank = get_rank()
size = get_group_size()
x = Tensor(np.random.rand(32, 4096).astype(np.float16)*0.01)
weight1 = np.random.rand(4096, 2048).astype(np.float16)*0.01
weight2 = np.random.rand(2048, 16).astype(np.float16)*0.01


class Net(nn.Cell):
    """
    class for matmul allreduce.
    """
    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(initializer(
            Tensor(weight1), weight1.shape), name='weight1')
        self.weight2 = Parameter(initializer(
            Tensor(weight2), weight2.shape), name='weight2')
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()

        self.op0 = "sum"
        self.op1 = "sum"

        self.all_reduce1 = P.AllReduce(self.op0, group=HCCL_WORLD_COMM_GROUP)
        self.all_reduce2 = P.AllReduce(self.op1, group=HCCL_WORLD_COMM_GROUP)

    def construct(self, input_x):
        output = self.matmul1(input_x, self.weight1)
        output = self.all_reduce1(output)
        output = output * 0.01
        output = self.matmul2(output, self.weight2)
        output = self.all_reduce2(output)
        return output


def test_MatMulAllReduce():
    """
    Feature: lccl MatMulAllReduce fustion operator test.
    Description: lccl MatMulAllReduce 8P case.
    Expectation: success
    """
    os.environ["DISABLE_MATMUL_ALLREDUCE_FUSION"] = "True"
    mmar_no_fusion_net = Net()
    output_no_fusion = mmar_no_fusion_net(x)

    os.environ["DISABLE_MATMUL_ALLREDUCE_FUSION"] = "False"
    mmar_fusion_net = Net()
    mmar_fusion_net.phase = "prefill"
    output_fusion = mmar_fusion_net(x)

    print(output_no_fusion.asnumpy(), output_fusion.asnumpy(),
          output_no_fusion.asnumpy() - output_fusion.asnumpy())
