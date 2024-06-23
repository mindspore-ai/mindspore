# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.ops import operations as P
import mindspore.communication.management as D
from mindspore import context, Tensor


grad = C.GradOperation(get_all=True)


class Net(nn.Cell):
    def __init__(self, seq_len, hidden_size, dp, mp):
        super(Net, self).__init__()
        self.layer_norm = nn.LayerNorm((hidden_size,))
        self.dense1 = nn.Dense(in_channels=hidden_size, out_channels=hidden_size).to_float(mstype.float16)
        self.dense2 = nn.Dense(in_channels=hidden_size, out_channels=hidden_size).to_float(mstype.float16)
        self.gelu = P.Gelu()

        self.layer_norm.layer_norm.shard(((dp * mp, 1), (1,), (1,)))
        self.dense1.matmul.shard(((dp, 1), (mp, 1)))
        self.dense1.bias_add.shard(((dp, mp), (mp,)))
        self.gelu.shard(((dp, mp),))
        self.dense2.matmul.shard(((dp, mp), (1, mp)), out_strategy=((dp * mp, 1),))
        self.dense2.bias_add.shard(((dp * mp, 1), (1,)))

    def construct(self, x):
        out = self.layer_norm(x)
        out = self.dense1(out)
        out = self.gelu(out)
        out = self.dense2(out)
        return out


class AllGatherMatmulNet(nn.Cell):
    def __init__(self, seq_len, hidden_size, dp, mp):
        super(AllGatherMatmulNet, self).__init__()
        self.gelu1 = P.Gelu()
        self.dense1 = nn.Dense(in_channels=hidden_size,
                               out_channels=hidden_size,
                               weight_init="ones").to_float(mstype.float16)
        self.gelu2 = P.Gelu()
        self.gelu1.shard(((dp * mp, 1),))
        self.dense1.matmul.shard(((dp, 1), (mp, 1)))
        self.dense1.bias_add.shard(((dp, mp), (mp,)))
        self.gelu2.shard(((dp, mp),))

    def construct(self, x):
        out = self.gelu1(x)
        out = self.dense1(out)
        out = self.gelu2(out)
        return out


class MatmulReduceScatterNet(nn.Cell):
    def __init__(self, seq_len, hidden_size, dp, mp):
        super(MatmulReduceScatterNet, self).__init__()
        self.dense1 = nn.Dense(in_channels=hidden_size,
                               out_channels=hidden_size,
                               weight_init="ones").to_float(mstype.float16)
        self.gelu1 = P.Gelu()
        self.gelu2 = P.Gelu()

        self.gelu1.shard((((dp, 1),)))
        self.dense1.matmul.shard(((dp, mp), (1, mp)), out_strategy=((dp * mp, 1),))
        self.dense1.bias_add.shard(((dp * mp, 1), (1,)))
        self.gelu2.shard(((dp, mp),))

    def construct(self, x):
        out = self.gelu1(x)
        out = self.dense1(out)
        out = self.gelu2(out)
        return out


class GradNet(nn.Cell):
    def __init__(self, network):
        super(GradNet, self).__init__()
        self.network = network

    def construct(self, *inputs):
        gout = grad(self.network)(*inputs)
        return gout


def test_all_gather_matmul_forward():
    '''
    Feature: MC2 fusion.
    Description: Test all_gather-matmul fusion in forward.
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_context(jit_config={"jit_level": "O2"})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")
    D.init()
    seq_len, hidden_size = 4096, 12288
    dp, mp = 1, 8
    x = Tensor(np.random.uniform(-3, 3, [seq_len, hidden_size]), dtype=mstype.float16)

    net = AllGatherMatmulNet(seq_len, hidden_size, dp, mp)
    expect_out = net(x).asnumpy()

    context.set_context(ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_for_mc2.json"})
    mc2_net = AllGatherMatmulNet(seq_len, hidden_size, dp, mp)
    mc2_out = mc2_net(x).asnumpy()

    assert np.allclose(expect_out, mc2_out, 1e-2, 1e-2)


def test_matmul_reduce_scatter_forward():
    '''
    Feature: MC2 fusion.
    Description: Test matmul-reduce_scatter fusion in forward.
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_context(jit_config={"jit_level": "O2"})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")
    D.init()
    seq_len, hidden_size = 4096, 12288
    dp, mp = 1, 8
    x = Tensor(np.random.uniform(-3, 3, [seq_len, hidden_size]), dtype=mstype.float16)

    net = MatmulReduceScatterNet(seq_len, hidden_size, dp, mp)
    expect_out = net(x).asnumpy()

    context.set_context(ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_for_mc2.json"})
    mc2_net = MatmulReduceScatterNet(seq_len, hidden_size, dp, mp)
    mc2_out = mc2_net(x).asnumpy()

    assert np.allclose(expect_out, mc2_out, 1e-2, 1e-2)


def test_mc2_fusion_forward_backward():
    '''
    Feature: MC2 fusion.
    Description: Test MC2 fusion
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")
    context.set_context(ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_for_mc2.json"})
    D.init()
    seq_len, hidden_size = 4096, 12288
    dp, mp = 1, 8
    x = Tensor(np.random.uniform(-3, 3, [seq_len, hidden_size]), dtype=mstype.float16)

    grad_net = GradNet(Net(seq_len, hidden_size, dp, mp))
    grad_net(x)
