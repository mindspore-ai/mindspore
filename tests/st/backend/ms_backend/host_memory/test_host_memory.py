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
# ==============================================================================
'''
Check host memory increment
'''
import os
import psutil
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from tests.mark_utils import arg_mark

class ContextModel(nn.Cell):
    hidden_dim = 32

    def __init__(self, num_types=27, num_mode=6):
        super().__init__()
        layers = nn.TransformerEncoderLayer(self.hidden_dim, 1,
                                            dim_feedforward=self.hidden_dim * 2, batch_first=True, norm_first=True)

        self.transformers = nn.TransformerEncoder(layers, 2)
        self.embed_type = nn.Embedding(num_types, self.hidden_dim)
        self.embed_mode = nn.Embedding(num_mode, self.hidden_dim)
        self.embed_power = nn.Dense(1, self.hidden_dim)
        self.embed_prb = nn.Dense(1, self.hidden_dim)
        self.embed_energy = nn.Dense(1, self.hidden_dim)

        self.predict_slop = nn.SequentialCell(nn.Dense(self.hidden_dim, self.hidden_dim),
                                              nn.ReLU(),
                                              nn.Dense(self.hidden_dim, 1))
        self.predict_bias = nn.SequentialCell(nn.Dense(self.hidden_dim, self.hidden_dim),
                                              nn.ReLU(),
                                              nn.Dense(self.hidden_dim, 1))
        self.ms_soft_plus = ms.ops.Softplus()

        self.tanh = nn.Tanh()

    @staticmethod
    def soft_plus(x, beta=ms.Tensor([np.log(2)])):
        return ms.ops.log(1 + ms.ops.exp(beta * x)) / beta

    def construct(self, ctx_type, ctx_data, query, initial):
        ctx_type, ctx_data, query = ctx_type.astype('float32'), ctx_data.astype('float32'), query.astype('float32')
        embedded_type = self.embed_type(ctx_type[:, :, 0].astype('int'))
        embedded_mode = self.embed_mode(ctx_type[:, :, 1].astype('int'))
        embedded_power = self.embed_power(ctx_type[:, :, 2]).unsqueeze(1)

        embedded_prb = self.embed_prb(ctx_data[:, :, 0].unsqueeze(-1))
        embedded_energy = self.embed_energy(ctx_data[:, :, 1].unsqueeze(-1))

        query_power = self.embed_power(query[:, :, 0].unsqueeze(-1))
        query_prb = self.embed_prb(query[:, :, 1].unsqueeze(-1))

        ctx = ms.ops.cat([embedded_type, embedded_mode, embedded_power], 1)

        power_prb_energy = ms.ops.stack([embedded_prb, embedded_energy], -1).permute(0, 1, 3, 2)
        power_prb_energy = self.flatten(power_prb_energy)

        query_power_prb = self.flatten(ms.ops.cat([query_power, query_prb], 1).unsqueeze(1))

        embedding = ms.ops.cat([ctx, power_prb_energy, query_power_prb], 1)

        embed_ctx = self.transformers(embedding)

        decay = self.tanh(query[:, :, 0] * 10 - ctx_type[:, :, 2] * 10)
        coe_slop = self.soft_plus(self.ms_soft_plus(self.predict_slop(embed_ctx[:, -1, :])) * decay)
        coe_bias = self.soft_plus(self.ms_soft_plus(self.predict_bias(embed_ctx[:, -1, :])) * decay)

        init_slope, init_intercept = initial[:, 0].unsqueeze(1), initial[:, 1].unsqueeze(1)

        coe_slop = coe_slop * init_slope
        coe_bias = coe_bias * init_intercept
        return coe_slop * query[:, :, 1] + coe_bias

    def flatten(self, inputs):
        return inputs.reshape(inputs.shape[0], -1, self.hidden_dim)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=490.00)
def test_host_memory_with_cyclic_multiple_model():
    """
    Feature: host memory
    Description: test pass host memory increment with cyclic multiple model 
    Expectation: host memory increment less than 700M
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    pid = os.getpid()
    process = psutil.Process(pid)
    memory_bytes_start = process.memory_info().rss
    for _ in range(10):
        model = ContextModel()
        ctx_type_input = Tensor(np.random.rand(12220, 1, 3), dtype=ms.float64)
        ctx_data_input = Tensor(np.random.rand(12220, 16, 2), dtype=ms.float64)
        query_input = Tensor(np.random.rand(12220, 1, 2), dtype=ms.float64)
        initial_input = Tensor(np.random.rand(12220, 2), dtype=ms.float64)
        _ = model(ctx_type_input, ctx_data_input, query_input, initial_input)
        model = None
    memory_bytes_end = process.memory_info().rss
    memory_bytes_increment = memory_bytes_end - memory_bytes_start
    print(f"Start memory bytes: {memory_bytes_start}B, End memory bytes: {memory_bytes_end}B, \
          increment: {memory_bytes_increment}B", flush=True)
    # increment less than 700M
    assert memory_bytes_increment < 700000000
