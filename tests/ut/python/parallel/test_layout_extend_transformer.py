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

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.nn import Cell
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.parallel.shard import Layout
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

grad_all = C.GradOperation(get_all=True)

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


def compile_net(net, input_x, y):
    net.set_auto_parallel()
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, input_x, y)
    return phase

class _LayerNorm(Cell):
    def __init__(self, normalized_shape, eps=1e-5, param_init_type=mstype.float32):
        super(_LayerNorm, self).__init__()
        self.layer_norm = P.LayerNorm(begin_norm_axis=-1,
                                      begin_params_axis=-1,
                                      epsilon=eps)
        self.gamma = Parameter(initializer('ones', normalized_shape, param_init_type), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape, param_init_type), name="beta",
                              parallel_optimizer=False)

    def construct(self, x):
        output, _, _ = self.layer_norm(x, self.gamma, self.beta)
        return output

    def shard(self, strategy):
        self.layer_norm.shard(strategy)
        return self


class _Linear(Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 transpose_b=True,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16):
        super(_Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        self.expert_flag = False
        self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")
        self.matmul = P.MatMul(transpose_b=transpose_b)
        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = Parameter(initializer(bias_init, [out_channels], param_init_type), name="bias")
            self.bias_add = P.Add()
        self.dtype = compute_dtype
        self.cast = P.Cast()

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.out_channels,)
        x = P.Reshape()(x, (-1, self.in_channels))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        output = P.Reshape()(x, out_shape)
        return output

    def shard(self, strategy_matmul, out_strategy_matmul=None, strategy_bias=None):
        self.matmul.shard(strategy_matmul, out_strategy=out_strategy_matmul)
        if self.has_bias:
            self.bias_add.shard(strategy_bias)
        return self


class Net(nn.Cell):
    def __init__(self, hidden_size, ffn_hidden_size, dp, mp, vp):
        super().__init__()
        transfomer_layout = Layout((dp, mp, vp), ("dp", "mp", "vp"))
        input_size = hidden_size
        output_size = ffn_hidden_size
        param_init_type = mstype.float32
        compute_dtype = mstype.float16
        self.mapping = _Linear(in_channels=input_size,
                               out_channels=output_size,
                               transpose_b=False,
                               param_init_type=param_init_type).to_float(compute_dtype)
        self.mapping.shard((transfomer_layout(("dp", "vp"), "None"), transfomer_layout("None", "mp")))
        self.projection = _Linear(in_channels=output_size,
                                  out_channels=input_size,
                                  transpose_b=False,
                                  param_init_type=param_init_type).to_float(compute_dtype)
        self.projection.shard((transfomer_layout(("dp", "vp"), "mp"), transfomer_layout("mp", "None")),
                              (transfomer_layout(("dp", "vp", "mp"), "None"),),
                              (transfomer_layout(("dp", "vp", "mp"), "None"), transfomer_layout("None")))
        self.attention_projection = _Linear(in_channels=hidden_size,
                                            out_channels=hidden_size,
                                            transpose_b=False,
                                            param_init_type=param_init_type).to_float(compute_dtype)
        self.attention_projection.shard((transfomer_layout(("dp", "vp"), "mp"), transfomer_layout("mp", "None")),
                                        (transfomer_layout(("dp", "vp", "mp"), "None"),),
                                        (transfomer_layout(("dp", "vp", "mp"), "None"), transfomer_layout("None")))
        self.qkv_dense = _Linear(hidden_size,
                                 hidden_size,
                                 param_init_type=param_init_type).to_float(compute_dtype)
        self.qkv_dense.shard((transfomer_layout(("dp", "vp"), "None"), transfomer_layout("mp", "None")))
        self.gelu = P.GeLU().shard((transfomer_layout(("dp", "vp"), "mp"),))
        self.add1 = P.Add().shard((transfomer_layout(("dp", "vp", "mp"), "None"),
                                   transfomer_layout(("dp", "vp", "mp"), "None")))
        self.add2 = P.Add().shard((transfomer_layout(("dp", "vp", "mp"), "None"),
                                   transfomer_layout(("dp", "vp", "mp"), "None")))
        self.begin1 = P.ReLU().shard(((dp, mp),))
        self.begin2 = P.ReLU().shard(((dp, 1),))
        self.end1 = P.ReLU().shard(((dp, mp),))
        self.end2 = P.ReLU().shard(((dp * mp, 1),))
        self.layernorm1 = _LayerNorm((hidden_size,)).to_float(mstype.float32)
        self.layernorm2 = _LayerNorm((hidden_size,)).to_float(mstype.float32)
        self.layernorm1.shard((transfomer_layout(("dp", "vp", "mp"), "None"),
                               transfomer_layout("None"), transfomer_layout("None")))
        self.layernorm2.shard((transfomer_layout(("dp", "vp", "mp"), "None"),
                               transfomer_layout("None"), transfomer_layout("None")))

    def construct(self, input_ids, y):
        # input_ids: (seq_len, hidden_size), y: (seq_len ,hidden_size)
        input_ids_begin = self.begin1(input_ids)
        add_begin = self.begin2(y)
        attention_proj = self.attention_projection(input_ids_begin)
        ffn_input = self.add1(attention_proj, add_begin)
        mapping_input = self.layernorm2(ffn_input)
        hidden = self.mapping(mapping_input)
        hidden = self.gelu(hidden)
        ffn_out = self.projection(hidden)
        ffn_out = self.add2(ffn_out, ffn_input)
        qkv_input = self.layernorm1(ffn_out)
        attention_input = self.qkv_dense(qkv_input)
        end1 = self.end1(attention_input)
        end2 = self.end2(ffn_out)
        return end1 + end2


def test_layout_extend_transformer():
    """
    Feature: test layout extend with tranformer, modify sequence micro interleaved
    Description: dp4,mp8,vp2.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=64, global_rank=0)
    bs_seq_len = 4096 * 4
    hidden_size = 12288
    ffn_hidden_size = hidden_size * 3
    dp = 4
    mp = 8
    vp = 2
    input_ids = Tensor(np.ones([bs_seq_len, hidden_size]), dtype=ms.float32)
    y = Tensor(np.ones([bs_seq_len, hidden_size]), dtype=ms.float32)
    net = Net(hidden_size, ffn_hidden_size, dp, mp, vp)
    _ = compile_net(net, input_ids, y)
