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
from mindspore import context, Parameter
from mindspore.parallel.shard import Layout
from mindspore.graph.api import _cell_graph_executor
from mindspore.common.initializer import initializer
from mindspore.nn import Cell, TrainOneStepCell, Momentum
import mindspore.common.dtype as mstype
import mindspore.ops as ops

class Linear(Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 transpose_b=True,
                 expert_num=1,
                 outer_batch=1,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16):
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        self.expert_num = expert_num
        self.outer_batch = outer_batch
        self.transpose_b = transpose_b
        self.expert_flag = True
        self.weight = Parameter(initializer(weight_init, [self.expert_num] + weight_shape, param_init_type),
                                name="weight")
        self.matmul = ops.BatchMatMul(transpose_b=transpose_b)

        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = Parameter(initializer(bias_init, [1, self.expert_num, 1, out_channels], param_init_type),
                                  name="bias")
            self.bias.parallel_optimizer = False
            self.bias_add = ops.Add()

        self.dtype = compute_dtype
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()

    def construct(self, x):
        out_shape = self.shape(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        if self.expert_flag:
            x = self.reshape(x, (self.outer_batch, self.expert_num, -1, self.in_channels))
        ori_dtype = ops.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        x = self.cast(x, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        x = ops.cast(x, ori_dtype)
        output = self.reshape(x, out_shape)
        return output

    def shard(self, strategy_matmul, strategy_bias=None, out_strategy_matmul=None):
        self.matmul.shard(in_strategy=strategy_matmul, out_strategy=out_strategy_matmul)
        if self.has_bias:
            self.bias_add.shard(strategy_bias)
        return self


class MoEFFNet(Cell):
    def __init__(self, hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, sp, has_bias=True, transpose_b=False,
                 bmm_output_sharding=False):
        super(MoEFFNet, self).__init__()
        input_size = hidden_size
        output_size = ffn_hidden_size
        param_init_type = mstype.float16
        compute_dtype = mstype.float16
        self.dp = dp
        self.use_sp = sp
        outer_bs = dp
        if sp:
            outer_bs = dp * ep
        self.mapping = Linear(in_channels=input_size,
                              out_channels=output_size,
                              has_bias=has_bias,
                              transpose_b=transpose_b,
                              expert_num=expert_num,
                              outer_batch=outer_bs,
                              param_init_type=param_init_type,
                              compute_dtype=compute_dtype)

        self.projection = Linear(in_channels=output_size,
                                 out_channels=input_size,
                                 has_bias=has_bias,
                                 transpose_b=transpose_b,
                                 expert_num=expert_num,
                                 outer_batch=outer_bs,
                                 param_init_type=param_init_type,
                                 compute_dtype=compute_dtype)

        if transpose_b:
            self.mapping.shard(strategy_matmul=((dp, ep, 1, 1), (ep, mp, 1)),
                               strategy_bias=((dp, ep, 1, mp), (1, ep, 1, mp)))
            if not bmm_output_sharding:
                self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, 1, mp)),
                                      strategy_bias=((dp, ep, mp, 1), (1, ep, 1, 1)))
            else:
                layout = Layout((dp, ep, mp), ("dp", "ep", "mp"))
                bmm_input_layout = (layout("dp", "ep", "None", "mp"), layout("ep", "None", "mp"))
                bmm_output_layout = (layout("dp", "ep", ("None", "mp"), "None"),)
                self.projection.shard(strategy_matmul=bmm_input_layout,
                                      strategy_bias=((dp, ep, mp, 1), (1, ep, 1, 1)),
                                      out_strategy_matmul=bmm_output_layout)
        else:
            self.mapping.shard(strategy_matmul=((dp, ep, 1, 1), (ep, 1, mp)),
                               strategy_bias=((dp, ep, 1, mp), (1, ep, 1, mp)))
            if not bmm_output_sharding:
                self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)),
                                      strategy_bias=((dp, ep, mp, 1), (1, ep, 1, 1)))
            else:
                self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)),
                                      strategy_bias=((dp, ep, mp, 1), (1, ep, 1, 1)),
                                      out_strategy_matmul=((dp, ep, mp, 1),))
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.stride_slice_ep = ops.StridedSlice().shard(((ep, 1, 1, 1),))
        self.stride_slice_ep_mp = ops.StridedSlice().shard(((ep, 1, mp, 1),))
        self.stride_slice_dp_ep = ops.StridedSlice().shard(((dp, ep, 1, 1, 1),))
        self.stride_slice_dp_ep_mp = ops.StridedSlice().shard(((dp, ep, 1, mp, 1),))
        self.stride_slice_sp_ep = ops.StridedSlice().shard(((dp, 1, ep, 1, 1),))
        self.stride_slice_sp_ep_mp = ops.StridedSlice().shard(((dp, 1, ep, mp, 1),))

    def construct_with_sp(self, x):
        x_shape = self.shape(x)
        x = self.stride_slice_sp_ep(x, (0, 0, 0, 0, 0), x_shape, (1, 1, 1, 1, 1))
        hidden = self.mapping(x)
        output = self.projection(hidden)
        output1 = self.reshape(output, x_shape)
        output2 = self.stride_slice_sp_ep_mp(output1, (0, 0, 0, 0, 0), x_shape, (1, 1, 1, 1, 1))
        return output2

    def construct_with_dp(self, x):
        x_shape = self.shape(x)
        x = self.stride_slice_dp_ep(x, (0, 0, 0, 0, 0), x_shape, (1, 1, 1, 1, 1))
        hidden = self.mapping(x)
        output = self.projection(hidden)
        output1 = self.reshape(output, x_shape)
        output2 = self.stride_slice_dp_ep_mp(output1, (0, 0, 0, 0, 0), x_shape, (1, 1, 1, 1, 1))
        return output2

    def construct(self, x):
        if self.use_sp:
            return self.construct_with_sp(x)
        if self.dp > 1:
            return self.construct_with_dp(x)
        x_shape = self.shape(x)
        x = self.stride_slice_ep(x, (0, 0, 0, 0), x_shape, (1, 1, 1, 1))
        hidden = self.mapping(x)
        output = self.projection(hidden)
        output1 = self.reshape(output, x_shape)
        output2 = self.stride_slice_ep_mp(output1, (0, 0, 0, 0), x_shape, (1, 1, 1, 1))
        return output2


def compile_net(net, x):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, x)
    context.reset_auto_parallel_context()
