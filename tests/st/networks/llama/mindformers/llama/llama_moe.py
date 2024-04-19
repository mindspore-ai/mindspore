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
"""LLaMA Model Layers' APIs."""

from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator

from .llama_layer import LlamaSiLU
from mindformers.modules.layers import Linear, _check_input_dtype, _args_type_validator_check, _valid_value_checks
from mindformers.modules.transformer.moe import Router, MoEConfig, calculate_expert_capacity
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config, _check_config, default_moeparallel_config

default_moe_config = MoEConfig()

class LlamaMoE(Cell):
    r"""
    LLaMA MoE.
    """
    def __init__(self, dim,
                 hidden_dim,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 moe_config=default_moe_config,
                 parallel_config=default_moeparallel_config):
        super(LlamaMoE, self).__init__()
        self.hidden_size = dim
        self.expert_dim = moe_config.expert_num
        self.capacity_factor = moe_config.capacity_factor
        self.aux_loss_factor = moe_config.aux_loss_factor
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.dp_group = parallel_config.data_parallel
        self.dp = parallel_config.data_parallel
        self.ep = parallel_config.expert_parallel
        self.mp = parallel_config.model_parallel
        self.comp_comm_parallel = moe_config.comp_comm_parallel
        self.comp_comm_parallel_degree = moe_config.comp_comm_parallel_degree
        self.group_wise_a2a = moe_config.group_wise_a2a
        if not (self.mp > 1 and self.dp == self.ep):
            self.group_wise_a2a = False

        self.ffn = LlamaMoEFeedForward(dim=self.hidden_size,
                                       hidden_dim=hidden_dim,
                                       expert_num=self.expert_dim,
                                       compute_dtype=compute_dtype,
                                       param_init_type=param_init_type,
                                       parallel_config=parallel_config)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.transpose_2dim = P.Transpose()
        self.transpose_3dim = P.Transpose()
        self.transpose_4dim = P.Transpose()
        self.transpose_4dim_dp = P.Transpose()
        self.batch_mm = P.BatchMatMul()
        self.batch_mm2 = P.BatchMatMul()
        self.mul = P.Mul()
        self.router = Router(d_model=self.hidden_size, moe_config=moe_config, routing_policy=None,
                             training=True, parallel_config=parallel_config)
        self.cast = P.Cast()
        self.concat = P.Concat(3)
        self.concat_dp = P.Concat(2)
        self.split = P.Split(axis=2, output_num=self.comp_comm_parallel_degree)
        self.stride_slice = P.StridedSlice()
        self.stride_slice_dp = P.StridedSlice()
        self.stride_slice_ep = P.StridedSlice()
        self.stride_slice_dp_mp = P.StridedSlice()
        self.stride_slice_ep_mp = P.StridedSlice()

    def ffn_infer(self, expert_input, capacity):
        """
        Computing the FFN.
        """
        expert_input = self.reshape(expert_input, (self.expert_dim * self.dp_group * capacity,
                                                   self.hidden_size))
        # expert_output's shape: (self.expert_dim, self.dp_group*expert_capacity, self.hidden_size)
        expert_output = self.ffn(expert_input)
        expert_output = self.reshape(expert_output, (self.expert_dim, self.dp_group,
                                                     capacity, self.hidden_size))

        # expert_output's shape: (self.dp_group, self.hidden_size, self.expert_dim, expert_capacity)
        expert_output = self.transpose_4dim(expert_output, (1, 3, 0, 2))
        return expert_output

    def construct(self, input_tensor):
        """forward process"""
        input_shape = F.shape(input_tensor)
        input_tensor = self.reshape(input_tensor, (-1, self.hidden_size))
        bs_and_dmodel = self.shape(input_tensor)
        tokens_per_group = bs_and_dmodel[0] // self.dp_group
        input_tensor = self.reshape(input_tensor, (self.dp_group, tokens_per_group, self.hidden_size))

        expert_capacity = calculate_expert_capacity(self.num_experts_chosen, tokens_per_group,
                                                    self.capacity_factor, self.expert_dim)
        # dispatch_tensor's shape: (self.dp_group, tokens_per_group, self.expert_dim, expert_capacity)
        # combine_tensor's shape: (self.dp_group, tokens_per_group, self.expert_dim, expert_capacity)
        dispatch_tensor, combine_tensor, aux_loss = self.router(input_tensor)

        # after transpose, input_tensor's shape: (self.dp_group, self.hidden_size, tokens_per_group)
        input_tensor = self.transpose_3dim(input_tensor, (0, 2, 1))
        dispatch_tensor = self.reshape(dispatch_tensor, (self.dp_group, tokens_per_group,
                                                         self.expert_dim * expert_capacity))
        dispatch_tensor = self.cast(dispatch_tensor, F.dtype(input_tensor))
        # expert_input's shape: (self.dp_group, self.hidden_size, self.expert_dim * expert_capacity)
        expert_input = self.batch_mm(input_tensor, dispatch_tensor)
        expert_input = self.reshape(expert_input, (self.dp_group, self.hidden_size, self.expert_dim,
                                                   expert_capacity))
        # The following four ops are to implement transpose(expert_input, (2, 0, 3, 1)), for that a single transpose
        # has bad performance
        expert_input = self.reshape(expert_input, (self.dp_group * self.hidden_size,
                                                   self.expert_dim * expert_capacity))
        expert_input = self.transpose_2dim(expert_input, (1, 0))
        expert_input = self.reshape(expert_input, (self.expert_dim, expert_capacity, self.dp_group,
                                                   self.hidden_size))
        # expert_input's shape: (self.expert_dim, self.dp_group, expert_capacity, self.hidden_size)
        expert_input = self.transpose_4dim_dp(expert_input, (0, 2, 1, 3))

        # expert_output's shape: (self.dp_group, self.hidden_size, self.expert_dim, expert_capacity)
        expert_output = self.ffn_infer(expert_input, expert_capacity)

        expert_output = self.reshape(expert_output, (self.dp_group, self.hidden_size,
                                                     self.expert_dim * expert_capacity))
        combine_tensor = self.reshape(combine_tensor, (self.dp_group, tokens_per_group,
                                                       self.expert_dim * expert_capacity))
        # combine_tensor's shape: (self.dp_group, self.expert_dim*expert_capacity, tokens_per_group)
        combine_tensor = self.transpose_3dim(combine_tensor, (0, 2, 1))
        combine_tensor = self.cast(combine_tensor, F.dtype(expert_output))

        # combined_output's shape: (self.dp_group, self.hidden_size, tokens_per_group)
        combined_output = self.batch_mm2(expert_output, combine_tensor)
        # combined_output's shape: (self.dp_group, tokens_per_group, self.hidden_size)
        combined_output = self.transpose_3dim(combined_output, (0, 2, 1))
        combined_output = self.reshape(combined_output, (bs_and_dmodel[0], bs_and_dmodel[1]))
        combined_output = self.reshape(combined_output, input_shape)

        aux_loss = self.mul(self.aux_loss_factor, aux_loss)
        #暂时先不将aux_loss返回
        return combined_output

    def shard(self, parallel_config):
        """Set shard for LlamaMoE"""
        self.transpose_2dim.shard(((self.dp, 1),))
        self.transpose_3dim.shard(((self.dp, 1, 1),))
        self.transpose_4dim.shard(((1, self.dp, 1, 1),))
        self.transpose_4dim_dp.shard(((1, 1, self.dp, 1),))
        self.batch_mm.shard(((self.dp, 1, 1), (self.dp, 1, 1)))
        self.batch_mm2.shard(((self.dp, 1, 1), (self.dp, 1, 1)))
        self.mul.shard(((), ()))
        self.concat.shard(tuple((self.dp, 1, 1, 1) for _ in range(self.comp_comm_parallel_degree)))
        self.concat_dp.shard(((1, self.dp, 1, 1), (1, self.dp, 1, 1)))
        self.split.shard(((1, self.dp, 1, 1),))
        self.stride_slice.shard(((self.dp, 1, 1, 1),))
        self.stride_slice_dp.shard(((1, self.dp, 1, 1),))
        self.stride_slice_ep.shard(((self.ep, 1, 1, 1),))
        self.stride_slice_dp_mp.shard(((1, self.dp, self.mp, 1),))
        self.stride_slice_ep_mp.shard(((self.ep, 1, self.mp, 1),))
        self.ffn.shard(parallel_config)

class LlamaMoEFeedForward(Cell):
    r"""
    Llama MoE FeedForward.
    """
    @_args_type_validator_check(dim=Validator.check_positive_int,
                                hidden_dim=Validator.check_positive_int,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16],
                                                                  "FeedForward"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "FeedForward"))
    def __init__(self, dim,
                 hidden_dim,
                 expert_num=8,
                 hidden_act=LlamaSiLU,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        super(LlamaMoEFeedForward, self).__init__()
        if hidden_act is None or not (isinstance(hidden_act, str) or issubclass(hidden_act, nn.Cell)):
            raise TypeError(f"For FeedForward cell, the hidden_act should str type or nn.Cell type, "
                            f"but got {hidden_act}.")
        _check_config(parallel_config)
        self.expert_num = expert_num
        self.dtype = compute_dtype
        self.hidden_act = hidden_act
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.mul = P.Mul()
        self.cast = P.Cast()

        self.w1 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         expert_num=expert_num,
                         activation=hidden_act,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        self.w2 = Linear(in_channels=hidden_dim,
                         out_channels=dim,
                         expert_num=expert_num,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

        self.w3 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         expert_num=expert_num,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type)

    def construct(self, x):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16], self.cls_name)
        x = self.cast(x, self.dtype)
        # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
        gate = self.w1(x) # dp,1 -> dp, mp
        hidden = self.w3(x) # dp,1 -> dp, mp
        hidden = self.mul(hidden, gate) # dp,mp -> dp, mp
        output = self.w2(hidden) # dp,mp -> dp, 1
        return output

    def shard(self, parallel_config):
        """Set shard for Llama_moe_feedforward"""
        mp = parallel_config.model_parallel
        if self.expert_num > 1:
            ep = parallel_config.expert_parallel
        else:
            ep = 1
        # ffn use less dp than other ops when use_moe, due to there are ops use dp and ep.
        dp = parallel_config.data_parallel // ep
        self.mul.shard(((dp * ep, mp), (dp * ep, mp)))
        if self.expert_num > 1:
            self.w1.shard(strategy_matmul=((dp, ep, 1, 1), (ep, mp, 1)),
                          strategy_activation=((dp, ep, mp, 1),))
            self.w2.shard(strategy_matmul=((dp, ep, 1, mp), (ep, 1, mp)))
            self.w3.shard(strategy_matmul=((dp, ep, 1, 1), (ep, mp, 1)))
        else:
            self.w1.shard(strategy_matmul=((dp, 1), (mp, 1)),
                          strategy_activation=((dp, mp),))
            self.w2.shard(strategy_matmul=((dp, mp), (1, mp)))
            self.w3.shard(strategy_matmul=((dp, 1), (mp, 1)))
