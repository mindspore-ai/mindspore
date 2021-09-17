# Copyright 2021 Huawei Technologies Co., Ltd
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
Note: Mixture of Expert (MoE) structure. This is an experimental interface that is subject to change and/or deletion.
"""
import math
import numpy as np
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr
from mindspore.nn.cell import Cell
from mindspore.nn.layer import Dense
from .op_parallel_config import default_dpmp_config

__all__ = [
    "MoEConfig"]


class MoEConfig:
    r"""
        The configuration of MoE (Mixture of Expert).

        Args:
            expert_num (int): The number of experts employed. Default: 1
            capacity_factor (float): The factor is used to indicate how much to expand expert capacity,
                which is >=1.0. Default: 1.1.
            aux_loss_factor (float): The factor is used to indicate how much the load balance loss (produced by the
                router) to be added to the entire model loss, which is < 1.0. Default: 0.05.
            num_experts_chosen (int): The number of experts is chosen by each token. Default: 1.
            noisy_policy (string): The noisy policy is used in routing tokens to experts. Default: None.
            noisy_epsilon (float): The parameter is used in adding noises in routing tokens to experts. Default: 1e-2.
    """
    def __init__(self, expert_num=1, capacity_factor=1.1, aux_loss_factor=0.05,
                 num_experts_chosen=1, noisy_policy=None, noisy_epsilon=1e-2):
        self.expert_num = expert_num
        self.capacity_factor = capacity_factor
        self.aux_loss_factor = aux_loss_factor
        self.num_experts_chosen = num_experts_chosen
        self.noisy_policy = noisy_policy
        self.noisy_epsilon = noisy_epsilon

default_moe_config = MoEConfig()

@constexpr
def calculate_expert_capacity(k, tokens_per_device, capacity_factor, expert_dim):
    return math.ceil(k * tokens_per_device * capacity_factor / expert_dim)


class MoE(Cell):
    """
    The mixture of experts (MoE) implementation. The implementation includes a router and a FeedForward layer.
    The router dispatches tokens to experts in FeedForward, then FeedForward does computation, and the final output is
    obtained by multiplying FeedForward's output and router's combine weight.

    Args:
        hidden_size (int): The dimension of the inputs.
        ffn_hidden_size (int): The intermediate hidden size.
        dropout_rate (float): The dropout rate for the second linear's output.
        hidden_act (str): The activation of the internal feedforward layer. Supports 'relu',
                         'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                         'hsigmoid', 'logsigmoid' and so on. Default: gelu.
        param_init_type (dtype.Number): The parameter initialization type. Can be dtype.float32 or dtype.float16.
        moe_config(MoEConfig): The configuration of MoE (Mixture of Expert).
        parallel_config(OpParallelConfig): The config of parallel setting, see `OpParallelConfig`.
                                           Default `default_dpmp_config`, a instance of `OpParallelConfig` with default
                                           args.

    Inputs:
        - **x** (Tensor) - should be `[batch, seq_length, hidden_size]`. Float tensor.

    Outputs:
        Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size]`.
    """
    def __init__(self, hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 hidden_act='gelu',
                 param_init_type=mstype.float32,
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config):
        super(MoE, self).__init__()
        self.hidden_size = hidden_size
        self.expert_dim = moe_config.expert_num
        self.capacity_factor = moe_config.capacity_factor
        self.aux_loss_factor = moe_config.aux_loss_factor
        self.num_experts_chosen = moe_config.num_experts_chosen
        self.expert_parallel = parallel_config.data_parallel
        self.dp = parallel_config.data_parallel
        from .transformer import FeedForward

        self.ffn = FeedForward(hidden_size=hidden_size,
                               ffn_hidden_size=ffn_hidden_size,
                               dropout_rate=dropout_rate,
                               hidden_act=hidden_act,
                               expert_num=self.expert_dim,
                               param_init_type=param_init_type,
                               parallel_config=parallel_config)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.transpose = P.Transpose().shard(((self.dp, 1, 1),))
        self.transpose2 = P.Transpose().shard(((self.dp, 1, 1, 1),))
        self.transpose3 = P.Transpose().shard(((self.dp, 1, 1, 1),))
        self.transpose4 = P.Transpose().shard(((self.dp, 1, 1),))
        self.transpose5 = P.Transpose().shard(((self.dp, 1, 1),))
        self.batch_mm = P.BatchMatMul().shard(((self.dp, 1, 1), (self.dp, 1, 1)))
        self.batch_mm2 = P.BatchMatMul().shard(((self.dp, 1, 1), (self.dp, 1, 1)))
        self.mul = P.Mul().shard(((), ()))
        self.router = Router(d_model=hidden_size, moe_config=moe_config, routing_policy=None,
                             training=True, parallel_config=parallel_config)
        self.cast = P.Cast()


    def construct(self, input_tensor):
        bs = self.shape(input_tensor)[0]
        input_tensor = self.reshape(input_tensor, (-1, self.hidden_size))
        bs_and_dmodel = self.shape(input_tensor)
        tokens_per_device = bs_and_dmodel[0] / self.expert_parallel
        input_tensor = self.reshape(input_tensor, (self.expert_parallel, tokens_per_device, self.hidden_size))

        expert_capacity = calculate_expert_capacity(self.num_experts_chosen, tokens_per_device,
                                                    self.capacity_factor, self.expert_dim)
        # dispatch_tensor's shape: (self.expert_parallel, tokens_per_device, self.expert_dim, expert_capacity)
        # combine_tensor's shape: (self.expert_parallel, tokens_per_device, self.expert_dim, expert_capacity)
        dispatch_tensor, combine_tensor, aux_loss = self.router(input_tensor)

        # after transpose, input_tensor's shape: (self.expert_parallel, self.hidden_size, tokens_per_device)
        input_tensor = self.transpose(input_tensor, (0, 2, 1))
        dispatch_tensor = self.reshape(dispatch_tensor, (self.expert_parallel, tokens_per_device,
                                                         self.expert_dim * expert_capacity))
        dispatch_tensor = self.cast(dispatch_tensor, F.dtype(input_tensor))
        # expert_input's shape: (self.expert_parallel, self.hidden_size, self.expert_dim * expert_capacity)
        expert_input = self.batch_mm(input_tensor, dispatch_tensor)
        expert_input = self.reshape(expert_input, (self.expert_parallel, self.hidden_size, self.expert_dim,
                                                   expert_capacity))
        # expert_input's shape: (self.expert_dim, self.expert_parallel, expert_capacity, self.hidden_size)
        expert_input = self.transpose2(expert_input, (2, 0, 3, 1))
        expert_input = self.reshape(expert_input, (self.expert_dim, self.expert_parallel * expert_capacity,
                                                   self.hidden_size))

        # expert_output's shape: (self.expert_dim, self.expert_parallel*expert_capacity, self.hidden_size)
        expert_output = self.ffn(expert_input)
        expert_output = self.reshape(expert_output, (self.expert_dim, self.expert_parallel,
                                                     expert_capacity, self.hidden_size))
        # expert_output's shape: (self.expert_parallel, self.hidden_size, self.expert_dim, expert_capacity)
        expert_output = self.transpose3(expert_output, (1, 3, 0, 2))
        expert_output = self.reshape(expert_output, (self.expert_parallel, self.hidden_size,
                                                     self.expert_dim*expert_capacity))
        combine_tensor = self.reshape(combine_tensor, (self.expert_parallel, tokens_per_device,
                                                       self.expert_dim*expert_capacity))
        # combine_tensor's shape: (self.expert_parallel, self.expert_dim*expert_capacity, tokens_per_device)
        combine_tensor = self.transpose4(combine_tensor, (0, 2, 1))
        combine_tensor = self.cast(combine_tensor, F.dtype(expert_output))

        # combined_output's shape: (self.expert_parallel, self.hidden_size, tokens_per_device)
        combined_output = self.batch_mm2(expert_output, combine_tensor)
        # combined_output's shape: (self.expert_parallel, tokens_per_device, self.hidden_size)
        combined_output = self.transpose5(combined_output, (0, 2, 1))
        combined_output = self.reshape(combined_output, (bs_and_dmodel[0], bs_and_dmodel[1]))
        combined_output = self.reshape(combined_output, (bs, -1, self.hidden_size))

        aux_loss = self.mul(self.aux_loss_factor, aux_loss)
        return combined_output, aux_loss


class _CumSum(Cell):
    r"""
        A layer used to calculate cumulative summation of a tensor along a dimension.

        Inputs:
            - **expert_mask** (Tensor) - Tensor of shape :math:`(expert\_parallel, tokens\_per\_device,
            expert\_dim)`.

        Outputs:
            Tensor of shape :math:`(expert\_parallel, tokens\_per\_device, expert\_dim)`.
    """

    def __init__(self, config):
        super(_CumSum, self).__init__()
        dp = config.data_parallel
        self.range = P.Range().shard(((1,),))
        self.reshape = P.Reshape()
        self.matmul = P.MatMul().shard(((dp, 1), (1, 1)))
        self.shape = P.Shape()
        self.cast = P.Cast()

        self.transpose = P.Transpose().shard(((dp, 1, 1),))
        self.transpose2 = P.Transpose().shard(((1, 1),))
        self.transpose3 = P.Transpose().shard(((dp, 1, 1),))
        self.expand = P.ExpandDims().shard(((1,),))
        self.greater = P.Greater().shard(((1, 1), (1, 1)))

        self.start = Tensor(0, mstype.int32)
        self.limit = Tensor(0, mstype.int32)
        self.delta = Tensor(1, mstype.int32)
        self.add = P.Add().shard(((1,), ()))

    def construct(self, expert_mask):
        # origin_shape: (expert_parallel, tokens_per_device, self.expert_dim)
        origin_shape = self.shape(expert_mask)
        tokens_per_device = origin_shape[1]
        # expert_mask_trans's shape: (expert_parallel, self.expert_dim, tokens_per_device)
        expert_mask_trans = self.transpose(expert_mask, (0, 2, 1))
        # expert_mask_reshaped's shape: (expert_parallel*self.expert_dim, tokens_per_device)
        expert_mask_reshaped = self.reshape(expert_mask_trans, (-1, tokens_per_device))

        one_dim = self.expand(self.range(self.start, self.add(self.limit, tokens_per_device), self.delta), 0)
        other_dim = self.transpose2(one_dim, (1, 0))
        # up_tri_matrix's shape: (tokens_per_device, tokens_per_device)
        up_tri_matrix = self.greater(one_dim, other_dim)
        up_tri_matrix = self.cast(up_tri_matrix, mstype.float32)

        # cum_sum's shape: (expert_parallel*self.expert_dim, tokens_per_device)
        cum_sum = self.matmul(expert_mask_reshaped, up_tri_matrix)
        # cum_sum's shape: (expert_parallel, self.expert_dim, tokens_per_device)
        cum_sum = self.reshape(cum_sum, (origin_shape[0], origin_shape[2], tokens_per_device))
        # cum_sum's shape: (expert_parallel, tokens_per_device, self.expert_dim)
        cum_sum = self.transpose3(cum_sum, (0, 2, 1))
        return cum_sum


class Router(Cell):
    r"""
        A router backbone used to calculate logits of each token, which should be cascaded by router implementations
        mapping tokens to experts.

        Args:
            d_model (int): The hidden size of each token.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert).
            routing_policy: The policy of mapping tokens to experts. Default: SwitchRouter
            training (bool): The value indicating whether is in training phase.
            parallel_config: The parallel-related configuration.
        Inputs:
            - **input_tensor** (Tensor) - Tensor of shape :math:`(expert\_parallel, tokens\_per\_device,
            hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(expert\_parallel, tokens\_per\_device, expert\_dim)`.
    """

    def __init__(self,
                 d_model,
                 moe_config,
                 routing_policy=None,
                 training=True,
                 parallel_config=None):
        super(Router, self).__init__()
        dp = parallel_config.data_parallel
        self.d_model = d_model
        self.expert_dim = moe_config.expert_num
        self.capacity_factor = moe_config.capacity_factor
        self.training = training
        self.routing_policy = routing_policy
        self.noisy_policy = moe_config.noisy_policy  # candidate: ["jitter", "rsample", "None"]
        self.noisy_epsilon = moe_config.noisy_epsilon
        self.noise = Tensor(np.random.uniform(1 - self.noisy_epsilon, 1 + self.noisy_epsilon, (d_model,)))

        self.dense = Dense(in_channels=self.d_model, out_channels=self.expert_dim, has_bias=False)
        self.dense.matmul.shard(((dp, 1), (1, 1)))
        self.mul = P.Mul().shard(((dp, 1, 1), (dp,)))
        self.cast = P.Cast()

        if self.routing_policy is None:
            self.router = SwitchRouter(d_model=d_model, moe_config=moe_config, training=training,
                                       parallel_config=parallel_config)
        else:
            self.router = routing_policy

    def construct(self, input_tensor):
        input_tensor = self.cast(input_tensor, mstype.float32)
        if self.noisy_policy == "jitter" and self.training is True:
            # Here, we temporarily implement the multiplicative jitter this way,
            # for the lack of UniforReal parallel operator.
            input_tensor = self.mul(input_tensor, self.noise)

        router_logits = self.dense(input_tensor)
        return self.router(router_logits)


class SwitchRouter(Cell):
    r"""
        A router implementation which maps each tokens to the top1 expert.
        Reference: https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py

        Args:
            d_model (int): The hidden size of each token.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert).
            training (bool): The value indicating whether is in training phase.
            config: The parallel-related configuration.
        Inputs:
            - **input_tensor** (Tensor) - Tensor of shape :math:`(expert\_parallel, tokens\_per\_device,
            hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(expert\_parallel, tokens\_per\_device, expert\_dim, expert\_capacity)`,
            Tensor of shape :math:`(expert\_parallel, tokens\_per\_device, expert\_dim, expert\_capacity)`,
            Tensor of shape :math:`(1)`.
    """

    def __init__(self,
                 d_model,
                 moe_config,
                 training=True,
                 parallel_config=None):
        super(SwitchRouter, self).__init__()
        dp = parallel_config.data_parallel
        self.d_model = d_model
        self.expert_dim = moe_config.expert_num
        self.capacity_factor = moe_config.capacity_factor
        self.training = training
        self.expert_parallel = dp
        self.noisy_policy = moe_config.noisy_policy
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.softmax = P.Softmax(axis=-1).shard(((dp, 1, 1,),))
        self.argmax = P.ArgMaxWithValue(axis=-1, keep_dims=False).shard(((dp, 1, 1),))

        self.onehot = P.OneHot().shard(((dp, 1, 1), (), ()))
        self.onehot2 = P.OneHot().shard(((dp, 1, 1), (), ()))
        self.onehot3 = P.OneHot().shard(((dp, 1, 1, 1), (), ()))
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

        self.reduce_mean = P.ReduceMean(keep_dims=False).shard(((dp, 1, 1),))
        self.reduce_mean2 = P.ReduceMean(keep_dims=False).shard(((dp, 1, 1),))
        self.reduce_mean3 = P.ReduceMean(keep_dims=False).shard(((dp, 1),))
        self.mul = P.Mul().shard(((dp, 1), (dp, 1)))
        self.mul2 = P.Mul().shard(((1,), ()))
        self.mul3 = P.Mul().shard(((1,), ()))
        self.mul4 = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
        self.mul5 = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
        self.mul6 = P.Mul().shard(((dp, 1), (dp, 1)))
        self.mul7 = P.Mul().shard(((dp, 1), (dp, 1)))
        self.mul8 = P.Mul().shard(((dp, 1, 1), (dp, 1, 1)))
        self.mul9 = P.Mul().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))

        self.cumsum = _CumSum(config=parallel_config)
        self.less = P.Less().shard(((dp, 1, 1), ()))
        self.reduce_sum = P.ReduceSum(keep_dims=False).shard(((dp, 1, 1),))
        self.expand = P.ExpandDims().shard(((dp, 1),))
        self.expand2 = P.ExpandDims().shard(((dp, 1, 1),))

    def _auxiliary_loss(self, expert_mask, router_prob):
        """
        Computing the load balance loss.
        """
        # density_1's shape: (expert_parallel, self.expert_dim)
        density_1 = self.reduce_mean(expert_mask, 1)
        # density_1_proxy's shape: (expert_parallel, self.expert_dim)
        density_1_proxy = self.reduce_mean2(router_prob, 1)
        loss = self.mul(density_1, density_1_proxy)
        loss = self.reduce_mean3(loss)
        loss = self.mul3(self.mul2(loss, self.expert_dim), self.expert_dim)
        return loss

    def _maskout_overflowed_tokens(self, expert_mask, expert_capacity, expert_gate):
        """
        Keeping only the tokens that fit within expert_capacity.
        """
        cumsum = self.cumsum(expert_mask)
        # position_in_expert's shape: (expert_parallel, tokens_per_device, self.expert_dim)
        position_in_expert = self.mul4(cumsum, expert_mask)
        less_result = self.less(position_in_expert, expert_capacity)
        # expert_mask's shape: (expert_parallel, tokens_per_device, self.expert_dim)
        expert_mask = self.mul5(less_result, expert_mask)
        # expert_mask_flat's shape: (expert_parallel, tokens_per_device)
        expert_mask_flat = self.reduce_sum(expert_mask, -1)

        # Mask out the experts that have overflowed the expert_capacity.
        # expert_gate's shape: (expert_parallel, tokens_per_device)
        expert_gate = self.mul6(expert_gate, expert_mask_flat)
        return expert_gate, expert_mask_flat, position_in_expert

    def construct(self, router_logits):
        router_logits_shape = self.shape(router_logits)
        router_logits = self.reshape(router_logits, (-1, router_logits_shape[-1]))
        logits_shape = self.shape(router_logits)
        tokens_per_device = logits_shape[0] / self.expert_parallel
        expert_capacity = calculate_expert_capacity(1, tokens_per_device, self.capacity_factor, self.expert_dim)
        router_logits = self.reshape(router_logits, (self.expert_parallel, tokens_per_device, self.expert_dim))
        # Currently, lack of gumbel sampler for router_logits.

        # Probabilities for each token of what expert is should be sent to
        router_prob = self.softmax(router_logits)
        # shape is : (expert_parallel, tokens_per_device)
        expert_index, expert_gate = self.argmax(router_prob)
        # expert_mask's shape: (expert_parallel, tokens_per_device, self.expert_dim)
        expert_mask = self.onehot(expert_index, self.expert_dim, self.on_value, self.off_value)

        # Computing the load balance loss:
        loss = self._auxiliary_loss(expert_mask, router_prob)

        expert_gate, expert_mask_flat, position_in_expert = \
            self._maskout_overflowed_tokens(expert_mask, expert_capacity, expert_gate)

        # combine_tensor's shape: (expert_parallel, tokens_per_device)
        combine_tensor = self.mul7(expert_gate, expert_mask_flat)
        # combine_tensor's shape: (expert_parallel, tokens_per_device, self.expert_dim)
        combine_tensor = self.mul8(self.expand(combine_tensor, -1),
                                   self.onehot2(expert_index, self.expert_dim, self.on_value, self.off_value))
        # combine_tensor's shape: (expert_parallel, tokens_per_device, self.expert_dim, self.expert_capacity)
        combine_tensor = self.mul9(self.expand2(combine_tensor, -1),
                                   self.onehot3(self.cast(position_in_expert, mstype.int32), expert_capacity,
                                                self.on_value, self.off_value))
        dispatch_tensor = self.cast(combine_tensor, mstype.bool_)
        return dispatch_tensor, combine_tensor, loss
