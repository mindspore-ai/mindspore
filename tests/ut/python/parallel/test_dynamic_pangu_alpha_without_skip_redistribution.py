# Copyright 2023 Huawei Technologies Co., Ltd
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
"""PanguAlpha model"""

import os
import math
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer, Normal, TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.common.api import _cell_graph_executor
from mindspore import context, Tensor, Parameter, lazy_inline
from mindspore.nn import TrainOneStepCell, Momentum
from mindspore.ops import composite as C
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size
from mindspore.parallel._utils import _get_enable_parallel_optimizer
from mindspore.nn.wrap.cell_wrapper import PipelineCell
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Dropout(nn.Cell):
    r"""
        A Dropout Implements
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__()
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError(
                "dropout probability should be a number in range (0, 1], but got {}".format(
                    keep_prob))
        self.keep_prob = keep_prob
        self.dropout = P.Dropout(keep_prob)

    def construct(self, x):
        r"""
           Input: a tensor
           Returns: a tensor
        """
        if not self.training:
            return x

        out, _ = self.dropout(x)
        return out


class LayerNorm(nn.Cell):
    r"""
        A self-defined layer norm operation using reduce sum and reduce mean
    """

    def __init__(self, normalized_shape, dp=4, eps=1e-5, scale=1e-3):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(initializer('ones', normalized_shape), name="gamma")
        self.beta = Parameter(initializer('zeros', normalized_shape), name="beta")
        self.mean = P.ReduceMean(keep_dims=True).shard(((dp, 1, 1),))
        self.square = P.Square().shard(((dp, 1, 1),))
        self.sqrt = P.Sqrt().shard(((dp, 1, 1),))
        self.sub1 = P.Sub().shard(((dp, 1, 1), (dp, 1, 1)))
        self.sub2 = P.Sub().shard(((dp, 1, 1), (dp, 1, 1)))
        self.add = P.TensorAdd().shard(((dp, 1, 1), ()))
        self.eps = eps
        self.mul = P.Mul().shard(((dp, 1, 1), (1,)))
        self.add2 = P.TensorAdd().shard(((dp, 1, 1), (1,)))
        self.real_div = P.RealDiv().shard(((dp, 1, 1), (dp, 1, 1)))
        self.scale_div = P.RealDiv().shard(((dp, 1, 1), ()))
        self.scale_mul = P.Mul().shard(((dp, 1, 1), ()))
        self.scale = scale

    def construct(self, x):
        mean = self.mean(x, -1)
        diff = self.sub1(x, mean)
        variance = self.mean(self.square(diff), -1)
        variance_eps = self.sqrt(self.add(variance, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        return output


class Mapping(nn.Cell):
    """
    A mapping function with a 3d input
    Args:
        input_size: the size of the last dimension of the input tensor
        output_size: the desired size of the last dimension of the output tensor
        dtype: the compute datatype
        scale: the scale factor for initialization
    Inputs:
        x: the 3d input
    Returns:
        output: Tensor, a 3d tensor after projection
    """

    def __init__(self, config, input_size, output_size, scale=1.0):
        super(Mapping, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.weight = Parameter(initializer(Normal(sigma=0.02 * scale),
                                            [input_size, output_size], config.param_init_type),
                                name="mapping_weight")
        self.bias = Parameter(initializer("zeros", [
            output_size,
        ], config.param_init_type),
                              name="mapping_bias",
                              parallel_optimizer=False)
        self.dtype = config.compute_dtype
        self.cast = P.Cast()
        self.add = P.TensorAdd().shard(((config.dp, 1), (1,)))
        self.matmul = P.MatMul().shard(((config.dp, config.mp), (config.mp, 1)))
        self.reshape = P.Reshape()

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.output_size,)
        x = self.reshape(x, (-1, self.input_size))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        x = self.add(x, self.cast(self.bias, self.dtype))
        output = self.reshape(x, out_shape)
        return output


class Mapping_output(nn.Cell):
    """
    A mapping function with a 3d input
    Args:
        input_size: the size of the last dimension of the input tensor
        output_size: the desired size of the last dimension of the output tensor
        dtype: the compute datatype
        scale: the scale factor for initialization
    Inputs:
        x: the 3d input
    Returns:
        output: Tensor, a 3d tensor after projection
    """

    def __init__(self, config, input_size, output_size, scale=1.0):
        super(Mapping_output, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.weight = Parameter(initializer(Normal(sigma=0.02 * scale),
                                            [input_size, output_size],
                                            config.param_init_type),
                                name="mapping_weight")
        self.bias = Parameter(initializer("zeros", [
            output_size,
        ], config.param_init_type),
                              name="mapping_bias")
        self.dtype = config.compute_dtype
        self.cast = P.Cast()
        self.add = P.TensorAdd().shard(((config.dp, config.mp), (config.mp,)))
        self.matmul = P.MatMul().shard(((config.dp, 1), (1, config.mp)))
        self.reshape = P.Reshape()

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.output_size,)
        x = self.reshape(x, (-1, self.input_size))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        x = self.add(x, self.cast(self.bias, self.dtype))
        output = self.reshape(x, out_shape)
        return output


class Output(nn.Cell):
    """
    The output mapping module for each layer
    Args:
        config(PanguAlphaConfig): the config of network
        scale: scale factor for initialization
    Inputs:
        x: output of the self-attention module
    Returns:
        output: Tensor, the output of this layer after mapping
    """

    def __init__(self, config, scale=1.0):
        super(Output, self).__init__()
        input_size = config.embedding_size
        output_size = config.embedding_size * config.expand_ratio
        # Project to expand_ratio*embedding_size
        self.mapping = Mapping_output(config, input_size, output_size)
        # Project back to embedding_size
        self.projection = Mapping(config, output_size, input_size, scale)
        self.activation = P.GeLU().shard(((config.dp, 1, config.mp),))
        self.dropout = Dropout(1 - config.dropout_rate)
        self.dropout.dropout.shard(((config.dp, 1, 1),))

    def construct(self, x):
        # [bs, seq_length, expand_ratio*embedding_size]
        hidden = self.activation(self.mapping(x))
        output = self.projection(hidden)
        # [bs, seq_length, expand_ratio]
        output = self.dropout(output)
        return output


class AttentionMask(nn.Cell):
    r"""
    Get the attention matrix for self-attention module
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        input_mask: the mask indicating whether each position is a valid input
    Returns:
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
    """

    def __init__(self, config):
        super(AttentionMask, self).__init__()
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((1, 1),))
        ones = np.ones(shape=(config.seq_length, config.seq_length))
        # Default lower triangle mask matrix
        self.lower_triangle_mask = Tensor(np.tril(ones), mstype.float32)
        self.multiply = P.Mul().shard(((config.dp, 1, 1), (1, 1, 1)))

    def construct(self, input_mask):
        r"""
        Generate the attention mask matrix.
        """
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        # Mask the padded inputs
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        lower_traiangle = self.expand_dim(self.lower_triangle_mask, 0)
        # [bs, seq_length, seq_length]
        attention_mask = self.multiply(
            attention_mask, lower_traiangle)
        return attention_mask


class EmbeddingLookup(nn.Cell):
    """
    The embedding lookup table for vocabulary
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs with datatype int32
    Returns:
        output: Tensor, the embedding vector for the input with shape (batch_size,
        seq_length, embedding_size)
        self.embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self, config):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        if config.load_ckpt_path:
            # Loading the embedding table from the ckpt path:
            embedding_path = os.path.join(config.load_ckpt_path, 'word_embedding.npy')
            if os.path.exists(embedding_path):
                e_table = np.load(embedding_path)
                e_table = Tensor(e_table, mstype.float32)
                self.embedding_table = Parameter(e_table, name="embedding_table")
            else:
                raise ValueError(f"{embedding_path} file not exits, please check whether word_embedding file exist.")
        else:
            self.embedding_table = Parameter(initializer(
                Normal(0.02), [self.vocab_size, self.embedding_size]), name="embedding_table")
        if config.word_emb_dp:
            self.gather = P.Gather().shard(((1, 1), (config.dp, 1)))
        else:
            self.gather = P.Gather().shard(((config.mp, 1), (1, 1)))
        self.shape = (-1, config.seq_length, config.embedding_size)

    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table


class Attention(nn.Cell):
    """
    Self-Attention module for each layer

    Args:
        config(PanguAlphaConfig): the config of network
        scale: scale factor for initialization
        layer_idx: current layer index
    """

    def __init__(self, config, scale=1.0, layer_idx=None):
        super(Attention, self).__init__()
        # Attention mask matrix
        self.get_attention_mask = AttentionMask(config)
        # Output layer
        self.projection = Mapping(config, config.embedding_size,
                                  config.embedding_size, scale)
        self.transpose = P.Transpose().shard(((config.dp, 1, config.mp, 1),))
        self.merger_head_transpose = P.Transpose().shard(
            ((config.dp, config.mp, 1, 1),))
        self.reshape = P.Reshape()
        self.n_head = config.num_heads
        # embedding size per head
        self.size_per_head = config.embedding_size // self.n_head
        self.concat_k = P.Concat(axis=3)
        self.concat_v = P.Concat(axis=2)
        self.multiply_data = Tensor([
            -10000.0,
        ], dtype=mstype.float32)
        self.batch_matmul = P.BatchMatMul().shard(
            ((config.dp, config.mp, 1, 1), (config.dp, config.mp, 1, 1)))
        self.scale = scale
        self.real_div = P.RealDiv().shard(((config.dp, config.mp, 1, 1), ()))
        self.sub = P.Sub().shard(
            ((1,), (config.dp, 1, 1, 1)))
        self.mul = P.Mul().shard(
            ((config.dp, 1, 1, 1), (1,)))
        self.add = P.TensorAdd().shard(
            ((config.dp, 1, 1, 1), (config.dp, config.mp, 1, 1)))
        # Normalize factor for attention, sqrt(dk) as widely used
        if self.scale:
            self.scale_factor = Tensor(math.sqrt(self.size_per_head))
        if layer_idx is not None:
            self.coeff = math.sqrt(layer_idx * math.sqrt(self.size_per_head))
            self.coeff = Tensor(self.coeff)
        self.use_past = config.use_past
        self.dropout = Dropout(1 - config.dropout_rate)
        self.dropout.dropout.shard(((config.dp, 1, 1),))
        self.prob_dropout = Dropout(1 - config.dropout_rate)
        self.prob_dropout.dropout.shard(((config.dp, config.mp, 1, 1),))
        self.softmax = nn.Softmax()
        self.softmax.softmax.shard(((config.dp, config.mp, 1),))
        self.expand_dims = P.ExpandDims().shard(((config.dp, 1, 1),))

        dense_shape = [config.embedding_size, config.embedding_size]
        bias_shape = [config.embedding_size]
        # Query
        self.dense1 = nn.Dense(config.embedding_size,
                               config.embedding_size,
                               weight_init=initializer(init='normal', shape=dense_shape,
                                                       dtype=config.param_init_type),
                               bias_init=initializer(init='zeros', shape=bias_shape,
                                                     dtype=config.param_init_type)).to_float(config.compute_dtype)
        self.dense1.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense1.bias_add.shard(((config.dp, config.mp), (config.mp,)))
        # Key
        self.dense2 = nn.Dense(config.embedding_size,
                               config.embedding_size,
                               weight_init=initializer(init='normal',
                                                       shape=dense_shape,
                                                       dtype=config.param_init_type),
                               bias_init=initializer(init='zeros',
                                                     shape=bias_shape,
                                                     dtype=config.param_init_type)).to_float(config.compute_dtype)
        self.dense2.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense2.bias_add.shard(((config.dp, config.mp), (config.mp,)))
        # Value
        self.dense3 = nn.Dense(config.embedding_size,
                               config.embedding_size,
                               weight_init=initializer(init='normal',
                                                       shape=dense_shape,
                                                       dtype=config.param_init_type),
                               bias_init=initializer(init='zeros',
                                                     shape=bias_shape,
                                                     dtype=config.param_init_type)).to_float(config.compute_dtype)
        self.dense3.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense3.bias_add.shard(((config.dp, config.mp), (config.mp,)))

    def construct(self, x, attention_mask, layer_past=None):
        """
        self-attention

        Inputs:
            x: output of previous layer
            attention_mask: the attention mask matrix with shape (batch_size, 1,
            seq_length, seq_length)
            layer_past: the previous feature map

        Returns:
            output: Tensor, the output logit of this layer
            layer_present: Tensor, the feature map of current layer
        """

        original_shape = F.shape(x)
        x = self.reshape(x, (-1, original_shape[-1]))
        # Self attention: query, key, value are derived from the same inputs
        query = self.dense1(x)
        key = self.dense2(x)
        value = self.dense3(x)
        # [bs, num_heads, seq_length, size_per_head]
        query = self.transpose(
            self.reshape(
                query,
                (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # [bs, num_heads, size_per_head, seq_length]
        key = self.transpose(
            self.reshape(
                key, (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        # [bs, num_heads, seq_length, size_per_head]
        value = self.transpose(
            self.reshape(
                value,
                (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        if self.use_past:
            past_value = layer_past[1]
            past_key = self.transpose(layer_past[0], (0, 1, 3, 2))
            key = self.concat_k((past_key, key))
            value = self.concat_v(past_value, value)
        key_present = key
        value_present = value
        layer_present = (key_present, value_present)
        # Self-attention considering attention mask
        attention = self._attn(query, key, value, attention_mask)
        # [bs, seq_length, embedding_size]
        attention_merge = self.merge_heads(attention)
        # Output
        output = self.projection(attention_merge)
        output = self.dropout(output)
        return output, layer_present

    def split_heads(self, x, transpose):
        """
        split 3d tensor to 4d and switch certain axes
        Inputs:
            x: input tensor
            transpose: tuple, the transpose sequence
        Returns:
            x_transpose: the 4d output
        """
        x_size = P.Shape()(x)
        new_x_shape = x_size[:-1] + (self.n_head, self.size_per_head)
        x = self.reshape(x, new_x_shape)
        x_transpose = self.transpose(x, transpose)
        return x_transpose

    def merge_heads(self, x):
        """
        convert a 4d input to a 3d output

        Inputs:
            x: input tensor

        Returns:
            x_merge: the 3d output
        """
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = P.Shape()(x)
        new_shape = x_shape[:-2] + (x_shape[-2] * x_shape[-1],)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size,
            1, seq_length, seq_length)
        Returns:
            weighted_values: Tensor, the weighted sum scores
        """
        # Normalize query and key before MatMul, default off
        if not self.scale:
            query = query / F.cast(self.coeff, F.dtype(query))
            key = key / F.cast(self.coeff, F.dtype(key))

        # Attention score [bs, num_heads, seq_length, seq_length]
        score = self.batch_matmul(query, key)
        # Normalize after query and key MatMul, default on
        if self.scale:
            score = self.real_div(
                score,
                P.Cast()(self.scale_factor, P.DType()(score)))

        ori_dtype = P.DType()(score)
        score = P.Cast()(score, mstype.float32)
        # Minus 10000 for the position where masked to exclude them from softmax
        multiplu_out = self.sub(
            P.Cast()(F.tuple_to_array((1.0,)), P.DType()(score)),
            P.Cast()(attention_mask, P.DType()(score)))

        adder = self.mul(multiplu_out, self.multiply_data)
        attention_scores = self.add(adder, score)

        shape = F.shape(attention_scores)
        # attention probs
        # FIXME: Problem could be here
        attention_probs = self.softmax(self.reshape(attention_scores, (shape[0], -1, shape[-1])))
        attention_probs = P.Cast()(attention_probs, ori_dtype)
        attention_probs = self.reshape(attention_probs, shape)

        attention_probs = self.prob_dropout(attention_probs)
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]
        weighted_values = self.batch_matmul(attention_probs, value)
        return weighted_values


class Block(nn.Cell):
    """
    The basic block of PanguAlpha network
    Args:
        config(PanguAlphaConfig): the config of network
        layer_idx: current layer index
    Inputs:
        x: the output of previous layer(input_ids for the first layer)
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
        layer_past: the previous feature map
    Returns:
        output: Tensor, the output logit of this layer
        layer_present: Tensor, the feature map of current layer
    """

    def __init__(self, config, layer_idx):
        super(Block, self).__init__()
        scale = 1 / math.sqrt(2.0 * config.num_layers)

        if config.self_layernorm:
            self.layernorm1 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
            self.layernorm2 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        else:
            self.layernorm1 = nn.LayerNorm((config.embedding_size,)).to_float(mstype.float32)
            self.layernorm1.layer_norm.shard(((config.dp, 1, 1), (1,), (1,)))
            self.layernorm2 = nn.LayerNorm((config.embedding_size,)).to_float(mstype.float32)
            self.layernorm2.layer_norm.shard(((config.dp, 1, 1), (1,), (1,)))

        self.layernorm1.gamma.parallel_optimizer = False
        self.layernorm1.beta.parallel_optimizer = False
        self.attention = Attention(config, scale, layer_idx)
        self.layernorm2.gamma.parallel_optimizer = False
        self.layernorm2.beta.parallel_optimizer = False
        # Feed Forward Network, FFN
        self.output = Output(config, scale)
        self.post_layernorm_residual = config.post_layernorm_residual
        self.add = P.TensorAdd().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.last_add = P.TensorAdd().shard(
            ((config.dp, 1, 1), (config.dp, 1, 1)))
        # Last activation of this layer will be saved for recompute in backward process
        self.last_add.recompute(False)
        self.dtype = config.compute_dtype

    def construct(self, x, input_mask, layer_past=None):
        r"""
        The forward process of the block.
        """
        # [bs, seq_length, embedding_size]
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)
        attention, layer_present = self.attention(input_x, input_mask,
                                                  layer_past)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)
        if self.post_layernorm_residual:
            output = self.last_add(output_x, mlp_logit)
        else:
            output = self.last_add(x, mlp_logit)
        return output, layer_present


class QueryLayerAttention(Attention):
    r"""
    Self-Attention module using input query vector.
    """

    def construct(self, x, query_hidden_state, attention_mask, layer_past=None):
        original_shape = F.shape(x)
        x = self.reshape(x, (-1, original_shape[-1]))
        query_hidden_state = self.reshape(query_hidden_state, (-1, original_shape[-1]))
        # For query_layer_attention, query are derived from outputs of previous layer and key, value are derived from an added parameter query_embedding
        query = self.dense1(query_hidden_state)
        key = self.dense2(x)
        value = self.dense3(x)
        query = self.transpose(
            self.reshape(
                query,
                (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        key = self.transpose(
            self.reshape(
                key, (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        value = self.transpose(
            self.reshape(
                value,
                (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        if self.use_past:
            past_value = layer_past[1]
            past_key = self.transpose(layer_past[0], (0, 1, 3, 2))
            key = self.concat_k((past_key, key))
            value = self.concat_v(past_value, value)
        key_present = key
        value_present = value
        layer_present = (key_present, value_present)
        attention = self._attn(query, key, value, attention_mask)
        attention_merge = self.merge_heads(attention)
        output = self.projection(attention_merge)
        output = self.dropout(output)
        return output, layer_present


class QueryLayer(nn.Cell):
    r"""
    A block usingooked out position embedding as query vector.
    This is used as the final block.
    """

    def __init__(self, config):
        super(QueryLayer, self).__init__()
        scale = 1 / math.sqrt(2.0 * config.num_layers)
        self.layernorm1 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        self.layernorm2 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        self.layernorm1.gamma.parallel_optimizer = False
        self.layernorm1.beta.parallel_optimizer = False
        self.attention = QueryLayerAttention(config, scale)
        self.layernorm2.gamma.parallel_optimizer = False
        self.layernorm2.beta.parallel_optimizer = False
        self.output = Output(config, scale)
        self.post_layernorm_residual = config.post_layernorm_residual
        self.add = P.TensorAdd().shard(((config.dp, 1, 1), (config.dp, 1, 1)))

        self.last_add = P.TensorAdd().shard(
            ((config.dp, 1, 1), (config.dp, 1,
                                 1))).add_prim_attr("recompute", False)
        self.dtype = config.compute_dtype

    def construct(self, x, query_hidden_state, input_mask, layer_past=None):
        r"""
        Query Layer shares a similar structure with normal layer block
        except that it is not a traditional self-attention.
        """
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)
        attention, layer_present = self.attention(input_x,
                                                  query_hidden_state,
                                                  input_mask,
                                                  layer_past)
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)
        if self.post_layernorm_residual:
            output = self.last_add(output_x, mlp_logit)
        else:
            output = self.last_add(x, mlp_logit)
        return output, layer_present


class PanguAlpha_Model(nn.Cell):
    """
    The backbone of PanguAlpha network
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs with datatype int32
        input_mask: the mask indicating whether each position is a valid input
        layer_past: the previous feature map
    Returns:
        output_state: Tensor, the output logit of backbone
        present_layer: Tensor, the current feature map
        embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self, config):
        super(PanguAlpha_Model, self).__init__()
        self.get_attention_mask = AttentionMask(config)
        # Word embedding
        self.word_embedding = EmbeddingLookup(config).set_comm_fusion(1)
        if config.load_ckpt_path:
            # Loading the embedding table from the ckpt path:
            embedding_path = os.path.join(config.load_ckpt_path, 'position_embedding.npy')
            if os.path.exists(embedding_path):
                p_table = np.load(embedding_path)
                position_table_param = Tensor(p_table, mstype.float32)
            else:
                raise ValueError(f"{embedding_path} file not exits, please check whether position_embedding file exit.")
        else:
            position_table_param = TruncatedNormal(0.02)

        # Position embedding
        self.position_embedding = nn.Embedding(
            config.seq_length,
            config.embedding_size,
            embedding_table=position_table_param).set_comm_fusion(1)
        self.word_embedding.embedding_table.parallel_optimizer = False
        self.position_embedding.embedding_table.parallel_optimizer = False
        self.position_embedding.gather.shard(((1, 1), (config.dp,)))
        self.position_embedding.expand.shard(((config.dp, 1),))

        self.get_attention_mask.pipeline_stage = 0
        self.word_embedding.pipeline_stage = 0
        self.position_embedding.pipeline_stage = 0

        self.blocks = nn.CellList()
        # Total fusion groups for HCCL operators. Specifically, the same tyep HCCL operators in same group will be fused.
        fusion_group_num = 4
        fusion_group_size = config.num_layers // fusion_group_num
        fusion_group_size = max(fusion_group_size, 1)

        num_layers = config.num_layers
        # If top_query_attention enabled, replace the last normal self-attention layers with this top_query_attention layer
        if config.use_top_query_attention:
            num_layers -= 1
        self.num_layers = num_layers
        print("After setting the layer is:", num_layers, flush=True)

        for i in range(num_layers):
            per_block = Block(config, i + 1).set_comm_fusion(int(i / fusion_group_size) + 2)
            # Each layer will be remoputed in the backward process. The output activation of each layer will be saved,
            # in other words, in backward process each block will be almosttotally recomputed.
            if config.use_recompute:
                per_block.recompute()
                # Dropout will not be recomputed to ensure the consistency between forward and the corresponding backward.
                per_block.attention.dropout.dropout.recompute(False)
                per_block.attention.prob_dropout.dropout.recompute(False)
                per_block.output.dropout.dropout.recompute(False)
            if config.param_init_type == mstype.float16:
                # If the model is initialized with fp16, the fusion of layernorm (fp32 gradient) will mix up with
                # the bias parameter in linear models (fp16 gradient), causing dtype error for communication operators.
                # so we fuse communications of layernorm to a large value(+100)
                per_block.layernorm1.set_comm_fusion(int(int(i / fusion_group_size) + 100))
                per_block.layernorm2.set_comm_fusion(int(int(i / fusion_group_size) + 100))
            if i < num_layers // 2:
                per_block.pipeline_stage = 0
            else:
                per_block.pipeline_stage = 1
            self.blocks.append(per_block)
        if config.self_layernorm:
            self.layernorm = LayerNorm((config.embedding_size,), config.dp).to_float(
                mstype.float32).set_comm_fusion(int((num_layers - 1) / fusion_group_size) + 2)
        else:
            self.layernorm = nn.LayerNorm((config.embedding_size,)).to_float(
                mstype.float32).set_comm_fusion(int((num_layers - 1) / fusion_group_size) + 2)
            self.layernorm.layer_norm.shard(((config.dp, 1, 1), (1,), (1,)))
        self.layernorm.gamma.parallel_optimizer = False
        self.layernorm.beta.parallel_optimizer = False
        self.layernorm.pipeline_stage = 1

        if config.param_init_type == mstype.float16:
            # If the model is initialized with fp16, the fusion of layernorm (fp32 gradient) will mix up with
            # the bias parameter in linear models (fp16 gradient), causing dtype error for communication operators.
            # so we fuse communications of layernorm to a large value(+100)
            self.layernorm.set_comm_fusion(int(num_layers / fusion_group_size + 100))
        self.use_past = config.use_past
        self.past = tuple([None] * config.num_layers)
        self.add = P.TensorAdd().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.expand_dims = P.ExpandDims().shard(((config.dp, 1, 1),))
        self.dtype = config.compute_dtype
        self.dropout = Dropout(1 - config.dropout_rate)
        self.dropout.dropout.shard(((config.dp, 1, 1),))
        self.eod_reset = config.eod_reset
        # If top_query_attention enabled, the input_position representing the position ids will be used as the index
        # for a query embedding table to obtain top query hidden states, together with the previous outputs of normal
        # self-attention layers, a new attention layer will be attached to the output of the model
        if config.use_top_query_attention:
            if config.load_ckpt_path:
                # Loading the embedding table from the ckpt path:
                embedding_path = os.path.join(config.load_ckpt_path, 'top_query_embedding.npy')
                if os.path.exists(embedding_path):
                    top_query_table = np.load(embedding_path)
                    top_query_table_param = Tensor(top_query_table, mstype.float32)
                else:
                    raise ValueError(
                        f"{embedding_path} file not exits, please check whether top_query_embedding file exist.")
            else:
                top_query_table_param = TruncatedNormal(0.02)

            self.top_query_embedding = nn.Embedding(config.seq_length, config.embedding_size,
                                                    embedding_table=top_query_table_param)
            # If the model is initialized with fp16, the fusion of layernorm (fp32 gradient) will mix up with
            # the bias parameter in linear models (fp16 gradient), causing dtype error for communication operators.
            # so we fuse communications of embedding to a large value(+100)
            self.top_query_embedding.set_comm_fusion(int((config.num_layers - 1) / fusion_group_num) + 200)
            self.top_query_embedding.embedding_table.parallel_optimizer = False
            self.top_query_embedding.gather.shard(((1, 1), (config.dp,)))
            self.top_query_embedding.expand.shard(((config.dp, 1),))
            self.top_query_layer = QueryLayer(config)
            self.top_query_embedding.pipeline_stage = 1
            self.top_query_layer.pipeline_stage = 1
            if config.use_recompute:
                self.top_query_layer.recompute()
                self.top_query_layer.output.dropout.dropout.recompute(False)
                self.top_query_layer.attention.dropout.dropout.recompute(False)
                self.top_query_layer.attention.prob_dropout.dropout.recompute(False)

            self.top_query_layer.set_comm_fusion(int((config.num_layers - 1) / fusion_group_num) + 2)
            self.top_query_layer.layernorm1.set_comm_fusion(int(config.num_layers / fusion_group_size + 100))
            self.top_query_layer.layernorm2.set_comm_fusion(int(config.num_layers / fusion_group_size + 100))

        self.use_top_query_attention = config.use_top_query_attention

    def construct(self, input_ids, input_mask, input_position=None, attention_mask=None, layer_past=None):
        """PanguAlpha model"""
        if not self.use_past:
            layer_past = self.past

        # Word embedding
        input_embedding, embedding_table = self.word_embedding(input_ids)
        # If eod_reset disabled, there will be only one input from the dataset, i.e., input_ids
        # and the corresponding input_position and attention_mask will be derived from it.
        if not self.eod_reset:
            batch_size, seq_length = F.shape(input_ids)
            input_position = F.tuple_to_array(F.make_range(seq_length))
            input_position = P.Tile()(input_position, (batch_size, 1))
            attention_mask = self.get_attention_mask(input_mask)
        position_embedding = self.position_embedding(input_position)
        # Input features [bs, seq_length, embedding_size]
        hidden_states = self.add(input_embedding, position_embedding)
        hidden_states = self.dropout(hidden_states)
        hidden_states = P.Cast()(hidden_states, mstype.float16)
        attention_mask = self.expand_dims(attention_mask, 1)

        present_layer = ()
        # Loop through each self-attention layer
        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](hidden_states,
                                                    attention_mask, layer_past)
            present_layer = present_layer + (present,)

        output_state = self.layernorm(hidden_states)
        output_state = F.cast(output_state, self.dtype)

        # Top query attention layer
        if self.use_top_query_attention:
            top_query_hidden_states = self.top_query_embedding(input_position)
            output_state, present = self.top_query_layer(output_state, top_query_hidden_states,
                                                         attention_mask, layer_past)
            present_layer = present_layer + (present,)
        return output_state, present_layer, embedding_table


class PanguAlpha_Head(nn.Cell):
    """
    Head for PanguAlpha to get the logits of each token in the vocab
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self, config):
        super(PanguAlpha_Head, self).__init__()
        if config.word_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((config.dp, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((config.dp, 1), (config.mp, 1)))
        self.embedding_size = config.embedding_size
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.compute_dtype
        self.cast = P.Cast()
        self.reshape = P.Reshape()

    def construct(self, state, embedding_table):
        state = self.reshape(state, (-1, self.embedding_size))
        # output logits over vocabulary [bs*seq_length, vocab_size]
        logits = self.matmul(state, self.cast(embedding_table, self.dtype))
        return logits


class PanguAlpha(nn.Cell):
    """
    The PanguAlpha network consisting of two parts the backbone and the head
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map
    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """

    def __init__(self, config):
        super(PanguAlpha, self).__init__()
        # Network backbone of PanguAlpha
        self.backbone = PanguAlpha_Model(config)
        # Network head to get logits over vocabulary
        self.head = PanguAlpha_Head(config)
        self.head.pipeline_stage = 1

    def construct(self, input_ids, input_mask, input_position=None, attention_mask=None, past=None):
        output_states, _, embedding_table = self.backbone(
            input_ids, input_mask, input_position, attention_mask, past)
        logits = self.head(output_states, embedding_table)
        return logits


class CrossEntropyLoss(nn.Cell):
    """
    Calculate the cross entropy loss
    Args:
        config(PanguAlphaConfig): the config of the network
    Inputs:
        logits: the output logits of the backbone
        label: the ground truth label of the sample
        input_mask: the mask indicating whether each position is a valid input
    Returns:
        loss: Tensor, the corrsponding cross entropy loss
    """

    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        self.mean = P.ReduceMean()
        self.sum = P.ReduceSum().shard(((config.dp, 1),))
        self.onehot = P.OneHot().shard(((config.dp, 1), (), ()))
        # on/off value for onehot, for smooth labeling, modify the off_value
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.vocab_size = config.vocab_size
        self.max = P.ArgMaxWithValue(axis=-1, keep_dims=True).shard(((config.dp, 1),))
        self.eps_const = Tensor(1e-24, mstype.float32)
        self.sub = P.Sub().shard(((config.dp, 1), (config.dp, 1)))
        self.exp = P.Exp().shard(((config.dp, 1),))
        self.div = P.RealDiv().shard(((config.dp, 1), (config.dp, 1)))
        self.log = P.Log().shard(((config.dp, 1),))
        self.add = P.TensorAdd().shard(((config.dp, 1), ()))
        self.mul = P.Mul().shard(((config.dp, 1), (config.dp, 1)))
        self.neg = P.Neg().shard(((config.dp, 1),))
        self.sum2 = P.ReduceSum().shard(((config.dp,),))  # dynamic, avoid all-gather

        self.mul2 = P.Mul().shard(((config.dp,), (config.dp,)))  # dynamic, avoid all-gather
        self.add2 = P.TensorAdd()
        self.div2 = P.RealDiv()
        self.reshape = P.Reshape()

    def construct(self, logits, label, input_mask):
        r"""
        Compute loss using logits, label and input mask
        """
        # [bs*seq_length, vocab_size]
        logits = F.cast(logits, mstype.float32)
        # LogSoftmax for logits over last dimension
        _, logit_max = self.max(logits)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1)
        exp_sum = self.reshape(exp_sum, (F.shape(exp_sum)[0], 1))
        softmax_result = self.div(logit_exp, exp_sum)
        log_softmax_result = self.log(self.add(softmax_result, self.eps_const))

        # Flatten label to [bs*seq_length]
        label = self.reshape(label, (-1,))
        # Get onehot label [bs*seq_length, vocab_size]
        one_hot_label = self.onehot(label, self.vocab_size, self.on_value,
                                    self.off_value)
        # Cross-Entropy loss
        loss = self.mul(log_softmax_result, one_hot_label)
        loss_unsum = self.neg(loss)
        loss_reduce = self.sum(loss_unsum, -1)
        # input_mask indicates whether there is padded inputs and for padded inputs it will not be counted into loss
        input_mask = self.reshape(input_mask, (-1,))
        numerator = self.sum2(self.mul2(loss_reduce, input_mask))

        denominator = self.add2(
            self.sum2(input_mask),
            P.Cast()(F.tuple_to_array((1e-5,)), mstype.float32))
        loss = self.div2(numerator, denominator)
        return loss


class PanguAlphaWithLoss(nn.Cell):
    """
    PanguAlpha training loss
    Args:
        network: backbone network of PanguAlpha
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss, eos_token=6):
        super(PanguAlphaWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        # id for end_of_sentence, 6 in the vocabulary
        self.eos_token = eos_token
        self.slice = P.StridedSlice().shard(((config.dp, 1),))
        self.not_equal = P.NotEqual().shard(((config.dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.eod_reset = config.eod_reset
        if self.eod_reset:
            self.slice_mask = P.StridedSlice().shard(((config.dp, 1, 1),))

    def construct(self, input_ids, input_position=None, attention_mask=None):
        r"""
        PanguAlphaWithLoss
        """
        # input_ids [bs, seq_length+1]
        # input_position [bs, seq_length] only available when eod_reset enabled
        # attention_mask [bs, seq_length, seq_length] only available when eod-reset enabled
        # Get input tokens [bs, seq_length]
        # dynamic use shape op to construct end for stridedslice
        bs = F.shape(input_ids)[0]
        seq = F.shape(input_ids)[1]
        tokens = self.slice(input_ids, (0, 0), (bs, seq - 1), (1, 1))

        if self.eod_reset:
            input_position = self.slice(input_position, (0, 0), (bs, seq - 1), (1, 1))
            attention_mask = self.slice_mask(attention_mask, (0, 0, 0),
                                             (bs, seq - 1, seq - 1),
                                             (1, 1, 1))
        # Check whether there is padding in inputs
        input_mask = F.cast(self.not_equal(tokens, self.eos_token), mstype.float32)
        logits = self.network(tokens, input_mask, input_position, attention_mask)
        # Get label corresponding to input tokens
        labels = self.slice(input_ids, (0, 1), (bs, seq), (1, 1))
        # Loss
        output = self.loss(logits, labels, input_mask)
        return output


class PANGUALPHAConfig:
    """
    PANGUALPHA config class which defines the model size
    """

    def __init__(self,
                 data_parallel_num,
                 model_parallel_num,
                 pipeline_parallel_num=1,
                 batch_size=32,
                 seq_length=1024,
                 vocab_size=51200,
                 embedding_size=768,
                 num_layers=1,
                 num_heads=12,
                 expand_ratio=4,
                 post_layernorm_residual=False,
                 dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 use_past=False,
                 self_layernorm=True,
                 word_emb_dp=True,
                 stage_num=16,
                 eod_reset=True,
                 micro_size=32,
                 load_ckpt_path=None,
                 use_top_query_attention=True,
                 param_init_type=mstype.float32,
                 use_recompute=True):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        # The expand ratio of feature size in FFN
        self.expand_ratio = expand_ratio
        # Use post-layernorm or pre-layernrom, default:pre-layernorm
        self.post_layernorm_residual = post_layernorm_residual
        self.dropout_rate = dropout_rate
        self.compute_dtype = compute_dtype
        # Whether use incremental inference
        self.use_past = use_past
        self.dp = data_parallel_num
        self.mp = model_parallel_num
        # Whether use self implemented layernorm
        self.self_layernorm = self_layernorm
        self.stage_num = stage_num
        self.micro_size = micro_size
        self.word_emb_dp = word_emb_dp
        self.eod_reset = eod_reset
        # Used for loading embedding tables
        self.load_ckpt_path = load_ckpt_path
        self.use_top_query_attention = use_top_query_attention
        self.use_recompute = use_recompute
        self.param_init_type = param_init_type
        if pipeline_parallel_num > 1:
            context.set_auto_parallel_context(pipeline_stages=pipeline_parallel_num)


def compile_net(net, _x1, _x2, _x3):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net.set_inputs(_x1, _x2, _x3)
    phase, _ = _cell_graph_executor.compile(train_net, _x1, _x2, _x3)
    context.reset_auto_parallel_context()
    return phase


def test_pangu_alpha_batch_dim_dynamic_and_data_parallel():
    '''
    Feature: batch dim is dynamic and using data parallel
    Description: all reshape skip redistribution
    Expectation: compile success
    '''
    # FIXME: test_pangu_alpha_batch_dim_dynamic_and_dp_mp_op pass, but this failed.
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=False)
    config = PANGUALPHAConfig(data_parallel_num=8, model_parallel_num=1, num_layers=2)
    pangu_alpha = PanguAlpha(config)
    loss = CrossEntropyLoss(config)
    pangu_alpha_loss = PanguAlphaWithLoss(config, pangu_alpha, loss)
    net = _VirtualDatasetCell(pangu_alpha_loss)
    input_ids = Tensor(shape=[None, 1025], dtype=mstype.int32)
    input_position = Tensor(shape=[None, 1024], dtype=mstype.int32)
    attention_mask = Tensor(shape=[None, 1024, 1024], dtype=mstype.float16)
    compile_net(net, input_ids, input_position, attention_mask)


def test_pangu_alpha_batch_dim_dynamic_and_dp_mp():
    '''
    Feature: batch dim is dynamic and using data parallel and mode parallel
    Description: all reshape skip redistribution
    Expectation: compile success
    '''
    context.reset_auto_parallel_context()
    ds_strategy = ((2, 1), (2, 1), (2, 1, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy=ds_strategy)
    config = PANGUALPHAConfig(data_parallel_num=2, model_parallel_num=4, num_layers=2)
    pangu_alpha = PanguAlpha(config)
    loss = CrossEntropyLoss(config)
    pangu_alpha_loss = PanguAlphaWithLoss(config, pangu_alpha, loss)
    net = _VirtualDatasetCell(pangu_alpha_loss)
    net._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[None, 1025], dtype=mstype.int32)
    input_position = Tensor(shape=[None, 1024], dtype=mstype.int32)
    attention_mask = Tensor(shape=[None, 1024, 1024], dtype=mstype.float16)
    compile_net(net, input_ids, input_position, attention_mask)


def test_pangu_alpha_batch_dim_dynamic_and_dp_mp_op():
    '''
    Feature: batch dim is dynamic and using data parallel and mode parallel and optimizer parallel
    Description: all reshape skip redistribution
    Expectation: compile success
    '''
    context.reset_auto_parallel_context()
    ds_strategy = ((2, 1), (2, 1), (2, 1, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy=ds_strategy,
                                      enable_parallel_optimizer=True)
    config = PANGUALPHAConfig(data_parallel_num=2, model_parallel_num=4, num_layers=2)
    pangu_alpha = PanguAlpha(config)
    loss = CrossEntropyLoss(config)
    pangu_alpha_loss = PanguAlphaWithLoss(config, pangu_alpha, loss)
    net = _VirtualDatasetCell(pangu_alpha_loss)
    net._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[None, 1025], dtype=mstype.int32)
    input_position = Tensor(shape=[None, 1024], dtype=mstype.int32)
    attention_mask = Tensor(shape=[None, 1024, 1024], dtype=mstype.float16)
    compile_net(net, input_ids, input_position, attention_mask)


def test_pangu_alpha_seq_dim_dynamic_and_dp_mp_op():
    '''
    Feature: seq dim is dynamic and using data parallel and mode parallel and optimizer parallel
    Description: all reshape skip redistribution
    Expectation: compile success
    '''
    context.reset_auto_parallel_context()
    ds_strategy = ((2, 1), (2, 1), (2, 1, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy=ds_strategy,
                                      enable_parallel_optimizer=True)
    config = PANGUALPHAConfig(data_parallel_num=2, model_parallel_num=4, num_layers=1)
    pangu_alpha = PanguAlpha(config)
    loss = CrossEntropyLoss(config)
    pangu_alpha_loss = PanguAlphaWithLoss(config, pangu_alpha, loss)
    net = _VirtualDatasetCell(pangu_alpha_loss)
    net._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[32, None], dtype=mstype.int32)
    input_position = Tensor(shape=[32, None], dtype=mstype.int32)
    attention_mask = Tensor(shape=[32, None, None], dtype=mstype.float16)
    compile_net(net, input_ids, input_position, attention_mask)


def test_pangu_alpha_batch_and_seq_dims_dynamic_and_dp_mp_op():
    '''
    Feature: batch and seq dims are dynamic and using data parallel and mode parallel and optimizer parallel
    Description: all reshape skip redistribution
    Expectation: compile success
    '''
    context.set_context(save_graphs=True, save_graphs_path="./dump_ir_without_skip")
    context.reset_auto_parallel_context()
    ds_strategy = ((2, 1), (2, 1), (2, 1, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy=ds_strategy,
                                      enable_parallel_optimizer=True)
    config = PANGUALPHAConfig(data_parallel_num=2, model_parallel_num=4, num_layers=2)
    pangu_alpha = PanguAlpha(config)
    loss = CrossEntropyLoss(config)
    pangu_alpha_loss = PanguAlphaWithLoss(config, pangu_alpha, loss)
    net = _VirtualDatasetCell(pangu_alpha_loss)
    net._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
    input_position = Tensor(shape=[None, None], dtype=mstype.int32)
    attention_mask = Tensor(shape=[None, None, None], dtype=mstype.float16)
    compile_net(net, input_ids, input_position, attention_mask)


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in [0, 1]:
        return grad
    dt = F.dtype(grad)
    # 0 for clip_by_value and 1 for clip_by_norm
    if clip_type == 0:
        new_grad = C.clip_by_value(
            grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
            F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad,
                                   F.cast(F.tuple_to_array((clip_value,)),
                                          dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")
shard_grad_scale = C.MultitypeFuncGraph("shard_grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


@grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_grad_scale_pipeline(scale, grad, accu_grad):
    accu_grad = F.depend(accu_grad, grad)
    new_grad = accu_grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    zeros = F.tensor_mul(accu_grad, 0.0)
    new_grad = F.depend(new_grad, F.assign(accu_grad, zeros))
    return new_grad


@shard_grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_shard_grad_scale_pipeline(scale, grad, accu_grad):
    new_grad = grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    new_grad = F.depend(new_grad, F.assign(accu_grad, F.zeros_like(accu_grad)))
    return new_grad


class PanguAlphaTrainOneStepWithLossScaleCell(TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of PanguAlpha network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self,
                 network,
                 optimizer,
                 scale_update_cell=None,
                 enable_global_norm=False,
                 config=None):
        super(PanguAlphaTrainOneStepWithLossScaleCell,
              self).__init__(network, optimizer, scale_update_cell)
        self.network = network
        self.config = config
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.default_lr = Tensor([0.0], dtype=mstype.float32)
        self.enable_global_norm = enable_global_norm
        self.clip = ClipByGlobalNorm(self.weights, config)
        self.cast = P.Cast()

    def construct(self, input_ids, input_position, attention_mask, layer_past=None, sens=None):
        """Defines the computation performed."""
        weights = self.weights
        # Forward process
        loss = self.network(input_ids, input_position, attention_mask)
        scaling_sens = self.scale_sense

        # alloc status and clear should be right before gradoperation
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        # Backward process using loss scale
        grads = self.grad(self.network,
                          weights)(input_ids,
                                   input_position, attention_mask,
                                   scaling_sens_filled)

        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(
            F.partial(grad_scale, scaling_sens), grads)

        if self.enable_global_norm:
            grads, _ = self.clip(grads)
        else:
            grads = self.hyper_map(
                F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE),
                grads)
        # Check whether overflow
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # If overflow, surpass weights update
        # if not, update weights
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        return F.depend(loss, succ), cond, scaling_sens


class PanguAlphaTrainPipelineWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of PanguAlpha network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None, enable_global_norm=True):
        super(PanguAlphaTrainPipelineWithLossScaleCell, self).__init__(auto_prefix=False)
        # self.config = config
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.optimizer = optimizer
        self.enable_global_norm = enable_global_norm
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus().add_prim_attr("_side_effect_flag", False)
        self.get_status = P.NPUGetFloatStatus().add_prim_attr("_side_effect_flag", False)
        self.clear_before_grad = P.NPUClearFloatStatus().add_prim_attr("_side_effect_flag", False)
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.reshape = P.Reshape()
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")
        # self.clip = ClipByGlobalNorm(self.weights, self.config)
        self.micro_size = 2
        self.opt_shard = _get_enable_parallel_optimizer()

    @C.add_flags(has_effect=True)
    def construct(self,
                  input_ids,
                  input_position,
                  attention_mask,
                  past=None,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids, input_position, attention_mask)
        if sens is None:
            scaling_sens = self.loss_scale
            scaling_sens = self.reshape(scaling_sens, (1,))
        else:
            scaling_sens = sens
        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        status_clear = self.clear_before_grad(init)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_position,
                                                 attention_mask,
                                                 self.cast(scaling_sens / self.micro_size,
                                                           mstype.float32))
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        loss = F.depend(loss, status_clear)
        # apply grad reducer on grads
        if self.opt_shard:
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(F.partial(shard_grad_scale, scaling_sens * self.degree), grads, self.accu_grads)
        else:
            accu_grads = self.grad_reducer(self.accu_grads)
            grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads, accu_grads)
        # if self.enable_global_norm:
        #     grads, _ = self.clip(grads)
        # else:
        grads = self.hyper_map(
            F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE),
            grads)
        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, overflow, scaling_sens)
        return F.depend(ret, succ)


def compile_pipeline_net(net, _x1, _x2, _x3):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_scale_value = math.pow(2, 32)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)
    train_net = PanguAlphaTrainPipelineWithLossScaleCell(net, optimizer, scale_update_cell=update_cell)
    train_net.set_train()
    train_net.set_inputs(_x1, _x2, _x3)
    phase, _ = _cell_graph_executor.compile(train_net, _x1, _x2, _x3)
    context.reset_auto_parallel_context()
    return phase


def test_pipeline_dp_mp_op_bs_and_seq_dynamic_stage0():
    '''
    Feature: batch dim and seq dim are dynamic, and using pp + dp + mp + op, test stage-0
    Description: all reshape skip redistribution, pipeline slice micro skip redistribution, and set virtual dataset
    Expectation: compile success
    '''
    context.reset_auto_parallel_context()
    os.environ['PIPELINE_SLICE_SKIP_REDISTRIBUTION'] = "1"
    ds_strategy = ((2, 1), (2, 1), (2, 1, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy=ds_strategy,
                                      enable_parallel_optimizer=True, global_rank=0)
    config = PANGUALPHAConfig(data_parallel_num=2, model_parallel_num=2, pipeline_parallel_num=2, num_layers=3)
    pangu_alpha = PanguAlpha(config)
    loss = CrossEntropyLoss(config)
    loss.pipeline_stage = 1
    pangu_alpha_loss = PanguAlphaWithLoss(config, pangu_alpha, loss)
    pangu_alpha_loss = PipelineCell(pangu_alpha_loss, 2)
    for i in range(pangu_alpha_loss.micro_size):
        pangu_alpha_loss.micro_inputs[i].strided_slice.add_prim_attr("out_shard_size", 2)
    net = _VirtualDatasetCell(pangu_alpha_loss)
    net._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")

    input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
    input_position = Tensor(shape=[None, None], dtype=mstype.int32)
    attention_mask = Tensor(shape=[None, None, None], dtype=mstype.float16)

    compile_pipeline_net(net, input_ids, input_position, attention_mask)
    del os.environ['PIPELINE_SLICE_SKIP_REDISTRIBUTION']


def test_pipeline_dp_mp_op_bs_and_seq_dynamic_stage1():
    '''
    Feature: batch dim and seq dim are dynamic, and using pp + dp + mp + op, test stage-1
    Description: all reshape skip redistribution, pipeline slice micro skip redistribution, and set virtual dataset
    Expectation: compile success
    '''
    context.reset_auto_parallel_context()
    os.environ['PIPELINE_SLICE_SKIP_REDISTRIBUTION'] = "1"
    ds_strategy = ((2, 1), (2, 1), (2, 1, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy=ds_strategy,
                                      enable_parallel_optimizer=True, global_rank=4)
    config = PANGUALPHAConfig(data_parallel_num=2, model_parallel_num=2, pipeline_parallel_num=2, num_layers=3)
    pangu_alpha = PanguAlpha(config)
    loss = CrossEntropyLoss(config)
    loss.pipeline_stage = 1
    pangu_alpha_loss = PanguAlphaWithLoss(config, pangu_alpha, loss)
    pangu_alpha_loss = PipelineCell(pangu_alpha_loss, 2)
    for i in range(pangu_alpha_loss.micro_size):
        pangu_alpha_loss.micro_inputs[i].strided_slice.add_prim_attr("out_shard_size", 2)
    net = _VirtualDatasetCell(pangu_alpha_loss)
    net._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")

    input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
    input_position = Tensor(shape=[None, None], dtype=mstype.int32)
    attention_mask = Tensor(shape=[None, None, None], dtype=mstype.float16)

    compile_pipeline_net(net, input_ids, input_position, attention_mask)
    del os.environ['PIPELINE_SLICE_SKIP_REDISTRIBUTION']


class PanguAlphaWithLossLazyInline(PanguAlphaWithLoss):
    """
    cell reuse
    """

    @lazy_inline
    def __init__(self, config, network, loss, eos_token=6):
        super(PanguAlphaWithLossLazyInline, self).__init__(config, network, loss, eos_token)


def test_pipeline_dp_mp_op_bs_and_seq_dynamic_cell_reuse_stage0():
    '''
    Feature: batch dim and seq dim are dynamic, and using pp + dp + mp + op, cell_reuse, test stage-0
    Description: all reshape skip redistribution, pipeline slice micro skip redistribution, and set virtual dataset
    Expectation: compile success
    '''
    context.reset_auto_parallel_context()
    os.environ['PIPELINE_SLICE_SKIP_REDISTRIBUTION'] = "1"
    ds_strategy = ((2, 1), (2, 1), (2, 1, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy=ds_strategy,
                                      enable_parallel_optimizer=True, global_rank=0)
    config = PANGUALPHAConfig(data_parallel_num=2, model_parallel_num=2, pipeline_parallel_num=2, num_layers=3)
    pangu_alpha = PanguAlpha(config)
    loss = CrossEntropyLoss(config)
    loss.pipeline_stage = 1
    pangu_alpha_loss = PanguAlphaWithLossLazyInline(config, pangu_alpha, loss)
    pangu_alpha_loss = PipelineCell(pangu_alpha_loss, 2)
    for i in range(pangu_alpha_loss.micro_size):
        pangu_alpha_loss.micro_inputs[i].strided_slice.add_prim_attr("out_shard_size", 2)
    net = _VirtualDatasetCell(pangu_alpha_loss)
    net._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")

    input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
    input_position = Tensor(shape=[None, None], dtype=mstype.int32)
    attention_mask = Tensor(shape=[None, None, None], dtype=mstype.float16)

    compile_pipeline_net(net, input_ids, input_position, attention_mask)
    del os.environ['PIPELINE_SLICE_SKIP_REDISTRIBUTION']


def test_pipeline_dp_mp_op_bs_and_seq_dynamic_cell_reuse_stage1():
    '''
    Feature: batch dim and seq dim are dynamic, and using pp + dp + mp + op, cell_reuse, test stage-1
    Description: all reshape skip redistribution, pipeline slice micro skip redistribution, and set virtual dataset
    Expectation: compile success
    '''
    context.reset_auto_parallel_context()
    os.environ['PIPELINE_SLICE_SKIP_REDISTRIBUTION'] = "1"
    ds_strategy = ((2, 1), (2, 1), (2, 1, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy=ds_strategy,
                                      enable_parallel_optimizer=True, global_rank=4)
    config = PANGUALPHAConfig(data_parallel_num=2, model_parallel_num=2, pipeline_parallel_num=2, num_layers=3)
    pangu_alpha = PanguAlpha(config)
    loss = CrossEntropyLoss(config)
    loss.pipeline_stage = 1
    pangu_alpha_loss = PanguAlphaWithLossLazyInline(config, pangu_alpha, loss)
    pangu_alpha_loss = PipelineCell(pangu_alpha_loss, 2)
    for i in range(pangu_alpha_loss.micro_size):
        pangu_alpha_loss.micro_inputs[i].strided_slice.add_prim_attr("out_shard_size", 2)
    net = _VirtualDatasetCell(pangu_alpha_loss)
    net._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")

    input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
    input_position = Tensor(shape=[None, None], dtype=mstype.int32)
    attention_mask = Tensor(shape=[None, None, None], dtype=mstype.float16)

    compile_pipeline_net(net, input_ids, input_position, attention_mask)
    del os.environ['PIPELINE_SLICE_SKIP_REDISTRIBUTION']
