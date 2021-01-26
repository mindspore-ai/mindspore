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

"""GPT model"""

import math
import numpy as np
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
from mindspore.common.initializer import TruncatedNormal, initializer, Normal
from mindspore.ops import operations as P
from mindspore.ops import functional as F


class LayerNorm(nn.Cell):
    """
    Layer Normalization

    Args:
        normalized_shape: the corresponding shape of the normalized axes
        eps: epsilon, a small number avoiding zero division

    Inputs:
        x: input tensor

    Returns:
        rescaled_output: Tensor, returned tensor after layernorm
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(initializer('ones', normalized_shape))
        self.beta = Parameter(initializer('zeros', normalized_shape))
        self.mean = P.ReduceMean(keep_dims=True)
        self.eps = eps

    def construct(self, x):
        mean = self.mean(x, -1)
        variance = self.mean(F.square(x - mean), -1)
        output = (x - mean) / F.sqrt(variance + self.eps)
        rescaled_output = output * self.gamma + self.beta
        return rescaled_output

class Softmax(nn.Cell):
    """
    softmax realization

    Args:
        axis: the axis to be applied softmax

    Inputs:
        x: input tensor

    Returns:
        output: Tensor, returned tensor after softmax
    """
    def __init__(self, axis=-1):
        super(Softmax, self).__init__()
        self.max = P.ArgMaxWithValue(axis=axis, keep_dims=True)
        self.sum = P.ReduceSum(keep_dims=True)
        self.axis = axis

    def construct(self, x):
        _, max_value = self.max(x)
        exp_x = F.tensor_pow(np.e, x - max_value)
        sum_x = self.sum(exp_x, self.axis)
        output = exp_x / sum_x
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
    def __init__(self, input_size, output_size, dtype, scale=1.0):
        super(Mapping, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.weight = Parameter(initializer(Normal(sigma=0.02*scale), [input_size, output_size]))
        self.bias = Parameter(initializer("zeros", [output_size,]))
        self.dtype = dtype
        self.cast = P.Cast()

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.output_size,)
        x = P.Reshape()(x, (-1, self.input_size))
        x = nn.MatMul()(x, self.cast(self.weight, self.dtype)) + self.cast(self.bias, self.dtype)
        output = P.Reshape()(x, out_shape)
        return output



class Output(nn.Cell):
    """
    The output mapping module for each layer

    Args:
        config(GPTConfig): the config of network
        scale: scale factor for initialization

    Inputs:
        x: output of the self-attention module

    Returns:
        output: Tensor, the output of this layer after mapping
    """
    def __init__(self, config, scale=1.0):
        super(Output, self).__init__()
        input_size = config.embedding_size
        output_size = config.embedding_size*config.expand_ratio
        self.mapping = Mapping(input_size, output_size, config.compute_dtype)
        self.projection = Mapping(output_size, input_size, config.compute_dtype, scale)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(1-config.dropout_rate)

    def construct(self, x):
        hidden = self.activation(self.mapping(x))
        output = self.projection(hidden)
        output = self.dropout(output)
        return output

class AttentionMask(nn.Cell):
    """
    Get the attention matrix for self-attention module

    Args:
        config(GPTConfig): the config of network

    Inputs:
        input_mask: the mask indicating whether each position is a valid input

    Returns:
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
    """
    def __init__(self, config):
        super(AttentionMask, self).__init__()
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul()
        ones = np.ones(shape=(config.seq_length, config.seq_length))
        self.lower_triangle_mask = Tensor(np.tril(ones), mstype.float32)
        self.multiply = P.Mul()


    def construct(self, input_mask):
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        lower_traiangle = P.ExpandDims()(self.lower_triangle_mask, 0)
        attention_mask = self.multiply(attention_mask, lower_traiangle)  #bs seq_length seq_length
        return attention_mask

class EmbeddingLookup(nn.Cell):
    """
    The embedding lookup table for vocabulary

    Args:
        config(GPTConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the embedding vector for the input with shape (batch_size, seq_length, embedding_size)
        self.embedding_table: Tensor, the embedding table for the vocabulary
    """
    def __init__(self, config):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.embedding_table = Parameter(initializer(TruncatedNormal(0.02), [self.vocab_size, self.embedding_size]))
        self.gather = P.Gather()
        self.shape = (-1, config.seq_length, config.embedding_size)
    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table


class Attention(nn.Cell):
    """
    Self-Attention module for each layer

    Args:
        config(GPTConfig): the config of network
        scale: scale factor for initialization
        layer_idx: current layer index
    """
    def __init__(self, config, scale=1.0, layer_idx=None):
        super(Attention, self).__init__()
        self.get_attention_mask = AttentionMask(config)
        self.projection = Mapping(config.embedding_size, config.embedding_size, config.compute_dtype, scale)
        self.split = P.Split(axis=-1, output_num=3)
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.n_head = config.num_heads
        self.size_per_head = config.embedding_size // self.n_head
        self.concat_k = P.Concat(axis=3)
        self.concat_v = P.Concat(axis=2)
        self.multiply_data = Tensor([-10000.0,], dtype=mstype.float32)
        self.batch_matmul = P.BatchMatMul()
        self.scale = scale
        if self.scale:
            self.scale_factor = Tensor(math.sqrt(self.size_per_head))
        if layer_idx is not None:
            self.coeff = math.sqrt(layer_idx * math.sqrt(self.size_per_head))
            self.coeff = Tensor(self.coeff)
        self.use_past = config.use_past
        self.dropout = nn.Dropout(1-config.dropout_rate)
        self.prob_dropout = nn.Dropout(1-config.dropout_rate)

        self.dense1 = nn.Dense(config.embedding_size, config.embedding_size).to_float(config.compute_dtype)
        self.dense2 = nn.Dense(config.embedding_size, config.embedding_size).to_float(config.compute_dtype)
        self.dense3 = nn.Dense(config.embedding_size, config.embedding_size).to_float(config.compute_dtype)

    def construct(self, x, attention_mask, layer_past=None):
        """
        self-attention

        Inputs:
            x: output of previous layer
            attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
            layer_past: the previous feature map

        Returns:
            output: Tensor, the output logit of this layer
            layer_present: Tensor, the feature map of current layer
        """

        original_shape = F.shape(x)
        x = F.reshape(x, (-1, original_shape[-1]))
        query = self.dense1(x)
        key = self.dense2(x)
        value = self.dense3(x)
        query = self.transpose(F.reshape(query, (-1, original_shape[1], self.n_head, self.size_per_head)), (0, 2, 1, 3))
        key = self.transpose(F.reshape(key, (-1, original_shape[1], self.n_head, self.size_per_head)), (0, 2, 3, 1))
        value = self.transpose(F.reshape(value, (-1, original_shape[1], self.n_head, self.size_per_head)), (0, 2, 1, 3))
        if self.use_past:
            past_value = layer_past[1]
            past_key = self.transpose(layer_past[0], (0, 1, 3, 2))
            key = self.concat_k((past_key, key))
            value = self.concat_v(past_value, value)
        layer_present = P.Stack()([self.transpose(key, (0, 1, 3, 2)), value])
        attention = self._attn(query, key, value, attention_mask)
        attention_merge = self.merge_heads(attention)
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
        x = self.transpose(x, (0, 2, 1, 3)) #bs, seq_length, head, size_per_head
        x_shape = P.Shape()(x)
        new_shape = x_shape[:-2] + (x_shape[-2]*x_shape[-1],)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)

        Returns:
            weighted_values: Tensor, the weighted sum scores
        """
        if not self.scale:
            query = query / F.cast(self.coeff, F.dtype(query))
            key = key / F.cast(self.coeff, F.dtype(key))

        score = self.batch_matmul(query, key)
        if self.scale:
            score = score / P.Cast()(self.scale_factor, P.DType()(score))

        ori_dtype = P.DType()(score)
        score = P.Cast()(score, mstype.float32)
        multiplu_out = P.Sub()(P.Cast()(F.tuple_to_array((1.0,)), P.DType()(score)),
                               P.Cast()(attention_mask, P.DType()(score)))

        adder = P.Mul()(multiplu_out, self.multiply_data)
        attention_scores = adder + score

        attention_scores = P.Cast()(attention_scores, ori_dtype)
        attention_probs = Softmax()(attention_scores)

        attention_probs = self.prob_dropout(attention_probs)
        weighted_values = self.batch_matmul(attention_probs, value)
        return weighted_values

class Block(nn.Cell):
    """
    The basic block of GPT network

    Args:
        config(GPTConfig): the config of network
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
        scale = 1.0
        self.layernorm1 = LayerNorm((config.embedding_size,)).to_float(config.compute_dtype)
        self.attention = Attention(config, scale, layer_idx)
        self.layernorm2 = LayerNorm((config.embedding_size,)).to_float(config.compute_dtype)
        self.output = Output(config, scale)
        self.post_layernorm_residual = config.post_layernorm_residual

    def construct(self, x, attention_mask, layer_past=None):
        """basic block of each layer"""
        input_x = self.layernorm1(x)
        attention, layer_present = self.attention(input_x, attention_mask, layer_past)
        if self.post_layernorm_residual:
            x = input_x + attention
        else:
            x = x + attention

        output_x = self.layernorm2(x)
        mlp_logit = self.output(output_x)
        if self.post_layernorm_residual:
            output = output_x + mlp_logit
        else:
            output = x + mlp_logit
        return output, layer_present

class GPT_Model(nn.Cell):
    """
    The backbone of GPT network

    Args:
        config(GPTConfig): the config of network

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
        super(GPT_Model, self).__init__()
        self.get_attention_mask = AttentionMask(config)
        self.word_embedding = EmbeddingLookup(config)
        self.position_embedding = nn.Embedding(config.seq_length, config.embedding_size,
                                               embedding_table=TruncatedNormal(0.02))
        self.blocks = nn.CellList()
        for i in range(config.num_layers):
            self.blocks.append(Block(config, i+1))
        self.layernorm = LayerNorm((config.embedding_size,)).to_float(config.compute_dtype)
        self.use_past = config.use_past
        self.past = tuple([None]*config.num_layers)
        self.num_layers = config.num_layers

    def construct(self, input_ids, input_mask, layer_past=None):
        """GPT model"""
        if not self.use_past:
            layer_past = self.past

        input_embedding, embedding_table = self.word_embedding(input_ids)

        batch_size, seq_length = F.shape(input_ids)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (batch_size, 1))


        position_embedding = self.position_embedding(input_position)
        hidden_states = input_embedding + position_embedding

        hidden_states = P.Cast()(hidden_states, mstype.float16)
        attention_mask = self.get_attention_mask(input_mask)
        attention_mask = P.ExpandDims()(attention_mask, 1)

        present_layer = ()
        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](hidden_states, attention_mask, layer_past)
            present_layer = present_layer + (present,)

        output_state = self.layernorm(hidden_states)
        return output_state, present_layer, embedding_table

class GPT_Head(nn.Cell):
    """
    Head for GPT to get the logits of each token in the vocab

    Args:
        config(GPTConfig): the config of network

    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary

    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """
    def __init__(self, config):
        super(GPT_Head, self).__init__()
        self.matmul = P.MatMul(transpose_b=True)
        self.embedding_size = config.embedding_size
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.compute_dtype
        self.cast = P.Cast()

    def construct(self, state, embedding_table):
        state = P.Reshape()(state, (-1, self.embedding_size))
        logits = self.matmul(state, self.cast(embedding_table, self.dtype))
        return logits

class GPT(nn.Cell):
    """
    The GPT network consisting of two parts the backbone and the head

    Args:
        config(GPTConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map

    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """
    def __init__(self, config):
        super(GPT, self).__init__()
        self.backbone = GPT_Model(config)
        self.head = GPT_Head(config)

    def construct(self, input_ids, input_mask, past=None):
        output_states, _, embedding_table = self.backbone(input_ids, input_mask, past)
        logits = self.head(output_states, embedding_table)
        return logits

class CrossEntropyLoss(nn.Cell):
    """
    Calculate the cross entropy loss

    Args:
        config(GPTConfig): the config of the network

    Inputs:
        logits: the output logits of the backbone
        label: the ground truth label of the sample
        input_mask: the mask indicating whether each position is a valid input

    Returns:
        loss: Tensor, the corrsponding cross entropy loss
    """
    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.mean = P.ReduceMean()
        self.sum = P.ReduceSum()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.vocab_size = config.vocab_size

    def construct(self, logits, label, input_mask):
        logits = self.log_softmax(P.Cast()(logits, mstype.float32))
        label = P.Reshape()(label, (-1,))
        one_hot_label = self.onehot(label, self.vocab_size, self.on_value, self.off_value)
        loss_sum = P.Neg()(self.sum(logits*one_hot_label, (-1,)))
        input_mask = P.Reshape()(input_mask, (-1,))
        numerator = self.sum(loss_sum*input_mask)
        denominator = self.sum(input_mask) + P.Cast()(F.tuple_to_array((1e-5,)), mstype.float32)
        loss = numerator / denominator
        return loss

class GPTWithLoss(nn.Cell):
    """
    GPT training loss

    Args:
        network: backbone network of GPT2/3
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token

    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map

    Returns:
        output: Tensor, the loss of the network
    """
    def __init__(self, network, loss, eos_token=50256):
        super(GPTWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token

    def construct(self, input_ids, past=None):
        tokens = input_ids[:, :-1]
        input_mask = F.cast(F.not_equal(tokens, self.eos_token), mstype.float32)
        logits = self.network(tokens, input_mask, past)
        labels = input_ids[:, 1:]
        output = self.loss(logits, labels, input_mask)
        return output

class EvalNet(nn.Cell):
    """
    GPT evaluation net

    Args:
        backbone: backbone network of GPT2/3
        generate: enable generate mode

    Inputs:
        input_ids: the tokenized inpus

    Returns:
        outputs: Tensor, corresponding output for different tasks
    """
    def __init__(self, backbone, generate=False):
        super(EvalNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.argmax = P.Argmax()
        self.generate = generate

    def construct(self, input_ids):
        """evaluation net"""
        input_mask = F.cast(F.not_equal(input_ids, 0), mstype.float32)
        logits = self.backbone(input_ids, input_mask)
        outputs = None
        if self.generate:
            outputs = nn.LogSoftmax()(logits)
            outputs = F.tensor_pow(np.e, outputs)
        else:
            outputs = self.argmax(logits)
        return outputs
