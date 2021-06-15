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
"""CPM model."""
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P

from src.embedding import EmbeddingPostprocessor, EmbeddingLookup
from src.attention import MaskedMultiHeadAttention
from src.util import LinearLayer, ResidualConnection, LayerNorm, Dropout


class MLPLayer(nn.Cell):
    """
    The output mapping module for each layer.

    Args:
        hidden_size (int): Length of last dim of hidden layer.
        config: The config of networks.
        dropout_prob (float): The dropout probability for network.
        is_training (bool): Whether is training.
    Inputs:
        x: output of the self-attention module.
    Returns:
        output: Tensor, the output of this layer after mapping.
    """

    def __init__(self, hidden_size, config=None, dropout_prob=0.1, is_training=False):
        super(MLPLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dense_fc = LinearLayer(hidden_size, 4 * hidden_size)

        self.dense_fc.bias_add.shard(((config.dp, config.mp), (config.mp,)))
        self.dense_fc.matmul.shard(((config.dp, 1), (config.mp, 1)))

        self.dense_proj = LinearLayer(4 * hidden_size, hidden_size)
        self.dense_proj.bias_add.shard(((config.dp, 1), (1,)))
        self.dense_proj.matmul.shard(((config.dp, config.mp), (1, config.mp)))
        self.dense_proj.matmul.add_prim_attr("recompute_comm_op", False)

        self.layernorm = LayerNorm((hidden_size,), config, epsilon=1e-5).to_float(mstype.float32)
        # parallel optimizer
        self.dense_proj.bias.parallel_optimizer = False
        self.layernorm.gamma.parallel_optimizer = False
        self.layernorm.beta.parallel_optimizer = False

        self.residual_connect = ResidualConnection()
        self.residual_connect.add.shard(((config.dp, 1), (config.dp, 1))).add_prim_attr("recompute", False)
        self.gelu = P.Gelu().shard(((config.dp, config.mp),))
        self.dropout = Dropout(1 - dropout_prob)
        self.dropout.dropout_gen_mask.shard(((config.dp, 1),))
        self.dropout.dropout_do_mask.shard(((config.dp, 1),))

        self.use_dropout = is_training
        self.reshape = P.Reshape()

    def construct(self, input_tensor):
        """FeedForward construct function."""
        # LayerNorm, eg: [5800, 2560].
        output = self.layernorm(input_tensor)

        # Feed Forward
        output = self.dense_fc(output)
        # eg: 5800, 10240
        output = self.gelu(output)
        output = self.dense_proj(output)
        if self.use_dropout:
            output = self.dropout(output)
        output = self.residual_connect(output, input_tensor)
        return output


class CPMTransformerLayer(nn.Cell):
    """
    The basic block of GPT network.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input tensor sequence.
        hidden_size (int): Length of last dim of hidden layer.
        config: The config of networks.
        num_attention_heads (int): Number of attention heads.
        attention_dropout (float): The dropout probability for attention.
        hidden_dropout (float): The dropout probability for hidden layers.
        has_attention_mask (bool): Specifies whether to use attention mask.
        is_training (bool): Whether is training.
        compute_type (:class:`mindspore.dtype`): Compute type in attention.

    Inputs:
        input_tensor: the output of previous layer(input_ids for the first layer).
        attention_mask: the attention mask matrix with shape (batch_size, seq_length, seq_length).

    Returns:
        output: Tensor, the output logit of this layer.
    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 hidden_size,
                 config=None,
                 num_attention_heads=32,
                 attention_dropout=0.1,
                 hidden_dropout=0.1,
                 has_attention_mask=True,
                 is_training=False,
                 compute_type=mstype.float16
                 ):
        super(CPMTransformerLayer, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))

        self.dim_per_head = int(hidden_size / num_attention_heads)

        self.masked_multi_head_attention = MaskedMultiHeadAttention(
            batch_size=batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            config=config,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            has_attention_mask=has_attention_mask,
            is_training=is_training,
            compute_type=compute_type
        )
        self.mlp = MLPLayer(hidden_size=hidden_size,
                            config=config,
                            dropout_prob=hidden_dropout,
                            is_training=is_training)

        self.reshape = P.Reshape()
        self.new_shape = (-1, hidden_size)

    def construct(self, input_tensor, attention_mask=None):
        # input tensor shape[batch_size, seq_length, hidden_size]
        input_tensor = self.reshape(input_tensor, self.new_shape)
        # masked multi head attention with ln, res
        attention_output = self.masked_multi_head_attention(input_tensor, attention_mask)
        # feed forward, [batch_size * seq_length, hidden_size]
        output = self.mlp(attention_output)

        return output


class CPMTransformer(nn.Cell):
    """
    Implements of gpt module.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Length of last dim of hidden layer.
        seq_length (int): Length of input tensor sequence.
        config: The config of networks.
        num_hidden_layers (int): Numbers of hidden layers.
        num_attention_heads (int): Number of attention heads.
        has_attention_mask (bool): Specifies whether to use attention mask.
        attention_dropout (float): The dropout probability for attention.
        hidden_dropout (float): The dropout probability for hidden layers.
        is_training (bool): Whether is training.
        compute_type (:class:`mindspore.dtype`): Compute type in attention.

    Returns:
        Tensor, shape of (N, T').
    """

    def __init__(self,
                 batch_size,
                 hidden_size,
                 seq_length,
                 config=None,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 has_attention_mask=True,
                 attention_dropout=0.1,
                 hidden_dropout=0.1,
                 is_training=False,
                 compute_type=mstype.float16):
        super(CPMTransformer, self).__init__()

        fusion_group_num = 4
        fusion_group_size = num_hidden_layers // fusion_group_num
        fusion_group_size = max(fusion_group_size, 1)

        layers = []
        for i in range(num_hidden_layers):
            layer = CPMTransformerLayer(batch_size=batch_size,
                                        seq_length=seq_length,
                                        hidden_size=hidden_size,
                                        config=config,
                                        num_attention_heads=num_attention_heads,
                                        attention_dropout=attention_dropout,
                                        hidden_dropout=hidden_dropout,
                                        has_attention_mask=has_attention_mask,
                                        is_training=is_training,
                                        compute_type=compute_type).set_comm_fusion(int(i / fusion_group_size) + 2)
            layer.recompute()
            layer.masked_multi_head_attention.masked_self_attention.dropout.dropout_gen_mask.recompute(False)
            layer.masked_multi_head_attention.masked_self_attention.dropout_probs.dropout_gen_mask.recompute(False)
            layer.mlp.dropout.dropout_gen_mask.recompute(False)

            layer.masked_multi_head_attention.masked_self_attention.dropout.dropout_do_mask.recompute(False)
            layer.masked_multi_head_attention.masked_self_attention.dropout_probs.dropout_do_mask.recompute(False)
            layer.mlp.dropout.dropout_do_mask.recompute(False)

            layer.masked_multi_head_attention.masked_self_attention.dropout.dropout_gen_mask.add_prim_attr(
                "_side_effect", True)
            layer.masked_multi_head_attention.masked_self_attention.dropout_probs.dropout_gen_mask.add_prim_attr(
                "_side_effect", True)
            layer.mlp.dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)
            layers.append(layer)

        self.layers = nn.CellList(layers)

        self.reshape = P.Reshape()
        self.final_layernorm = LayerNorm((hidden_size,), config, epsilon=1e-5).to_float(mstype.float32).set_comm_fusion(
            int((num_hidden_layers - 1) / fusion_group_size) + 2)

        self.final_layernorm.gamma.parallel_optimizer = False
        self.final_layernorm.beta.parallel_optimizer = False
        self.new_shape = (-1, hidden_size)

    def construct(self, input_tensor, attention_mask=None):
        """gpt module."""
        prev_output = self.reshape(input_tensor, self.new_shape)
        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask)
            prev_output = layer_output

        prev_output = self.final_layernorm(prev_output)
        return prev_output


class CPMModel(nn.Cell):
    """
    Implements of CPM model.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input tensor sequence.
        vocab_size (int): Size of the dictionary of embeddings.
        hidden_size (int): Length of last dim of hidden layer.
        num_hidden_layers (int): Numbers of hidden layers.
        num_attention_heads (int): Number of attention heads.
        config: The config of networks.
        use_one_hot_embedding (bool): Whether use one-hot embedding. Default: False.
        hidden_dropout (float): The dropout probability for hidden layers.
        attention_dropout (float): The dropout probability for attention.
        max_position_embeddings (int): The max length of position embedding.
        initializer_range (int): The initialize range of parameters.
        input_mask_from_dataset (bool): Specifies whether to use input mask.
        is_training (bool): Whether is training.
        compute_type (:class:`mindspore.dtype`): Compute type in attention.

    Returns:
        Tensor, shape of (N, T').
    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 vocab_size,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads,
                 config=None,
                 use_one_hot_embedding=False,
                 hidden_dropout=0.1,
                 attention_dropout=0.1,
                 max_position_embeddings=1024,
                 initializer_range=0.02,
                 input_mask_from_dataset=True,
                 is_training=False,
                 compute_type=mstype.float16):
        super(CPMModel, self).__init__()
        self.is_training = is_training
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.input_mask_from_dataset = input_mask_from_dataset
        self.compute_type = compute_type

        self.last_idx = self.num_hidden_layers - 1
        self.word_embedding = EmbeddingLookup(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            config=config,
            use_one_hot_embeddings=use_one_hot_embedding,
            compute_type=self.compute_type
        ).set_comm_fusion(1)
        self.position_embedding = EmbeddingPostprocessor(
            max_seq_length=self.max_position_embeddings,
            embedding_dim=self.embedding_dim,
            config=config,
            use_one_hot_embeddings=use_one_hot_embedding,
            compute_type=self.compute_type
        ).set_comm_fusion(1)
        self.transformer = CPMTransformer(batch_size=self.batch_size,
                                          hidden_size=self.hidden_size,
                                          seq_length=self.seq_length,
                                          config=config,
                                          num_hidden_layers=self.num_hidden_layers,
                                          num_attention_heads=self.num_attention_heads,
                                          has_attention_mask=True,
                                          attention_dropout=self.attention_dropout,
                                          hidden_dropout=self.hidden_dropout,
                                          is_training=self.is_training,
                                          compute_type=self.compute_type)
        self.dropout = Dropout(1 - self.hidden_dropout)
        self.dropout.dropout_gen_mask.shard(((config.dp, 1, 1),))
        self.dropout.dropout_do_mask.shard(((config.dp, 1, 1),))

        self.matmul = P.MatMul(transpose_b=True).shard(((config.dp, 1), (1, 1)))
        self.reshape = P.Reshape()
        self.add = P.TensorAdd().shard(((config.dp, 1, config.mp), (config.dp, 1, config.mp)))
        self.cast = P.Cast()
        self.out_shape = (-1, self.seq_length, self.vocab_size)

    def construct(self, input_ids, position_ids=None, attention_mask=None):
        """
        Construct network.

        Args:
            input_ids (Tensor): Input sentences with shape (N, T).
            position_ids (Tensor): Target of input sentences with shape (N, T).
            attention_mask (Tensor): Source sentences padding mask with shape (N, T, T).

        Returns:
            Tensor, network outputs.
        """
        words_embeddings, embedding_tab = self.word_embedding(input_ids)
        position_embedding = self.position_embedding(position_ids)
        embedding = self.add(words_embeddings, position_embedding)

        if self.is_training:
            embedding = self.dropout(embedding)

        transformer_output = self.transformer(embedding, attention_mask)

        transformer_output = self.cast(transformer_output, self.compute_type)
        logits_output = self.matmul(transformer_output,
                                    self.cast(embedding_tab, self.compute_type))
        logits_output = self.reshape(logits_output, self.out_shape)
        logits_output = self.cast(logits_output, mstype.float32)

        return logits_output
