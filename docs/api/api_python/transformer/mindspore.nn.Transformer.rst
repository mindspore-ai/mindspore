.. py:class:: mindspore.nn.transformer.Transformer(hidden_size, batch_size, ffn_hidden_size, src_seq_length, tgt_seq_length, encoder_layers=3, decoder_layers=3, num_heads=2, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, hidden_act="gelu", post_layernorm_residual=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, lambda_func=None, use_past=False, moe_config=default_moe_config, parallel_config=default_transformer_config)

    Transformer模块，包括编码器和解码器。与原始的实现方式的区别在于该模块在实行层归一化之前使用了残差加法。默认的激活层为`gelu`。
    详细信息可参考 `Attention Is All You Need <https://arxiv.org/pdf/1706.03762v5.pdf>`_ 。

    .. note::
        这是一个实验接口，可能会被更改或者删除。

    **参数：**

    - **batch_size** (int) - 表示输入的批次大小。
    - **encoder_layers** (int) - 表示`TransformerEncoderLayer`的层数。
    - **decoder_layers** (int) - 表示`TransformerDecoderLayer`的层数。
    - **hidden_size** (int) - 表示输入向量的大小。
    - **ffn_hidden_size** (int) - 表示前馈层中bottleneck的隐藏大小。
    - **src_seq_length** (int) - 表示编码器的输入Tensor的seq_length。
    - **tgt_seq_length** (int) - 表示解码器的输入Tensor的seq_length。
    - **num_heads** (int) - 表示注意力头的数量。默认值：2
    - **hidden_dropout_rate** (float) - 表示作用在隐藏层输出的丢弃率。默认值：0.1
    - **attention_dropout_rate** (float) - 表示注意力score的丢弃率。默认值：0.1
    - **post_layernorm_residual** (bool) - 表示是否在LayerNorm之前使用残差，即是否选择残差为Post-LayerNorm或者Pre-LayerNorm。默认值：False
    - **layernorm_compute_type** (dtype.Number) - 表示LayerNorm的计算类型。其值应为dtype.float32或dtype.float16。默认值为dtype.float32。
    - **softmax_compute_type** (dtype.Number) - 表示注意力机制中softmax的计算类型。其值应为dtype.float32或dtype.float16。默认值为mstype.float32。
    - **param_init_type** (dtype.Number) - 表示模块的参数初始化类型。其值应为dtype.float32或dtype.float16。默认值为dtype.float32。
    - **hidden_act** (str) - 表示前馈层的激活行为。其值可为'relu'、'relu6'、'tanh'、'gelu'、'fast_gelu'、'elu'、'sigmoid'、'prelu'、'leakyrelu'、'hswish'、'hsigmoid'、'logsigmoid'等等。默认值：gelu。
    - **moe_config** (MoEConfig) - 表示MoE (Mixture of Expert)的配置。
    - **lambda_func** - 表示设置融合索引、pipeline阶段和重计算属性的函数。如果用户想确定pipeline阶段和梯度融合，用户可以传递一个接受 `network` 、 `layer_id` 、 `offset` 、 `parallel_config` 和 `layers` 的函数。 `network(Cell)` 表示transformer块， `layer_id(int)` 表示当前模块的层索引，从零开始计数， `offset(int)` 表示如果网络中还有其他模块，则layer_id需要一个偏移。pipeline的默认设置为： `(layer_id + offset) // ((encoder_layers + decoder_length) / pipeline_stage)` 。
    - **parallel_config** (TransformerOpParallelConfig) - 表示并行配置。默认值为 `default_transformer_config` ，表示带有默认参数的 `TransformerOpParallelConfig` 实例。

    **输入：**

    - **encoder_inputs** (Tensor) - shape为[batch_size, seq_length, hidden_size]或[batch_size * seq_length, hidden_size]的输入Tensor。
    - **encoder_masks** (Tensor) - shape为[batch_size, seq_length, seq_length]的解码器的注意力掩码。
    - **decoder_inputs** (Tensor) - shape为[batch_size, seq_length, hidden_size]或[batch_size * seq_length, hidden_size]的编码器的输出。如果解码器层数为0，则此值应为None。
    - **decoder_masks** (Tensor) - shape为[batch_size, seq_length, seq_length]的解码器的注意力掩码。
    - **memory_mask** (Tensor) - shape为[batch, tgt_seq_length,  src_seq_length]的交叉注意力的memory掩码，其中tgt_seq_length表示解码器的长度。如果解码器层为0，则shape为[batch_size, seq_length, hidden_size]的编码器的输出应为None。
    - **init_reset** (Tensor) - shape为[1]的bool tensor，用于清除增量预测中使用的past key参数和past value参数。仅当use_past为True时有效。默认值为True。
    - **batch_valid_length** (Tensor) - shape为[batch_size]的Int32 tensor，表示过去所计算的索引。当use_past为True时，它用于增量预测。默认值为None。

    **输出：**

    Tuple，表示包含(`output`, `encoder_layer_present`, `encoder_layer_present`)的元组。

    - **output** (Tensor) - 如果只有编码器，则表示编码器层的输出logit。shape为[batch, src_seq_length, hidden_size] or [batch * src_seq_length, hidden_size]。如果有编码器和解码器，则输出来自于解码器层。shape为[batch, tgt_seq_length, hidden_size]或[batch * tgt_seq_length, hidden_size]。
    - **encoder_layer_present** (Tuple) - 大小为num_layers的元组，其中每个元组都是shape为((batch_size, num_heads, size_per_head, src_seq_length)或(batch_size, num_heads, src_seq_length, size_per_head))的自注意力中的投影key向量和value向量的tensor。
    - **decoder_layer_present** (Tuple) - 大小为num_layers的元组，其中每个元组都是shape为((batch_size, num_heads, size_per_head, tgt_seq_length)或(batch_size, num_heads, tgt_seq_length, size_per_head))的self attention中的投影key向量和value向量的tensor，或者是shape为(batch_size, num_heads, size_per_head, src_seq_length)或(batch_size, num_heads, src_seq_length, size_per_head))的交叉注意力中的投影key向量和value向量的tensor。如果未设置解码器，返回值将为None。
