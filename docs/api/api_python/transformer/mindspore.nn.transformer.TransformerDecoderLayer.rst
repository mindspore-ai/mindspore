.. py:class:: mindspore.nn.transformer.TransformerDecoderLayer(hidden_size, ffn_hidden_size, num_heads, batch_size, src_seq_length, tgt_seq_length, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, post_layernorm_residual=False, use_past=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, hidden_act="gelu", moe_config=default_moe_config, parallel_config=default_dpmp_config)

    Transformer的解码器层。Transformer的解码器层上的单层的实现，包括自注意力层、交叉注意力层和前馈层。当encoder_output为None时，交叉注意力将无效。

    参数：
        - **batch_size** (int) - 表示增量预测时输入张量的批量大小，应该是正整数。当进行训练或预测时，该参数将不起作用，用户可将None传递给此参数。
        - **hidden_size** (int) - 表示输入的隐藏大小。
        - **src_seq_length** (int) - 表示输入源序列长度。
        - **tgt_seq_length** (int) - 表示输入目标序列长度。
        - **ffn_hidden_size** (int) - 表示前馈层中bottleneck的隐藏大小。
        - **num_heads** (int) - 表示注意力头的数量。
        - **hidden_dropout_rate** (float) - 表示作用在隐藏层输出的丢弃率。默认值：0.1
        - **attention_dropout_rate** (float) - 表示注意力score的丢弃率。默认值：0.1
        - **post_layernorm_residual** (bool) - 表示是否在LayerNorm之前使用残差，即是否选择残差为Post-LayerNorm或者Pre-LayerNorm。默认值：False
        - **hidden_act** (str) - 表示内部前馈层的激活函数。其值可为'relu'、'relu6'、'tanh'、'gelu'、'fast_gelu'、'elu'、'sigmoid'、'prelu'、'leakyrelu'、'hswish'、'hsigmoid'、'logsigmoid'等等。默认值：gelu。
        - **layernorm_compute_type** (dtype.Number) - 表示LayerNorm的计算类型。其值应为mstype.float32或mstype.float16。默认值为mstype.float32。
        - **softmax_compute_type** (dtype.Number) - 表示注意力中softmax的计算类型。其值应为mstype.float32或mstype.float16。默认值为mstype.float32。
        - **param_init_type** (dtype.Number) - 表示模块的参数初始化类型。其值应为mstype.float32或mstype.float16。默认值为mstype.float32。
        - **use_past** (bool) - 使用过去状态进行计算，用于增量预测。默认值：False。
        - **moe_config** (MoEConfig) - 表示MoE (Mixture of Expert)的配置。默认值为 `default_moe_config` ，表示带有默认参数的 `MoEConfig` 实例。
        - **parallel_config** (OpParallelConfig, MoEParallelConfig) - 表示并行配置。默认值为 `default_dpmp_config` ，表示一个带有默认参数的 `OpParallelConfig` 实例。

    输入：
        - **hidden_stats** (Tensor) - shape为[batch_size, tgt_seq_length, hidden_size]或[batch_size * tgt_seq_length, hidden_size]的输入tensor。
        - **decoder_mask** (Tensor) - shape为[batch_size, src_seq_length, seq_length]的解码器的注意力掩码。或者为None，None表示将不会在self attention中的Softmax计算中引入掩码计算。
        - **encoder_output** (Tensor) - shape为[batch_size, seq_length, hidden_size]或[batch_size * seq_length, hidden_size]的编码器的输出。注：当网络位于最外层时，此参数不能通过None传递。默认值为None。
        - **memory_mask** (Tensor) - shape为[batch, tgt_seq_length, src_seq_length]的交叉注意力的memory掩码，其中tgt_seq_length表示解码器的长度。或者为None，None表示将不会在cross attention中的Softmax计算中引入掩码计算。
        - **init_reset** (Tensor) - shape为[1]的bool tensor，用于清除增量预测中使用的past key参数和past value参数。仅当use_past为True时有效。默认值为True。
        - **batch_valid_length** (Tensor) - shape为[batch_size]的Int32 tensor，表示过去所计算的索引。当use_past为True时，它用于增量预测。默认值为None。

    输出：
        Tuple，表示一个包含(`output`, `layer_present`)的元组。

        - **output** (Tensor) - 此层的输出logit。shape为[batch, seq_length, hidden_size]或[batch * seq_length, hidden_size]。
        - **layer_present** (Tuple) - 元组，其中每个元组都是shape为((batch_size, num_heads, size_per_head, tgt_seq_length)或(batch_size, num_heads, tgt_seq_length, size_per_head)的自注意力中的投影key向量和value向量的tensor的元组，或者是shape为(batch_size, num_heads, size_per_head, src_seq_length)或(batch_size, num_heads, src_seq_length, size_per_head))的交叉注意力中的投影key向量和value向量的tensor的元组。
