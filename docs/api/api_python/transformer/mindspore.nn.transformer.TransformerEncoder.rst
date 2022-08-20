.. py:class:: mindspore.nn.transformer.TransformerEncoder(batch_size, num_layers, hidden_size, ffn_hidden_size, seq_length, num_heads, attention_dropout_rate=0.1, hidden_dropout_rate=0.1, hidden_act="gelu", post_layernorm_residual=False, layernorm_compute_type=mstype.float32, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, lambda_func=None, offset=0, use_past=False, moe_config=default_moe_config, parallel_config=default_transformer_config)

    Transformer中的编码器模块，具有多层堆叠的 `TransformerEncoderLayer` ，包括多头自注意力层和前馈层。

    参数：
        - **batch_size** (int) - 表示增量预测时输入张量的批量大小，应该是正整数。当进行训练或预测时，该参数将不起作用，用户可将None传递给此参数。
        - **num_layers** (int) - 表示 `TransformerEncoderLayer` 的层。
        - **hidden_size** (int) - 表示输入的隐藏大小。
        - **ffn_hidden_size** (int) - 表示前馈层中bottleneck的隐藏大小。
        - **seq_length** (int) - 表示输入序列长度。
        - **num_heads** (int) - 表示注意力头的数量。
        - **hidden_dropout_rate** (float) - 表示作用在隐藏层输出的丢弃率。默认值：0.1
        - **attention_dropout_rate** (float) - 表示注意力score的丢弃率。默认值：0.1
        - **post_layernorm_residual** (bool) - 表示是否在LayerNorm之前使用残差，即是否选择残差为Post-LayerNorm或者Pre-LayerNorm。默认值：False
        - **hidden_act** (str) - 表示内部前馈层的激活函数。其值可为'relu'、'relu6'、'tanh'、'gelu'、'fast_gelu'、'elu'、'sigmoid'、'prelu'、'leakyrelu'、'hswish'、'hsigmoid'、'logsigmoid'等等。默认值：gelu。
        - **layernorm_compute_type** (dtype.Number) - 表示LayerNorm的计算类型。其值应为mstype.float32或mstype.float16。默认值为mstype.float32。
        - **softmax_compute_type** (dtype.Number) - 表示注意力中softmax的计算类型。其值应为mstype.float32或mstype.float16。默认值为mstype.float32。
        - **param_init_type** (dtype.Number) - 表示模块的参数初始化类型。其值应为mstype.float32或mstype.float16。默认值为mstype.float32。
        - **use_past** (bool) - 使用过去状态进行计算，用于增量预测。例如，如果我们有两个单词，想生成十个或以上单词。我们只需要计算一次这两个单词的状态，然后逐个生成下一个单词。当use_past为True时，有两个步骤可以运行预测。第一步是通过 `model.add_flags_recursive(is_first_iteration=True)` 将is_first_iteration设为True，并传递完整的输入。然后，通过 `model.add_flags_recursive(is_first_iteration=False)` 将is_first_iteration设为False。此时，传递step的输入tensor，并对其进行环回。默认值：False。
        - **lambda_func** (function) - 表示设置融合索引、pipeline阶段和重计算属性的函数。如果用户想确定pipeline阶段和梯度聚合融合，用户可以传递一个接受 `network` 、 `layer_id` 、 `offset` 、 `parallel_config` 和 `layers` 的函数。 `network(Cell)` 表示transformer块， `layer_id(int)` 表示当前模块的层索引，从零开始计数， `offset(int)` 表示如果网络中还有其他模块，则layer_index需要一个偏置。pipeline的默认设置为： `(layer_id + offset) // (layers / pipeline_stage)` 。默认值：None。
        - **offset** (int) - 表示 `decoder` 的初始层索引。其用于设置梯度聚合的融合值和流水线并行的stage值。默认值：0。
        - **moe_config** (MoEConfig) - 表示MoE (Mixture of Expert)的配置。默认值为 `default_moe_config` ，表示带有默认参数的 `MoEConfig` 实例。
        - **parallel_config** (TransformerOpParallelConfig) - 表示并行配置。默认值为 `default_transformer_config` ，表示带有默认参数的 `TransformerOpParallelConfig` 实例。

    输入：
        - **hidden_states** (Tensor) - Tensor。如果use_past为False或者is_first_iteration为True，shape为[batch_size, seq_length, hidden_size]或者[batch_size * seq_length, hidden_size]。否则，shape应为[batch_size, 1, hidden_size]。
        - **attention_mask** (Tensor) - Tensor。use_past为False或者is_first_iteration为True时，表示shape为[batch_size, seq_length, seq_length]的注意力掩码，或者为None，None表示在Softmax计算中将不会进行掩码。否则，shape应为[batch_size, 1, hidden_size]。
        - **init_reset** (Tensor) - shape为[1]的bool tensor，用于清除增量预测中使用的past key参数和past value参数。仅当use_past为True时有效。默认值为True。
        - **batch_valid_length** (Tensor) - shape为[batch_size]的Int32 tensor，表示过去所计算的索引。当use_past为True时，它用于增量预测。默认值为None。

    输出：
        Tuple，表示一个包含(`output`, `layer_present`)的元组。

        - **output** (Tensor) - use_past为False或is_first_iteration为True时，表示shape为(batch_size, seq_length, hidden_size)或(batch_size * seq_length, hidden_size)的层输出的float tensor。否则，shape将为(batch_size, 1, hidden_size)。
        - **layer_present** (Tuple) - 大小为num_layers的元组，其中每个元组都包含shape为((batch_size, num_heads, size_per_head, seq_length)或(batch_size, num_heads, seq_length, size_per_head))的投影key向量和value向量的Tensor的元组。
