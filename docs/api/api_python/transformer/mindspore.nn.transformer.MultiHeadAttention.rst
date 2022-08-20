.. py:class:: mindspore.nn.transformer.MultiHeadAttention(batch_size, src_seq_length, tgt_seq_length, hidden_size, num_heads, hidden_dropout_rate=0.1, attention_dropout_rate=0.1, compute_dtype=mstype.float16, softmax_compute_type=mstype.float32, param_init_type=mstype.float32, use_past=False, parallel_config=default_dpmp_config)

    论文 `Attention Is All You Need <https://arxiv.org/pdf/1706.03762v5.pdf>`_ 中所述的多头注意力的实现。给定src_seq_length长度的query向量，tgt_seq_length长度的key向量和value，注意力计算流程如下：

    .. math::
           MultiHeadAttention(query, key, vector) = Dropout(Concat(head_1, \dots, head_h)W^O)

    其中， :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)` 。注意：输出层的投影计算中带有偏置参数。

    如果query tensor、key tensor和value tensor相同，则上述即为自注意力机制的计算过程。

    参数：
        - **batch_size** (int) - 表示增量预测时输入张量的批量大小，应该是正整数。当进行训练或预测时，该参数将不起作用，用户可将None传递给此参数。
        - **src_seq_length** (int) - 表示query向量的序列长度。
        - **tgt_seq_length** (int) - 表示key向量和value向量的序列长度。
        - **hidden_size** (int) - 表示输入的向量大小。
        - **num_heads** (int) - 表示注意力机制中头的数量。
        - **hidden_dropout_rate** (float) - 表示最后dense输出的丢弃率。默认值：0.1
        - **attention_dropout_rate** (float) - 表示注意力score的丢弃率。默认值：0.1
        - **compute_dtype** (dtype.Number) - 表示dense中矩阵乘法的计算类型。默认值：mstype.float16。其值应为mstype.float32或mstype.float16。
        - **param_init_type** (dtype.Number) - 表示模块的参数初始化类型。默认值：mstype.float32。其值应为mstype.float32或mstype.float16。
        - **softmax_compute_type** (dtype.Number) - 表示softmax计算模块的类型。默认值：mstype.float32。  其值应为mstype.float32或mstype.float16。
        - **use_past** (bool) - 使用过去状态进行计算，用于增量预测。例如，如果我们有两个单词，想生成十个或以上单词。我们只需要计算一次这两个单词的状态，然后逐个生成下一个单词。当use_past为True时，有两个步骤可以执行预测。
          第一步是通过 `model.add_flags_recursive(is_first_iteration=True)` 将is_first_iteration设为True，并传递完整的输入。然后，通过 `model.add_flags_recursive(is_first_iteration=False)` 将is_first_iteration设为False。此时，传递step的输入tensor，并对其进行循环。默认值：False
        - **parallel_config** (OpParallelConfig) - 表示并行配置。默认值为 `default_dpmp_config` ，表示一个带有参数的 `OpParallelConfig` 实例。

    输入：
        - **query_tensor** (Tensor) - use_past为False或is_first_iteration为True时，表示shape为(batch_size, src_seq_length, hidden_size)或(batch_size * src_seq_length, hidden_size)的query向量。否则，shape必须为(batch_size, 1, hidden_size)。
        - **key_tensor** (Tensor) - use_past为False或is_first_iteration为True时，表示shape为(batch_size, tgt_seq_length, hidden_size)或(batch_size * tgt_seq_length, hidden_size)的key向量。否则，shape必须为(batch_size, 1, hidden_size)。
        - **value_tensor** (Tensor) - use_past为False或is_first_iteration为True时，表示shape为(batch_size, tgt_seq_length, hidden_size)或(batch_size * tgt_seq_length, hidden_size)的value向量。否则，shape必须为(batch_size, 1, hidden_size)。
        - **attention_mask** (Tensor) - use_past为False或is_first_iteration为True时，表示shape为(batch_size, src_seq_length, tgt_seq_length)的注意力掩码矩阵, 或者为None，None表示在Softmax计算中将不会进行掩码。否则，shape必须为(batch_size, 1, tgt_seq_length)。
        - **key_past** (Tensor) - shape为(batch_size, num_heads, size_per_head, tgt_seq_length)的Float16 tensor，表示过去所计算的key向量。
          当use_past为True时，需要传入非None值用于增量预测。默认值为None。
        - **value_past** (Tensor) - shape为(batch_size, num_heads, tgt_seq_length, size_per_head)的Float16 tensor，表示过去所计算的value向量。
          当use_past为True时，需要传入非None值用于增量预测。默认值为None。
        - **batch_valid_length** (Tensor) - shape为(batch_size,)的Int32 tensor，表示已经计算的token索引。
          当use_past为True时，需要传入非None值用于增量预测。默认值为None。

    输出：
        Tuple，表示一个包含(`output`, `layer_present`)的元组。

        - **output** (Tensor) - Tensor。use_past为False或is_first_iteration为True时，表示shape为(batch_size, src_seq_length, hidden_size)或(batch_size * src_seq_length, hidden_size)的层输出的float tensor。否则，shape将为(batch_size, 1, hidden_size)。
        - **layer_present** (Tuple) - 表示shape为((batch_size, num_heads, size_per_head, tgt_seq_length)或(batch_size, num_heads, tgt_seq_length, size_per_head))的投影key向量和value向量的Tensor的元组。
