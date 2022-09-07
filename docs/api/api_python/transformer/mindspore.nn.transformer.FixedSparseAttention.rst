.. py:class:: mindspore.nn.transformer.FixedSparseAttention(batch_size, num_heads, size_per_head, block_size, seq_length=1024, num_different_global_patterns=4, parallel_config=default_dpmp_config)

    固定稀疏注意力层。

    此接口实现了Sparse Transformer中使用的稀疏注意力原语，更多详情，请见论文 `Generating Long Sequences with Sparse Transformers <https://arxiv.org/abs/1904.10509>`_ 。

    具体来说，它包括以下内容：

    1. 正常注意力的更快实现（不计算上三角，并且融合了许多操作）。
    2. 如论文Sparse Transformers所述，“分散”和“固定”注意力的实现。

    参数：
        - **batch_size** (int) - 表示输入batch size的数量。
        - **num_heads** (int) - 表示注意力头数。
        - **block_size** (int) - 表示用来确定block size的整数。目前稀疏自注意力的实现基于稀疏块矩阵。此参数定义了稀疏矩阵块的大小。目前仅支持64。
        - **seq_length** (int) - 表示输入序列的长度。目前只支持1024。默认值为1024。
        - **num_different_global_patterns** (int) - 表示用于确定不同的全局注意力数量。虽然全局注意力由局部的代表性的块决定，
          但由于有多个头，所以每个头都可以使用不同的全局代表。目前只支持4。默认值为4。
        - **size_per_head** (int) - 表示每个注意力头的向量大小。目前仅支持64和128。
        - **parallel_config** (OpParallelConfig) - 并行设置，内容请参阅 `OpParallelConfig` 的定义。默认值为 `default_dpmp_config` ，一个用默认参数初始化的 `OpParallelConfig` 的实例。

    输入：
        - **q** (Tensor) - Tensor query (:class:`mstype.fp16` [batch_size, seq_length, hidden_size])：表示上下文的query向量。
        - **k** (Tensor) - Tensor key (:class:`mstype.fp16` [batch_size, seq_length, hidden_size])：表示上下文的key向量。
        - **v** (Tensor) - Tensor value (:class:`mstype.fp16` [批次大小, seq_length, hidden_size])：表示上下文的value向量。
        - **attention_mask** (Tensor) - Float Tensor the mask of (:class:`mstype.fp32` , :class:`mstype.fp16` [batch_size, seq_length, seq_length])：
          表示掩码的下三角形矩阵。

    输出：
        Tensor，shape为[batch_size, seq_length, hidden_size]。
