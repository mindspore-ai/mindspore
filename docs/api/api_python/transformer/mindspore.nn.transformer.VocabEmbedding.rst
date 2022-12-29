.. py:class:: mindspore.nn.transformer.VocabEmbedding(vocab_size, embedding_size, parallel_config=default_embedding_parallel_config, param_init="normal")

    根据输入的索引查找参数表中的行作为返回值。当设置并行模式为 `AUTO_PARALLEL_MODE` 时，如果parallel_config.vocab_emb_dp为True时，那么embedding lookup表采用数据并行的方式，数据并行度为 `parallel_config.data_parallel` ，如果为False，按 `parallel_config.model_parallel` 对embedding表中的第零维度进行切分。

    .. note::
        启用 `AUTO_PARALLEL` / `SEMI_AUTO_PARALLEL` 模式时，此层仅支持二维度的输入，因为策略是为2D输入而配置的。

    参数：
        - **vocab_size** (int) - 表示查找表的大小。
        - **embedding_size** (int) - 表示查找表中每个嵌入向量的大小。
        - **param_init** (Union[Tensor, str, Initializer, numbers.Number]) - 表示embedding_table的Initializer。当指定字符串时，请参见 `initializer` 类了解字符串的值。默认值：'normal'。
        - **parallel_config** (EmbeddingOpParallelConfig) - 表示网络的并行配置。默认值为 `default_embedding_parallel_config` ，表示带有默认参数的 `EmbeddingOpParallelConfig` 实例。

    输入：
        - **input_ids** (Tensor) - shape为(batch_size, seq_length)的输入，其数据类型为int32。

    输出：
        Tuple，表示一个包含(`output`, `embedding_table`)的元组。

        - **output** (Tensor) - shape为(batch_size, seq_length, embedding_size)嵌入向量查找结果。
        - **embedding_table** (Tensor) - shape为(vocab_size, embedding_size)的嵌入表。

    异常：
        - **ValueError** - parallel_config.vocab_emb_dp为True时，词典的大小不是parallel_config.model_parallel的倍数。
        - **ValueError** - `vocab_size` 不是正值。
        - **ValueError** - `embedding_size` 不是正值。
        - **TypeError** - `parallel_config` 不是OpParallelConfig的子类。
