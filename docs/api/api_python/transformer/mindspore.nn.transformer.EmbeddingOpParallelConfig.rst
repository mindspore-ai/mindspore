.. py:class:: mindspore.nn.transformer.EmbeddingOpParallelConfig(data_parallel=1, model_parallel=1, vocab_emb_dp=True)

    `VocabEmbedding` 类中的并行配置，用来设置对嵌入表进行数据并行或者模型并行。

    参数：
        - **data_parallel** (int) - 表示数据并行度。根据这个数值，VocabEmbedding层的的输入数据将会被切分成原来的1/data_parallel。默认值：1。
        - **model_parallel** (int) - 表示模型并行度。根据这个数值，VocabEmbedding层的的权重将会在第0维度被切分成原来的1/model_parallel。默认值：1。
        - **vocab_emb_dp** (bool) - 表示将权重进行模型切分或数据并行。如果是True，嵌入表查找的操作将会以数据并行的方式进行，此时model_parallel的值将会被忽略。如果是False,嵌入表将会在第0维度进行切分成model_parallel份数。默认值：True。
