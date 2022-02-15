.. py:class:: mindspore.parallel.nn.EmbeddingOpParallelConfig(data_parallel=1, model_parallel=1, vocab_emb_dp=True)

    `VocabEmbedding` 类中的并行配置。当vocab_emb_dp为True时，设置Embedding查找为数据并行，其中model_parallel参数会被忽略。当vocab_emb_dp为False时，在Embedding表的第0轴进行按model_parallel的大小进行切分。

    **参数：**

    - **data_parallel** (int) - 表示数据并行度。默认值：1。
    - **model_parallel** (int) - 表示模型平行度。默认值：1。
    - **vocab_emb_dp** (bool) - 表示模型并行或数据并行中的Shard embedding。默认值：True。

    .. py:method:: dp_mp_config()

        获取包含有data_parallel和model_parallel属性的DPMPlConfig类。
