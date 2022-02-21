.. py:class:: mindspore.nn.transformer.TransformerOpParallelConfig(data_parallel=1, model_parallel=1, pipeline_stage=1, micro_batch_num=1, recompute=False, optimizer_shard=False, gradient_aggregation_group=4, vocab_emb_dp=True)

    用于设置数据并行、模型并行等等并行配置的TransformerOpParallelConfig。

    .. note::
        除recompute参数外，当用户未将auto_parallel_context设为 `SEMI_AUTO_PARALLEL` 或 `AUTO_PARALLEL` 时，其他参数将无效。
        在训练时，micro_batch_num的值必须大于或等于equal to pipeline_stage的值。data_parallel\*model_parallel  \*pipeline_stage的值必须等于或小于总设备的数量。设置pipeline_stage和optimizer_shard时，其配置将覆盖auto_parallel_context的配置。

    **参数：**

    - **data_parallel** (int) - 表示数据并行数。默认值：1。
    - **model_parallel** (int) - 表示模型并行数。默认值：1。
    - **pipeline_stage** (int) - 表示将Transformer切分成的stage数目。其值应为正数。默认值：1。
    - **micro_batch_num** (int) - 表示用于pipeline训练的batch的微型大小。默认值：1。
    - **optimizer_shard** (bool) - 表示是否使能优化器切分。默认值：False。
    - **gradient_aggregation_group** (int) - 表示优化器切分的融合组大小。默认值：4。
    - **recompute** (bool) - 表示是否启用transformer每层的的重计算。默认值：False。
    - **vocab_emb_dp** (bool) - 表示Embedding表是否为数据并行，否则将在查找表中的第0维度根据模型并行度进行切分。默认值：True。

    .. py:method:: dp_mp_config()

        获取包含数据并行、模型并行度的DPMPlConfig。

    .. py:method:: embedding_dp_mp_config()

        获取包含数据并行、模型并行和embedding并行度的EmbeddingParallelConfig。
