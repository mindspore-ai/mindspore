mindspore.merge_pipeline_strategys
======================================

.. py:function:: mindspore.merge_pipeline_strategys(src_strategy_dirs, dst_strategy_file)

    流水线并行模式下，汇聚所有流水线并行子图的切分策略文件。关于更多分布式Checkpoint转换的细节，请参考：`分布式弹性训练与推理 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/resilience_train_and_predict.html>`_。

    .. note::
        src_strategy_dirs必须包含所有流水线并行的子图的切分策略文件。

    参数：
        - **src_strategy_dirs** (str) - 包含所有流水线并行的子图的切分策略文件的目录，切分策略文件由mindspore.set_auto_parallel_context(strategy_ckpt_save_file)接口存储得到。
        - **dst_strategy_file** (str) - 保存汇聚后的切分策略的文件路径。

    异常：
        - **NotADirectoryError** - `src_strategy_dirs` 不是一个目录。
