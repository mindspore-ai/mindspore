mindspore.parallel.merge_pipeline_strategys
========================================================================

.. py:function:: mindspore.parallel.merge_pipeline_strategys(src_strategy_dirs, dst_strategy_file)

    汇聚所有流水线并行子图的切分策略文件到目的文件。

    .. note::
        `src_strategy_dirs` 必须包含所有流水线并行的子图的切分策略文件。

    参数：
        - **src_strategy_dirs** (str) - 包含所有流水线并行的子图的切分策略文件的目录，切分策略文件由 :func:`mindspore.parallel.auto_parallel.AutoParallel.save_param_strategy_file` 接口存储得到。
        - **dst_strategy_file** (str) - 保存汇聚后的切分策略的文件路径。

    异常：
        - **NotADirectoryError** - `src_strategy_dirs` 不是一个目录。
