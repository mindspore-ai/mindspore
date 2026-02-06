mindspore.parallel.convert_checkpoints
======================================

.. py:function:: mindspore.parallel.convert_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix, src_strategy_file=None, dst_strategy_file=None, process_num=1, output_format="ckpt")

    将一个分布式网络的Checkpoint由源切分策略转换到目标切分策略。

    .. note::
        `src_checkpoints_dir` 目录必须设置为“src_checkpoints_dir/rank_0/a.ckpt”这样的目录结构。rank号必须作为子目录，并且该rank的Checkpoint必须放置于该子目录内。如果多个文件存在于一个rank目录下，将会选择名字的字典序最高的文件。
        多进程设置数量与主机规模有关，不推荐设置太大，否则容易导致卡死。

    参数：
        - **src_checkpoints_dir** (str) - 源Checkpoint文件所在的目录。
        - **dst_checkpoints_dir** (str) - 目标Checkpoint文件存储的目录。
        - **ckpt_prefix** (str) - 目标Checkpoint前缀名。
        - **src_strategy_file** (str, 可选) - 源切分策略proto文件名，由 :func:`mindspore.parallel.auto_parallel.AutoParallel.save_param_strategy_file` 接口存储下来的文件。当其为 ``None`` 时，表示切分策略为不切分。默认值： ``None`` 。
        - **dst_strategy_file** (str, 可选) - 目标切分策略proto文件名，由 :func:`mindspore.parallel.auto_parallel.AutoParallel.save_param_strategy_file` 接口存储下来的文件。当其为 ``None`` 时，表示切分策略为不切分。默认值： ``None`` 。
        - **process_num** (int, 可选) - 控制并行处理的进程数量。默认值： ``1``。
        - **output_format** (str, 可选) - 控制转换后输出的 checkpoint 格式。可以设置为 ``"ckpt"`` 或 ``"safetensors"`` 。默认值： ``"ckpt"`` 。

    异常：
        - **ValueError** - `src_strategy_file` 或者 `dst_strategy_file` 不是正确的切分策略proto文件。
        - **NotADirectoryError** - `src_checkpoints_dir` 或者 `dst_checkpoints_dir` 不是一个目录。
        - **ValueError** - `src_checkpoints_dir` 中缺失Checkpoint文件。
        - **TypeError** - `src_strategy_file` 或者 `dst_strategy_file` 不是字符串。
