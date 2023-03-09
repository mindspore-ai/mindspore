mindspore.transform_checkpoints
======================================

.. py:function:: mindspore.transform_checkpoints(src_checkpoints_dir, dst_checkpoints_dir, ckpt_prefix, src_strategy_file=None, dst_strategy_file=None)

    将一个分布式网络的Checkpoint由源切分策略转换到目标切分策略。关于更多分布式Checkpoint转换的细节，请参考：`分布式弹性训练与推理 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/resilience_train_and_predict.html>`_。

    .. note::
        `src_checkpoints_dir` 目录必须组织为“src_checkpoints_dir/rank_0/a.ckpt”这样的目录结构，rank号必须作为子目录并且该rank的Checkpoint必须放置于该子目录内。如果多个文件存在于一个rank目录下，将会选名字的字典序最高的文件。

    参数：
        - **src_checkpoints_dir** (str) - 源Checkpoint文件所在的目录。
        - **dst_checkpoints_dir** (str) - 目标Checkpoint文件存储的目录。
        - **ckpt_prefix** (str) - 目标Checkpoint前缀名。
        - **src_strategy_file** (str) - 源切分策略proto文件名，由mindspore.set_auto_parallel_context(strategy_ckpt_save_file)接口存储下来的文件。当其为None时，表示切分策略为不切分。默认值：None。
        - **dst_strategy_file** (str) - 目标切分策略proto文件名，由mindspore.set_auto_parallel_context(strategy_ckpt_save_file)接口存储下来的文件。当其为None时，表示切分策略为不切分。默认值：None。

    异常：
        - **ValueError** - `src_strategy_file` 或者 `dst_strategy_file` 不是正确的切分策略proto文件。
        - **NotADirectoryError** - `src_checkpoints_dir` 或者 `dst_checkpoints_dir` 不是一个目录。
        - **ValueError** - `src_checkpoints_dir` 中缺失了Checkpoint文件。
        - **TypeError** - `src_strategy_file` 或者 `dst_strategy_file` 不是字符串。
