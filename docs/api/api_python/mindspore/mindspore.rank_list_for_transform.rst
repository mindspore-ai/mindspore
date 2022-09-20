mindspore.rank_list_for_transform
======================================

.. py:function:: mindspore.rank_list_for_transform(rank_id, src_strategy_file=None, dst_strategy_file=None)

    在对分布式Checkpoint转换的过程中，获取为了得到目标rank的Checkpoint文件所需的源Checkpoint文件rank列表。关于更多分布式Checkpoint转换的细节，请参考：[分布式弹性训练与推理](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/resilience_train_and_predict.html)。

    .. note::
        当前暂时不支持流水线并行维度的转换。

    参数：
        - **rank_id** (int) - 待转换得到的Checkpoint的rank号。
        - **src_strategy_file** (str) - 源切分策略proto文件名，由context.set_auto_parallel_context(strategy_ckpt_save_file)接口存储下来的文件。当其为None时，表示切分策略为不切分。默认值：None。
        - **dst_strategy_file** (str) - 目标切分策略proto文件名，由context.set_auto_parallel_context(strategy_ckpt_save_file)接口存储下来的文件。当其为None时，表示切分策略为不切分。默认值：None。

    异常：
        - **ValueError** - src_strategy_file或者dst_strategy_file不是正确的切分策略proto文件。
        - **TypeError** - src_strategy_file或者dst_strategy_file不是字符串。
        - **TypeError** - rank_id不是一个整数。
