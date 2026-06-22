mindspore.dataset.config.get_num_parallel_workers
==================================================

.. py:function:: mindspore.dataset.config.get_num_parallel_workers()

    获取并行工作线程数量的全局配置，该配置将作用于每个操作。如果之前未调用 `set_num_parallel_workers`，将返回默认值 8。

    返回：
        int，表示每个操作中默认的并行工作线程的数量。
