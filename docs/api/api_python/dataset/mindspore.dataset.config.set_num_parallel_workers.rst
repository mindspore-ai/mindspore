mindspore.dataset.config.set_num_parallel_workers
==================================================

.. py:function:: mindspore.dataset.config.set_num_parallel_workers(num)

    为并行工作线程数量设置新的全局配置默认值。
    此设置会影响所有数据集操作的并行性。

    参数：
        - **num** (int) - 表示并行工作线程的数量，用作为每个数据集操作的默认值。

    异常：
        - **TypeError** - `num` 不是int类型。
        - **ValueError** - `num` 小于等于0或大于 `INT32_MAX(2147483647)`。
