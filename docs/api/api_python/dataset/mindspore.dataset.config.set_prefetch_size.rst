mindspore.dataset.config.set_prefetch_size
===========================================

.. py:function:: mindspore.dataset.config.set_prefetch_size(size)

    设置管道中线程的队列容量。

    参数：
        - **size** (int) - 表示缓存队列的长度。 `size` 取值必须大于0，否则线程的队列容量无效。

    异常：
        - **TypeError** - `size` 不是int类型。
        - **ValueError** - 如果 `size` 不为正数。

    .. note::
        用于预取的总内存可能会随着工作线程数量的增加而快速增长，所以当工作线程数量大于4时，每个工作线程的预取大小将减少。
        每个工作线程在运行时预取大小将是 `prefetchsize` * (4 / `num_parallel_workers` )。
