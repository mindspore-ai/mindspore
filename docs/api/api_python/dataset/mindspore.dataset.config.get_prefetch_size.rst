mindspore.dataset.config.get_prefetch_size
===========================================

.. py:function:: mindspore.dataset.config.get_prefetch_size()

    获取数据处理管道的输出缓存队列长度。
    如果 `set_prefetch_size` 方法未被调用，那么将会返回默认值16。

    返回：
        int，表示预取的总行数。
