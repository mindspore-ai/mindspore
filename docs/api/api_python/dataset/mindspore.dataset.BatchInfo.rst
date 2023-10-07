mindspore.dataset.BatchInfo
===========================

.. py:class:: mindspore.dataset.BatchInfo

    当 `batch` 操作中参数 `batch_size` 或 `per_batch_map` 的传入对象是回调函数时，可以通过此类提供的方法获取数据集信息。

    .. py:method:: get_batch_num()

        返回当前epoch已经处理的batch数，数值从0开始。

    .. py:method:: get_epoch_num()

        返回当前的epoch数，数值从0开始。
