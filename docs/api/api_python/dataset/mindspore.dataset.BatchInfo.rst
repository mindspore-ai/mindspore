mindspore.dataset.BatchInfo
===========================

.. py:class:: mindspore.dataset.BatchInfo

    此类提供了两种方法获取数据集的批处理数量（batch size）和迭代数（epoch）属性。
    这些属性可以用于 `batch` 操作中的输入参数 `batch_size` 和 `per_batch_map`。

    .. py:method:: get_batch_num()

        返回数据集的批处理数量（batch size）。

    .. py:method:: get_epoch_num()

        返回数据集的迭代数（epoch）。
