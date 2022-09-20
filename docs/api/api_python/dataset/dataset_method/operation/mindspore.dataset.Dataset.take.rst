mindspore.dataset.Dataset.take
===============================

.. py:method:: mindspore.dataset.Dataset.take(count=-1)

    从数据集中获取最多 `count` 的元素。

    .. note::
        1. 如果 `count` 大于数据集中的数据条数或等于-1，则取数据集中的所有数据。
        2. take和batch操作顺序很重要，如果take在batch操作之前，则取给定条数，否则取给定batch数。

    参数：
        - **count** (int, 可选) - 要从数据集对象中获取的数据条数，默认值：-1，获取所有数据。

    返回：
        TakeDataset，take操作后的数据集对象。
