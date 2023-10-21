mindspore.dataset.Dataset.repeat
================================

.. py:method:: mindspore.dataset.Dataset.repeat(count=None)

    重复此数据集 `count` 次。如果 `count` 为 ``None`` 或 ``-1`` ，则无限重复迭代。

    .. note::
        repeat和batch的顺序反映了batch的数量。建议：repeat操作在batch操作之后使用。

    参数：
        - **count** (int) - 数据集重复的次数。默认值： ``None`` 。

    返回：
        Dataset，应用了上述操作的新数据集对象。
