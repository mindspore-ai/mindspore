mindspore.Tensor.reshape
========================

.. py:method:: mindspore.Tensor.reshape(*shape)

    不改变数据的情况下，将Tensor的shape改为输入的新shape。

    参数：
        - **shape** (Union[int, tuple(int), list(int)]) - 新的shape应与原来的shape兼容。如果参数值为整数，则结果是该长度的一维数组。shape的维度可以为-1。在这种情况下，将根据数组的长度和剩下的维度计算出该值。

    返回：
        Tensor，具有新shape的Tensor。

    异常：
        - **TypeError** - 新shape不是整数、列表或元组。
        - **ValueError** - 新shape与原来Tensor的shape不兼容。