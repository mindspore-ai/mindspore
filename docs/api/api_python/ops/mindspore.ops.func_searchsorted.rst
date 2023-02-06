mindspore.ops.searchsorted
==========================

.. py:function:: mindspore.ops.searchsorted(sorted_sequence, values, *, out_int32=False, right=False)

    返回位置索引，根据这个索引将 `values` 插入 `sorted_sequence` 后，`sorted_sequence` 的最内维度的顺序保持不变。

    参数：
        - **sorted_sequence** (Tensor) - Tensor的形状为 :math:`(x_1, x_2, …, x_R-1, x_R)` 或 `x_1`。在最里面的维度上必须包含单调递增的序列。
        - **values** (Tensor) - 要插入元素的值。Tensor的形状为 :math:`(x_1, x_2, …, x_R-1, x_S)`。

    关键字参数：
        - **out_int32** (bool, 可选) - 输出数据类型。如果为True，则输出数据类型将为int32；如果为False，则输出数据类型将为int64。默认值：False。
        - **right** (bool, 可选) - 搜索策略。如果为True，则返回找到的最后一个合适的索引；如果为False，则返回第一个合适的索引。默认值：False。

    返回：
        表示 `sorted_sequence` 最内维度的索引的Tensor，如果插入 `values` tensor中相应的值，则 `sorted_sequence` tensor的顺序将被保留；如果out_int32为True，则返回的数据类型为int32，否则为int64，并且形状与values的形状相同。

    异常：
        - **ValueError** - 如果 `sorted_sequence` 的维度不是1，并且除 `sorted_sequence` 和 `values` 的最后一个维度之外的维度不同。
