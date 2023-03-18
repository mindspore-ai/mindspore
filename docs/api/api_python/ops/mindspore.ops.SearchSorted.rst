mindspore.ops.SearchSorted
===========================

.. py:class:: mindspore.ops.SearchSorted(dtype=mstype.int64, right=False)

    返回 `values` 中每个元素应该插入到 `sorted_sequence` 中的位置所对应的索引，以使 `sorted_sequence` 中元素的顺序在插值之后维持不变。

    更多参考详见 :func:`mindspore.ops.searchsorted`。

    参数：
        - **dtype** (class:`mindspore.dtype` ， 可选) - 输出数据类型。可选值为： `mstype.int32` 和 `mstype.int64` 。默认值： `mstype.int64` 。
        - **right** (bool, 可选) - 搜索策略。如果为True，则返回找到的最后一个合适的索引；如果为False，则返回第一个合适的索引。默认值：False。

    输入：
        - **sorted_sequence** (Tensor) - Tensor的形状为 :math:`(x_1, x_2, …, x_R-1, x_R)` 或 `x_1`。在最里面的维度上必须包含单调递增的序列。
        - **values** (Tensor) - 要插入元素的值。Tensor的形状为 :math:`(x_1, x_2, …, x_R-1, x_S)`。

    输出：
        表示 `sorted_sequence` 最内维度的索引的Tensor，如果插入 `values` tensor中相应的值，则 `sorted_sequence` tensor的顺序将被保留；如果out_int32为True，则返回的数据类型为int32，否则为int64，并且形状与values的形状相同。
