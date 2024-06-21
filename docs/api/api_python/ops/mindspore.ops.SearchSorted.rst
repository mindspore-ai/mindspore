mindspore.ops.SearchSorted
===========================

.. py:class:: mindspore.ops.SearchSorted(dtype=mstype.int64, right=False)

    返回位置索引，根据这个索引将 `values` 插入 `sorted_sequence` 后，`sorted_sequence` 的元素大小顺序保持不变。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.searchsorted`。

    参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 输出数据类型。可选值为： ``mstype.int32`` 和 ``mstype.int64`` 。默认值： ``mstype.int64`` 。
        - **right** (bool, 可选) - 搜索策略。如果为 ``True`` ，则返回找到的最后一个合适的索引；如果为 ``False`` ，则返回第一个合适的索引。默认值： ``False`` 。

    输入：
        - **sorted_sequence** (Tensor) - Tensor的shape为 :math:`(x_1, x_2, ..., x_R-1, x_R)` 或 `x_1`。在最里面的维度上必须包含单调递增的序列。
        - **values** (Tensor) - 要插入元素的值。Tensor的shape为 :math:`(x_1, x_2, ..., x_R-1, x_S)`。
        - **sorter** (Tensor, 可选) -  如果提供，shape将与 `sorted_sequence` 一致, 类型必须为int64，包含整数索引，这些索引将在最内层维度上按升序给 `sorted_sequence` 排序。默认值： ``None`` 。

    输出：
        表示 `sorted_sequence` 最内维度的索引的Tensor，如果插入 `values` Tensor中相应的值，则 `sorted_sequence` Tensor的顺序将被保留；如果out_int32为True，则返回的数据类型为int32，否则为int64，并且shape与values的shape相同。

    异常：
        - **ValueError** - 如果 `sorted_sequence` 的维度不是1，并且除 `sorted_sequence` 和 `values` 的最后一个维度之外的维度不同。
        - **ValueError** - 如果 `sorted_sequence` 是Scalar。
        - **ValueError** - 如果 `values` 是Scalar，并且 `sorted_sequence` 的维度不是1。