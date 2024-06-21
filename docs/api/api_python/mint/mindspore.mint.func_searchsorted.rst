mindspore.mint.searchsorted
===========================

.. py:function:: mindspore.mint.searchsorted(sorted_sequence, values, *, out_int32=False, right=False, side=None, sorter=None)

    返回位置索引，根据这个索引将 `values` 插入 `sorted_sequence` 后，`sorted_sequence` 的最内维度的顺序保持不变。

    参数：
        - **sorted_sequence** (Tensor) - 输入Tensor。在最里面的维度上必须包含单调递增的序列。
        - **values** (Tensor) - 要插入元素的值。

    关键字参数：
        - **out_int32** (bool, 可选) - 输出数据类型。如果为 ``True`` ，则输出数据类型将为int32；如果为 ``False`` ，则输出数据类型将为int64。默认值： ``False`` 。
        - **right** (bool, 可选) - 搜索策略。如果为 ``True`` ，则返回找到的最后一个合适的索引；如果为 ``False`` ，则返回第一个合适的索引。默认值： ``False`` 。
        - **side** (str, 可选) - 跟参数 `right` 功能一致，如果参数值为 ``'left'``，相当于 `right` 为 ``False``。如果参数值为 ``'right'`` ，相当于 `right` 为 ``True``。如果值为 ``'left'`` 但是 `right` 为 ``True`` 则会报错。默认值： ``None`` 。
        - **sorter** (Tensor, 可选) - 如果提供，shape将与 `sorted_sequence` 一致, 类型必须为int64，包含整数索引，这些索引将在最内层维度上按升序给 `sorted_sequence` 排序。默认值： ``None`` 。

    返回：
        表示 `sorted_sequence` 最内维度的索引的Tensor，如果插入 `values` Tensor中相应的值，则 `sorted_sequence` Tensor的顺序将被保留；如果out_int32为True，则返回的数据类型为int32，否则为int64，并且形状与values的形状相同。

    异常：
        - **ValueError** - 如果 `sorted_sequence` 的维度不是1，并且除 `sorted_sequence` 和 `values` 的最后一个维度之外的维度不同。
        - **ValueError** - 如果 `sorted_sequence` 是Scalar。
        - **ValueError** - 如果 `values` 是Scalar，并且 `sorted_sequence` 的维度不是1。