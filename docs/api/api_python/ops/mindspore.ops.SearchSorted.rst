mindspore.ops.SearchSorted
===========================

.. py:class:: mindspore.ops.SearchSorted(dtype=mstype.int64, right=False)

    返回元素能够插入输入tensor的位置索引，以维持原tensor的递增顺序。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.searchsorted`。

    参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 输出的数据类型。可选值为： ``mstype.int32`` 和 ``mstype.int64`` 。默认 ``mstype.int64`` 。
        - **right** (bool, 可选) - 搜索策略。如果为 ``True`` ，则返回找到的最后一个合适的索引；如果为 ``False`` ，则返回第一个合适的索引。默认 ``False`` 。

    输入：
        - **sorted_sequence** (Tensor) - 输入tensor。如果未提供 `sorter` ，最内层的维度上须为递增的序列。
        - **values** (Tensor) - 要插入元素的值。
        - **sorter** (Tensor, 可选) - 按 `sorted_sequence` 最内层维度升序排序的索引序列，与未排序的 `sorted_sequence` 共同使用。CPU和GPU只支持 ``None`` 。默认 ``None`` 。

    输出：
        Tensor