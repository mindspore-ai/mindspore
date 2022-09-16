mindspore.Tensor.searchsorted
=============================

.. py:method:: mindspore.Tensor.searchsorted(v, side='left', sorter=None)

    查找应插入元素以保存顺序的位置索引。

    参数：
        - **v** (Union[int, float, bool, list, tuple, Tensor]) - 要插入元素的值。
        - **side** ('left', 'right', 可选) - 如果参数值为'left'，则给出找到的第一个合适位置的索引。如果参数值为'right'，则返回最后一个这样的索引。如果没有合适的索引，则返回0或N（其中N是Tensor的长度）。默认值：'left'。
        - **sorter** (Union[int, float, bool, list, tuple, Tensor]) - 整数索引的可选一维数组，将Tensor按升序排序。它们通常是NumPy argsort方法的结果。默认值：None。

    返回：
        Tensor，shape与 `v` 相同的插入点数组。

    异常：
        - **ValueError** - `side` 或 `sorter` 的参数无效。