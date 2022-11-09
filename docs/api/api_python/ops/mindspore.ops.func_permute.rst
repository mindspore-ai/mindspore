mindspore.ops.permute
=====================

.. py:function:: mindspore.ops.permute(x, dims)

    按照输入 `dims` 的维度顺序排列输入Tensor。

    参数：
        - **x** (Tensor) - 输入Tensor。
        - **dims** (Union[tuple(int), list(int), int]) - 维度的顺序，permute根据 `dims` 的顺序重新排列 `x` 。

    返回：
        Tensor，具有和输入Tensor相同的维数，按照 `dims` 重新排列。

    异常：
        - **ValueError** - `dims` 为None。
        - **ValueError** - `dims` 的元素总量不等于 `x` 的维数。
