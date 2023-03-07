mindspore.ops.permute
=====================

.. py:function:: mindspore.ops.permute(input, axis)

    按照输入 `axis` 的维度顺序排列输入Tensor。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **axis** (Union[tuple(int), list(int), int]) - 维度的顺序，permute根据 `axis` 的顺序重新排列 `input` 。

    返回：
        Tensor，具有和输入Tensor相同的维数，按照 `axis` 重新排列。

    异常：
        - **ValueError** - `axis` 为None。
        - **ValueError** - `axis` 的元素总量不等于 `input` 的维数。
