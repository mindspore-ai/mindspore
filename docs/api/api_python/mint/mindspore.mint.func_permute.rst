mindspore.mint.permute
======================

.. py:function:: mindspore.mint.permute(input, dims)

    按照输入 `dims` 的维度顺序排列输入Tensor。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **dims** (tuple(int)) - 维度的顺序，permute根据 `dims` 的顺序重新排列 `input` 。

    返回：
        Tensor，具有和输入Tensor相同的维数，按照 `dims` 重新排列。

    异常：
        - **ValueError** - `dims` 为None。
        - **ValueError** - `dims` 的元素总量不等于 `input` 的维数。
