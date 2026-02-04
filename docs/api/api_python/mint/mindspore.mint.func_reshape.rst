mindspore.mint.reshape
=====================================

.. py:function:: mindspore.mint.reshape(input, shape)

    根据给定形状重新排列输入张量。

    .. note::
        参数 `shape` 中的 ``-1`` 表示该维度的大小将根据其他维度和输入张量总元素数自动推导得出。

    参数：
        - **input** (Tensor) - 输入张量。
        - **shape** (Union[tuple[int], list[int], Tensor[int]]) - 新的形状。

    返回：
        Tensor
