mindspore.ops.diagonal
=======================

.. py:function:: mindspore.ops.diagonal(input, offset=0, dim1=0, dim2=1)

    返回输入tensor在指定维度上的对角线视图。

    若输入tensor是二维的，则返回一个一维tensor，包含给定偏移位置处的对角线。

    若输入tensor超过二维，则返回由 `dim1` 和 `dim2` 指定的二维平面的对角线。返回tensor的shape为移除 `input` 的 `dim1` 和 `dim2` 维度，并且由 `dim1` 和 `dim2` 确定的对角线元素插入 `input` 的最后一维。

    参数：
        - **input** (Tensor) - 维度至少为二维的输入tensor。
        - **offset** (int, 可选) - 对角线偏移量。默认 ``0`` 。

          - 当 `offset` 是正整数时，对角线向上方偏移。
          - 当 `offset` 是负整数时，对角线向下方偏移。
        - **dim1** (int, 可选) - 返回指定平面对角线的第一维度。默认 ``0`` 。
        - **dim2** (int, 可选) - 返回指定平面对角线的第二维度。默认 ``1`` 。

    返回：
        Tensor