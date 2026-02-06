mindspore.ops.diagonal_scatter
==============================

.. py:function:: mindspore.ops.diagonal_scatter(input, src, offset=0, dim1=0, dim2=1)

    将源tensor的值嵌入到输入tensor的对角线中（基于 `dim1` 和 `dim2` 指定的二维平面）。

    .. note::
        目前， `input` 和 `src` 中的元素不支持 ``inf`` 值。

    参数：
        - **input** (Tensor) - 维度大于1的输入tensor。
        - **src** (Tensor) - 要嵌入的源tensor。
        - **offset** (int, 可选) - 对角线偏移量。默认 ``0`` 。

          - 当 `offset` 是正整数时，对角线向上方偏移。
          - 当 `offset` 是负整数时，对角线向下方偏移。

        - **dim1** (int, 可选) - 确定指定平面对角线的第一维度。默认 ``0`` 。
        - **dim2** (int, 可选) - 确定指定平面对角线的第二维度。默认 ``1`` 。

    返回：
        Tensor
