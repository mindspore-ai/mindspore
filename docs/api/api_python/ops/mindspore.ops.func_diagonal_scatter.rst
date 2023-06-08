mindspore.ops.diagonal_scatter
==============================

.. py:function:: mindspore.ops.diagonal_scatter(input, src, offset=0, dim1=0, dim2=1)

    `dim1` 和 `dim2` 指定 `input` 的两个维度，这两个维度上的元素将被视为矩阵的元素，并且将 `src` 嵌入到该矩阵的对角线上。

    参数：
        - **input** (Tensor) - 输入Tensor，其维度大于1。
        - **src** (Tensor) - 要嵌入的源Tensor。
        - **offset** (int, 可选) - 控制选择哪条对角线。可以是正值或负值。默认值： ``0`` 。

          - 当 `offset` 是 ``0`` 时，选择的对角线是主对角线。
          - 当 `offset` 是正整数时，选择的对角线在主对角线上方。
          - 当 `offset` 是负整数时，选择的对角线在主对角线下方。

        - **dim1** (int, 可选) - 二维子数组的第一个轴，对角线应该从这里开始。默认值： ``0`` 。
        - **dim2** (int, 可选) - 二维子数组的第二个轴，对角线应该从这里开始。默认值： ``1`` 。

    返回：
        嵌入后的Tensor，具有与 `input` 相同的shape和dtype。

    异常：
        - **TypeError** - `input` 或 `src` 不是Tensor。
        - **TypeError** - `offset` ， `dim1` 或 `dim2` 不是整数。
