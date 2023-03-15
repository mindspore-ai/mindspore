mindspore.ops.diagflat
======================

.. py:function:: mindspore.ops.diagflat(input, offset=0)

    创建一个二维Tensor，用展开后的 `input` 作为它的对角线。

    参数：
        - **input** (Tensor) - 输入Tensor，展开后作为输出的对角线。
        - **offset** (int, 可选) - `offset` 控制选择哪条对角线。默认值：0。

          - 当 `offset` 是0时，选择的对角线是主对角线。
          - 当 `offset` 是正整数时，选择的对角线在主对角线上。
          - 当 `offset` 是负整数时，选择的对角线在主对角线下。

    返回：
        二维Tensor，对角线是展开的 `input` 。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `offset` 不是整数。
