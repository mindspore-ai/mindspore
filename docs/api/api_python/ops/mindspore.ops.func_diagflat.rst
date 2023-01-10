mindspore.ops.diagflat
======================

.. py:function:: mindspore.ops.diagflat(x, offset=0)

    创建一个二维Tensor，用展开后的输入作为它的对角线。

    参数：
        - **x** (Tensor) - 输入Tensor，展开后作为输出的对角线。
        - **offset** (int, 可选) - `offset` 控制选择哪条对角线。默认值：0。

          - 当 `offset` 是0时，选择的对角线是主对角线。
          - 当 `offset` 大于0时，选择的对角线在主对角线上。
          - 当 `offset` 小于0时，选择的对角线在主对角线下。

    返回：
        二维Tensor。

    异常：
        - **TypeError** - `x` 不是Tensor。
