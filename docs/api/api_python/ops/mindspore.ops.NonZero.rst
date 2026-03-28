mindspore.ops.NonZero
======================

.. py:class:: mindspore.ops.NonZero

    返回所有非零元素的索引位置。

    输入：
        - **input** (Tensor) - 输入Tensor。

          - Ascend: 其秩可以等于0。
          - CPU/GPU: 其秩应大于等于1。

    输出：
        二维Tensor，数据类型为int64，包含所有输入中的非零元素的索引位置。
        如果 `input` 的维数为 `D` ， `input` 中的非零个数为 `N` ，则输出的shape为 :math:`(N, D)` 。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **RuntimeError** - 在CPU或者GPU中，如果 `input` 的维度为0。
