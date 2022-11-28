mindspore.ops.isreal
====================

.. py:function:: mindspore.ops.isreal(x)

    逐元素判断是否为实数。
    一个复数的虚部是0时也被看作是实数。

    参数：
        - **x** (Tensor) - 输入Tensor。

    返回：
        Tensor，对应 `x` 元素为负inf的位置是true，反之为false。

    异常：
        - **TypeError** - `x` 不是Tensor。
