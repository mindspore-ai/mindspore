mindspore.ops.signbit
======================

.. py:function:: mindspore.ops.signbit(x)

    在符号位已设置（小于零）的情况下，按元素位置返回True。

    参数：
        - **x** (Tensor) - 输入Tensor。

    返回：
        Tensor， `x` 的signbit计算结果。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。