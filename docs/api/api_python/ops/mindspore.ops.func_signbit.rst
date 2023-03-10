mindspore.ops.signbit
======================

.. py:function:: mindspore.ops.signbit(input)

    在符号位已设置（小于零）的情况下，按元素位置返回True。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor， `input` 的signbit计算结果。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。