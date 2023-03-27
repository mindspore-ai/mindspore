mindspore.ops.signbit
======================

.. py:function:: mindspore.ops.signbit(input)

    判断每个元素的符号，如果元素值小于0则对应输出的位置为True，否则为False。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor， `input` 的signbit计算结果。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。