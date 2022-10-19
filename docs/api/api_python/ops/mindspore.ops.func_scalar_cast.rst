mindspore.ops.scalar_cast
==========================

.. py:function:: mindspore.ops.mindspore.ops.scalar_cast(input_x, input_y)

    将输入Scalar转换为其他类型。

    参数：
        - **input_x** (scalar) - 输入Scalar。只允许常量值。
        - **input_y** (mindspore.dtype) - 要强制转换的类型。只允许常量值。

    返回：
        Scalar，类型与 `input_y` 对应的python类型相同。

    异常：
        - **TypeError** - 如果 `input_x` 或 `input_y` 不是常量值。
