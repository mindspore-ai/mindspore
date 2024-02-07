mindspore.ops.scalar_cast
==========================

.. py:function:: mindspore.ops.scalar_cast(input_x, input_y)

    该接口从2.3版本开始已被弃用，并将在未来版本中被移除，建议使用 `int(x)` 或 `float(x)` 代替。

    将输入Scalar转换为其他类型。

    参数：
        - **input_x** (scalar) - 输入Scalar。
        - **input_y** (mindspore.dtype) - 要强制转换的类型。只允许常量值，并且只能为以下值之一：mindspore.int64、mindspore.float64、mindspore.bool\_。

    返回：
        Scalar，类型与 `input_y` 对应的python类型相同。

    异常：
        - **TypeError** - 如果 `input_y` 不是合法值。
