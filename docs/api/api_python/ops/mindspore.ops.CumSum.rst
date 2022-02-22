mindspore.ops.CumSum
=====================

.. py:class:: mindspore.ops.CumSum(exclusive=False, reverse=False)

    在指定轴上计算输入Tensor的累加和。

    .. math::
        y_i = x_1 + x_2 + x_3 + ...+ x_i

    **参数：**

    - **exclusive** (bool) - 如果为True，则执行独占模式。默认值：False。
    - **reverse** (bool) - 如果为True，则逆向计算累加和。默认值：False。

    **输入：**

    - **input** (Tensor) - 输入要计算的Tensor。
    - **axis** (int) - 指定要累加和的轴。仅支持常量值。该值在[-rank(input), rank(input))范围中。

    **输出：**

    Tensor。输出Tensor的shape与输入Tensor的shape一致。

    **异常：**

    - **TypeError** - `exclusive` 或 `reverse` 不是bool。
    - **TypeError** - `axis` 不是int。