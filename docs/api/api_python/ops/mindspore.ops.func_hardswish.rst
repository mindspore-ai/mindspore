mindspore.ops.hardswish
=======================

.. py:function:: mindspore.ops.hardswish(input)

    逐元素计算Hard Swish。输入是一个Tensor，具有任何有效的shape。

    Hard Swish定义如下：

    .. math::
        \text{Hardswish}(input) =
        \begin{cases}
        0, & \text{ if } input \leq -3, \\
        input, & \text{ if } input \geq +3, \\
        input*(input + 3)/6, & \text{ otherwise }
        \end{cases}

    HSwish函数图：

    .. image:: ../images/HSwish.png
        :align: center

    参数：
        - **input** (Tensor) - Hard Swish的输入。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `input` 不是一个Tensor。
        - **TypeError** - `input` 不是int或者float类型。