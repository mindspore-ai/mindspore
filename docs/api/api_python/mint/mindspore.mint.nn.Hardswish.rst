mindspore.mint.nn.Hardswish
============================

.. py:class:: mindspore.mint.nn.Hardswish

    逐元素计算Hard Swish激活函数。

    Hard Swish定义如下：

    .. math::
        \text{HardSwish}(input) =
        \begin{cases}
        0, & \text{ if } input \leq -3, \\
        input, & \text{ if } input \geq +3, \\
        input*(input + 3)/6, & \text{ otherwise }
        \end{cases}

    HardSwish函数图：

    .. image:: ../images/Hardswish.png
        :align: center

    输入：
        - **input** (Tensor) - 输入Tensor。

    输出：
        Tensor，输出tensor的数据类型和shape与输入相同。

