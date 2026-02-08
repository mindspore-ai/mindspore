mindspore.mint.nn.Hardsigmoid
==============================

.. py:class:: mindspore.mint.nn.Hardsigmoid

    逐元素应用Hard Sigmoid激活函数。

    Hard Sigmoid定义为：

    .. math::
        \text{HardSigmoid}(input) =
        \begin{cases}
        0, & \text{ if } input \leq -3, \\
        1, & \text{ if } input \geq +3, \\
        input/6 + 1/2, & \text{ otherwise }
        \end{cases}

    HardSigmoid函数图：

    .. image:: ../images/Hardsigmoid.png
        :align: center

    输入：
        - **input** (Tensor) - 输入tensor。

    输出：
        Tensor，输出tensor的数据类型和shape与输入相同。

