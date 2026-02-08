mindspore.mint.nn.Hardshrink
============================

.. py:class:: mindspore.mint.nn.Hardshrink(lambd=0.5)

    逐元素计算Hard Shrink激活函数。公式定义如下：

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    HardShrink函数图：

    .. image:: ../images/Hardshrink.png
        :align: center

    参数：
        - **lambd** (number，可选) - Hard Shrink激活函数公式中定义的阈值 :math:`\lambda` 。默认 ``0.5`` 。

    输入：
        - **input** (Tensor) - 输入tensor。支持的数据类型：

          - Ascend：float16、float32、bfloat16。

    输出：
        Tensor，输出Tensor的shape和数据类型与输入相同。

