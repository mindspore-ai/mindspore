mindspore.nn.HShrink
=============================

.. py:class:: mindspore.nn.HShrink(lambd=0.5)

    逐元素计算Hard Shrink激活函数。公式定义如下：

    .. math::
        \text{HShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    HShrink函数图：

    .. image:: ../images/HShrink.png
        :align: center

    参数：
        - **lambd** (number，可选) - Hard Shrink激活函数公式中定义的阈值 :math:`\lambda` 。默认 ``0.5`` 。

    输入：
        - **input** (Tensor) - 输入tensor。支持的数据类型：

          - Ascend：float16、float32、bfloat16。
          - CPU/GPU：float16、float32。
    输出：
        Tensor，输出tensor的shape和数据类型与输入相同。
