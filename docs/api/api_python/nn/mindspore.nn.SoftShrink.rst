mindspore.nn.SoftShrink
========================

.. py:class:: mindspore.nn.SoftShrink(lambd=0.5)

    逐元素计算SoftShrink激活函数。

    .. math::
        \text{SoftShrink}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    SoftShrink函数图：

    .. image:: ../images/Softshrink.png
        :align: center

    参数：
        - **lambd** (number, 可选) - SoftShrink公式中的 :math:`\lambda` ，必须不小于零。默认 ``0.5`` 。

    输入：
        - **input** (Tensor) - 输入tensor。支持数据类型：

          - Ascend：mindspore.float16、mindspore.float32、mindspore.bfloat16。
          - CPU/GPU：mindspore.float16、mindspore.float32。

    输出：
        Tensor。