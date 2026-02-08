mindspore.mint.nn.Softshrink
============================

.. py:class:: mindspore.mint.nn.Softshrink(lambd=0.5)

    逐元素计算Soft Shrink激活函数。公式定义如下：

    .. math::
        \text{SoftShrink}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    SoftShrink激活函数图：

    .. image:: ../images/Softshrink.png
        :align: center

    参数：
        - **lambd** (number，可选) - Softshrink公式定义的阈值 :math:`\lambda` ，不小于零。默认 ``0.5`` 。

    输入：
        - **input** (Tensor) - 输入tensor。支持数据类型：

          - Ascend：mindspore.float16、mindspore.float32、mindspore.bfloat16。
          - CPU/GPU：mindspore.float16、mindspore.float32。

    输出：
        Tensor。