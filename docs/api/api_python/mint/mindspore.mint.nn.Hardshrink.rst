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

    HShrink函数图：

    .. image:: ../images/HShrink.png
        :align: center

    参数：
        - **lambd** (number，可选) - Hard Shrink公式定义的阈值 :math:`\lambda` 。默认值： ``0.5`` 。

    输入：
        - **input** (Tensor) - Hard Shrink的输入。支持数据类型：

          - Ascend：float16、float32、bfloat16。
          - CPU/GPU：float16、float32。
    输出：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `lambd` 不是float、int或bool。
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的dtype不是float16、float32或bfloat16。
