mindspore.mint.nn.functional.hardshrink
=======================================

.. py:function:: mindspore.mint.nn.functional.hardshrink(input, lambd=0.5)

    Hard Shrink激活函数。按输入元素计算输出。公式定义如下：

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    HShrink激活函数图：

    .. image:: ../images/HShrink.png
        :align: center

    参数：
        - **input** (Tensor) - Hard Shrink的输入。支持数据类型：

          - Ascend：float16、float32、bfloat16。
        - **lambd** (number，可选) - Hard Shrink公式定义的阈值 :math:`\lambda` 。默认值： ``0.5`` 。

    返回：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `lambd` 不是float, bool和int。
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的dtype不是float16、float32和bfloat16。