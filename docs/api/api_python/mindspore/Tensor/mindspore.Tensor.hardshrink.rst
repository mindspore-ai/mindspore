mindspore.Tensor.hardshrink
===========================

.. py:method:: mindspore.Tensor.hardshrink(lambd=0.5)

    对tensor应用Hard Shrink激活函数。按输入元素计算输出。公式定义如下：

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    参数：
        - **lambd** (float) - Hard Shrink公式定义的阈值 :math:`\lambda` 。默认值：0.5。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `lambd` 不是float。
        - **TypeError** - 原始Tensor的dtype既不是float16也不是float32。