mindspore.Tensor.soft_shrink
============================

.. py:method:: mindspore.Tensor.soft_shrink(lambd=0.5)

    Soft Shrink激活函数，按输入元素计算输出。公式定义如下：

    .. math::
        \text{SoftShrink}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    参数：
        - **lambd** (float) - :math:`\lambda` 应大于等于0。默认值：0.5。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `lambd` 不是float。
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - 原始Tensor的dtype既不是float16也不是float32。
        - **ValueError** - `lambd` 小于0。