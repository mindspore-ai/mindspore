mindspore.nn.SoftShrink
========================

.. py:class:: mindspore.nn.SoftShrink(lambd=0.5)

    SoftShrink激活函数。

    .. math::
        \text{SoftShrink}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    参数：
        - **lambd** (float) - Softshrink公式中的 :math:`\lambda` ，必须不小于零。默认值：0.5。

    输入：
        - **input_x** (Tensor) - SoftShrink的输入，任意维度的Tensor，数据类型为float16或float32。

    输出：
        Tensor，shape和数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - `lambd` 不是float。
        - **TypeError** - `input_x` 不是tensor。
        - **TypeError** - `input_x` 的数据类型既不是float16也不是float32。
        - **ValueError** - `lambd` 小于0。