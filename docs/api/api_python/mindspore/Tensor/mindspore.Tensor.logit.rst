mindspore.Tensor.logit
======================

.. py:method:: mindspore.Tensor.logit(eps=None)

    逐元素计算张量的logit值。当 eps 不是 None 时， `x` 中的元素被截断到范围[eps, 1-eps]内。
    当 eps 为 None 时，输入 `x` 不进行数值截断。

    `x` 指的当前 Tensor。

    .. math::
        \begin{align}
        y_{i} & = \ln(\frac{z_{i}}{1 - z_{i}}) \\
        z_{i} & = \begin{cases}
        x_{i} & \text{if eps is None} \\
        \text{eps} & \text{if } x_{i} \lt \text{eps} \\
        x_{i} & \text{if } \text{eps} \leq x_{i} \leq 1 - \text{eps} \\
        1 - \text{eps} & \text{if } x_{i} \gt 1 - \text{eps}
        \end{cases}
        \end{align}

    参数：
        - **eps** (float) - epsilon值。输入的数值界限被定义[eps, 1-eps]。默认值：None。

    返回：
        Tensor，具有与 `x` 相同的shape。

    异常：
        - **TypeError** - `eps` 不是float类型。
        - **TypeError** - `x` 不是Tensor类型。
        - **TypeError** - `x` 的数据类型不是float16、float32或float64。