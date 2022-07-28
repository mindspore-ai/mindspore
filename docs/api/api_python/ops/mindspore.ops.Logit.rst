mindspore.ops.Logit
===================

.. py:class:: mindspore.ops.Logit(eps=-1.0)

    逐元素计算张量的logit值。 `x` 中的元素被截断到范围[eps, 1-eps]内。

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

    更多参考详见 :func:`mindspore.ops.logit`。
