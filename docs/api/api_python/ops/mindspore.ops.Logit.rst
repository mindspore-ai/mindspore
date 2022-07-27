mindspore.ops.Logit
===================

.. py:class:: mindspore.ops.Logit(eps=-1.0)

    逐元素计算张量的logit值。 `x` 中的元素被截断到范围[eps, 1-eps]内。

    .. math::
        y_{i} = \ln(\frac{z_{i}}{1 - z_{i}}) \\
        z_{i} = \begin{cases}
        x_{i} &amp; \text{if eps is None} \\
        \text{eps} &amp; \text{if } x_{i} &lt; \text{eps} \\
        x_{i} &amp; \text{if } \text{eps} \leq x_{i} \leq 1 - \text{eps} \\
        1 - \text{eps} &amp; \text{if } x_{i} &gt; 1 - \text{eps}
        \end{cases}

    更多参考详见 :func:`mindspore.ops.logit`。
