mindspore.ops.logit
===================

.. py:function:: mindspore.ops.logit(input, eps=None)

    逐元素计算Tensor的logit值。当 `eps` 不是 None 时， `input` 中的元素被截断到范围[eps, 1-eps]内。
    当 `eps` 为 None 时，输入 `input` 不进行数值截断。

    .. math::
        \begin{align}
        y_{i} & = \ln(\frac{z_{i}}{1 - z_{i}}) \\
        z_{i} & = \begin{cases}
        input_{i} & \text{if eps is None} \\
        \text{eps} & \text{if } x_{i} \lt \text{eps} \\
        input_{i} & \text{if } \text{eps} \leq input_{i} \leq 1 - \text{eps} \\
        1 - \text{eps} & \text{if } input_{i} \gt 1 - \text{eps}
        \end{cases}
        \end{align}

    参数：
        - **input** (Tensor) - Tensor输入。
        - **eps** (float, 可选) - epsilon值。输入的数值界限被定义[eps, 1-eps]。默认值：None。

    返回：
        Tensor，具有与 `input` 相同的shape。

    异常：
        - **TypeError** - `eps` 不是float类型。
        - **TypeError** - `input` 不是Tensor类型。
        - **TypeError** - `input` 的数据类型不是float16、float32或float64。
