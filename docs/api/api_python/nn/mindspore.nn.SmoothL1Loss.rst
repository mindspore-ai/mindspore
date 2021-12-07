mindspore.nn.SmoothL1Loss
============================

.. py:class:: mindspore.nn.SmoothL1Loss(beta=1.0)

    创建一个标准来计算loss函数，如果输入的绝对误差小于 `beta` 则用平方项，否则用绝对误差项。

    SmoothL1Loss可以看成 :class:`mindspore.nn.L1Loss` 的修改版本，也可以看成 :class:`mindspore.nn.L1Loss` 和 :class:`mindspore.ops.L2Loss` 的组合。 :class:`mindspore.nn.L1Loss` 计算两个输入Tensor之间的绝对误差，而 :class:`mindspore.ops.L2Loss` 计算两个输入Tensor之间的平方误差。 :class:`mindspore.ops.L2Loss` 通常更快收敛，但对离群值的鲁棒性较差。

    给定两个输入 :math:`x,\  y`，长度为 :math:`N`， unreduced SmoothL1Loss定义如下：

    .. math::
        L_{i} =
        \begin{cases}
        \frac{0.5 (x_i - y_i)^{2}}{\text{beta}}, & \text{if } |x_i - y_i| < \text{beta} \\
        |x_i - y_i| - 0.5 \text{beta}, & \text{otherwise.}
        \end{cases}

    其中， :math:`\text{beta}` 控制loss函数从二次变为线性。 默认值为1.0。 :math:`N` 为batch size。该函数返回一个unreduced loss Tensor。

    **参数：**

    **beta** (float) - 用于控制loss函数从二次变为线性的参数。默认值：1.0。

    **输入：**

    - **logits** (Tensor) - 预测值，shape为 :math:`(N, *)` 的Tensor，其中 :math:`*` 表示任意的附加维度数。数据类型必须为float16或float32。
    - **labels** (Tensor) - 目标值，shape为 :math:`(N, *)` 的Tensor，数据类型和shape与 `logits` 相同。

    **输出：**

    Tensor，shape和数据类型与 `logits` 相同。

    **异常：**

    - **TypeError** - `beta` 不是float。
    - **TypeError** - `logits` 或 `labels` 的数据类型既不是float16，也不是float32。
    - **ValueError** - `beta` 小于或等于0。
    - **ValueError** - `logits` 的shape与 `labels` 不同。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> loss = nn.SmoothL1Loss()
    >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
    >>> labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
    >>> output = loss(logits, labels)
    >>> print(output)
    [0.  0.  0.5]
    