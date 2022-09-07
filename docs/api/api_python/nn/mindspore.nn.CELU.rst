mindspore.nn.CELU
==================

.. py:class:: mindspore.nn.CELU(alpha=1.0)

    CELU激活层（CELU Activation Operator）。

    根据Continuously Differentiable Exponential Linear Units对输入Tensor逐元素计算。

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

    其返回值为 :math:`\max(0,x)+\min(0,\Alpha*(\exp(x/\Alpha)-1)` 。

    更多详情，请查看： `CELU <https://arxiv.org/abs/1704.07483>`_ 。

    参数：
        - **alpha** (float) - CELU公式中的 :math:`\alpha` 值。默认值：1.0。

    输入：
        - **x** (Tensor) - CELU的输入。其数据类型为float16或float32，shape为 :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。

    输出：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `α` 不是float。
        - **ValueError** - 如果 `α` 的值为0。
        - **TypeError** - 如果 输入 `x` 不是Tensor。
        - **TypeError** - 如果输入 `x` 的数据类型既不是float16也不是float32。
