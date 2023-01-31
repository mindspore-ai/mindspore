mindspore.ops.silu
==================

.. py:function:: mindspore.ops.silu(x)

    按输入逐元素计算激活函数SiLU（Sigmoid Linear Unit）。该激活函数定义为：

    .. math::
        \text{SiLU}(x) = x * \sigma(x),

    其中，math:`\sigma(x)` 是Logistic Sigmoid函数。

    .. math::

        \text{sigma}(x_i) = \frac{1}{1 + \exp(-x_i)},

    其中，:math:`x_i` 是输入x的元素。

    更多详情请参考 :class:`mindspore.nn.SiLU`。
