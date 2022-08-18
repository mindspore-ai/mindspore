mindspore.Tensor.tanh
=====================

.. py:method:: mindspore.Tensor.tanh()

    Tanh激活函数。

    按元素计算当前Tensor元素的双曲正切。Tanh函数定义为：

    .. math::
        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    其中 :math:`x_i` 是输入Tensor的元素。

    返回：
        Tensor，数据类型和shape与当前Tensor相同。

    异常：
        - **TypeError** - 如果当前Tensor的数据类型不为以下类型： float16、 float32、 float64、 complex64 和 complex128。
