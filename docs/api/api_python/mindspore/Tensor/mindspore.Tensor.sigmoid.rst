mindspore.Tensor.sigmoid
=============================

.. py:method:: mindspore.Tensor.sigmoid

    Sigmoid激活函数，按元素计算Sigmoid激活函数。

    Sigmoid函数定义为：

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    其中 :math:`x_i` 是原Tensor的元素。

    返回：
        Tensor，数据类型和shape与原Tensor的相同。

    异常：
        - **TypeError** - 原Tensor的数据类型不是float16、float32、float64、complex64或complex128。
