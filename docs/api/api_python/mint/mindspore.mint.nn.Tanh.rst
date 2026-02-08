mindspore.mint.nn.Tanh
======================

.. py:class:: mindspore.mint.nn.Tanh

    逐元素计算Tanh（双曲正切值）激活函数。

    Tanh函数定义为：

    .. math::
        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    其中 :math:`x_i` 是输入Tensor的元素。

    Tanh函数图：

    .. image:: ../images/Tanh.png
        :align: center

    输入：
        - **input** (Tensor) - 输入tensor。

    输出：
        Tensor。
