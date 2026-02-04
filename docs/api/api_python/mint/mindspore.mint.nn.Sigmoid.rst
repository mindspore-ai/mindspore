mindspore.mint.nn.Sigmoid
=============================

.. py:class:: mindspore.mint.nn.Sigmoid

    逐元素计算Sigmoid激活函数。

    Sigmoid函数定义为：

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    其中 :math:`x_i` 是x的一个元素。

    Sigmoid函数图：

    .. image:: ../images/Sigmoid.png
        :align: center

    输入：
        - **input** (Tensor) - 输入tensor。
    输出：
        Tensor。