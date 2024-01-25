mindspore.nn.Sigmoid
=============================

.. py:class:: mindspore.nn.Sigmoid

    逐元素计算Sigmoid激活函数。

    Sigmoid函数定义为：

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    其中 :math:`x_i` 是x的一个元素。

    Sigmoid函数图：

    .. image:: ../images/Sigmoid.png
        :align: center

    输入：
        - **input** (Tensor) - `input` 即为上述公式中的 :math:`x`。数据类型为float16、float32、float64、complex64或complex128的Sigmoid输入。任意维度的Tensor。

    输出：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 的数据类型不是float16、float32、float64、complex64或complex128。
        - **TypeError** - `input` 不是Tensor。
