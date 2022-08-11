mindspore.nn.Sigmoid
=============================

.. py:class:: mindspore.nn.Sigmoid

    Sigmoid激活函数。

    按元素计算Sigmoid激活函数。

    Sigmoid函数定义为：

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    其中 :math:`x_i` 是输入的元素。

    关于Sigmoid的图例见 `Sigmoid <https://en.wikipedia.org/wiki/Sigmoid_function#/media/File:Logistic-curve.svg>`_ 。

    输入：
        - **input_x** (Tensor) - 数据类型为float16、float32、float64、complex64或complex128的Sigmoid输入。任意维度的Tensor。

    输出：
        Tensor，数据类型和shape与 `input_x` 的相同。

    异常：
        - **TypeError** - `input_x` 的数据类型不是float16、float32、float64、complex64或complex128。
