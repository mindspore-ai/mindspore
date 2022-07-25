mindspore.nn.SiLU
=============================

.. py:class:: mindspore.nn.SiLU

    SiLU激活函数。

    逐元素计算SiLU激活函数。

    SiLU函数定义为：

    .. math::

        \text{SiLU}(x) = x * \sigma(x),

    其中 :math:`x_i` 是输入的元素， :math:`\sigma(x)` 是Sigmoid函数。

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    关于SiLU的图例见 `SiLU <https://en.wikipedia.org/wiki/Activation_function#/media/File:Swish.svg>`_ 。

    输入：
        - **x** (Tensor) - 数据类型为float16或float32的输入。任意维度的Tensor。

    输出：
        Tensor，数据类型和shape与 `x` 的相同。

    异常：
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
