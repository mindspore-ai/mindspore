mindspore.ops.silu
==================

.. py:function:: mindspore.ops.silu(x)

    激活函数SiLU（Sigmoid Linear Unit）。

    该激活函数定义为：

    .. math::
        \text{SiLU}(x) = x * \sigma(x),

    其中 :math:`x_i` 是输入的元素， math:`\sigma(x)` Logistic Sigmoid函数。

    .. math::

        \text{sigma}(x_i) = \frac{1}{1 + \exp(-x_i)},

    关于SiLU的图例见 `SiLU <https://en.wikipedia.org/wiki/Activation_function#/media/File:Swish.svg>`_ 。

    参数：
        - **x** (Tensor) - 数据类型为float16, float32, float64, complex64 或 complex128的输入。任意维度的Tensor。

    返回：
        Tensor，数据类型和shape与 `x` 的相同。

    异常：
        - **TypeError** - `x` 的数据类型不是float16, float32, float64, complex64 或 complex128。