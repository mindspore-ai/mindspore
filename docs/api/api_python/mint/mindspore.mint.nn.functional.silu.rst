mindspore.mint.nn.functional.silu
==================================

.. py:function:: mindspore.mint.nn.functional.silu(input)

    逐元素计算激活函数SiLU（Sigmoid Linear Unit）。有时也被称作Swish函数。该激活函数定义为：

    .. math::
        \text{SiLU}(x) = x * \sigma(x),

    其中 :math:`x` 是输入的元素， :math:`\sigma(x)` 是Sigmoid函数。

    .. math::

        \text{sigma}(x_i) = \frac{1}{1 + \exp(-x_i)},

    SiLU函数图：

    .. image:: ../images/SiLU.png
        :align: center

    参数：
        - **input** (Tensor) - `input` 即为上述公式中的 :math:`x`。数据类型为float16或float32的输入。

    返回：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 的数据类型既不是float16也不是float32。
