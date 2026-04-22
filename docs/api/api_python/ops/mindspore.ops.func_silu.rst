mindspore.ops.silu
==================

.. py:function:: mindspore.ops.silu(input)

    按输入逐元素计算激活函数SiLU（Sigmoid Linear Unit），也称为Swish函数。该激活函数定义为：

    .. math::
        \text{SiLU}(x) = x * \sigma(x),

    其中 :math:`x` 是输入的元素， :math:`\sigma(x)` 是Sigmoid函数。

    .. math::

        \text{sigma}(x_i) = \frac{1}{1 + \exp(-x_i)},

    SiLU函数图：

    .. image:: ../images/SiLU.png
        :align: center

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将新增可选参数 `inplace` ，接口签名将变更为
        ``mindspore.ops.silu(input, inplace=False)`` 。
        该变更与当前版本兼容，不使用 `inplace` 参数时行为不变。

    参数：
        - **input** (Tensor) - `input` 即为上述公式中的 :math:`x`。数据类型为float16或float32的输入。

    返回：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 的数据类型不是float16或float32。
