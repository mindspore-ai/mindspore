mindspore.nn.SiLU
=============================

.. py:class:: mindspore.nn.SiLU

    逐元素计算SiLU激活函数。有时也被称作Swish函数。

    SiLU函数定义为：

    .. math::

        \text{SiLU}(x) = x * \sigma(x),

    其中 :math:`x_i` 是输入的元素， :math:`\sigma(x)` 是Sigmoid函数。

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    SiLU函数图：

    .. image:: ../images/SiLU.png
        :align: center

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将新增可选构造参数 `inplace` ，接口签名将变更为
        ``mindspore.nn.SiLU(inplace=False)`` 。
        该变更与当前版本兼容，不使用 `inplace` 参数时行为不变。

    输入：
        - **input** (Tensor) - `input` 即为上述公式中的 :math:`x`。数据类型为float16或float32，维度任意。

    输出：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 的数据类型既不是float16也不是float32。
