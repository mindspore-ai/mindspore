mindspore.ops.Sigmoid
=====================

.. py:class:: mindspore.ops.Sigmoid

    Sigmoid激活函数，逐元素计算Sigmoid激活函数。Sigmoid函数定义为：

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)}

    其中， :math:`x_i` 是输入Tensor的一个元素。

    更多参考详见 :func:`mindspore.ops.sigmoid`。

    输入：
        - **input_x** (Tensor) - 任意维度的Tensor，数据类型为float16或float32。

    输出：
        Tensor，数据类型和shape与 `input_x` 的相同。
