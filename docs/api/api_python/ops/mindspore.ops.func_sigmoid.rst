mindspore.ops.sigmoid
=====================

.. py:function:: mindspore.ops.sigmoid(input_x)

    逐元素计算Sigmoid激活函数。Sigmoid函数定义为：

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)}

    其中， :math:`x_i` 是input_x的一个元素。

    参数：
        - **input_x** (Tensor) - 任意维度的Tensor，数据类型为float16、float32、float64、complex64或complex128。

    返回：
        Tensor，数据类型和shape与 `input_x` 的相同。

    异常：
        - **TypeError** - `input_x` 的数据类型不是float16、float32、float64、complex64或complex128。
        - **TypeError** - `input_x` 不是Tensor。