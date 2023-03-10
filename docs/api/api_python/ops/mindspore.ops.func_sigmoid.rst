mindspore.ops.sigmoid
=====================

.. py:function:: mindspore.ops.sigmoid(input)

    逐元素计算Sigmoid激活函数。Sigmoid函数定义为：

    .. math::

        \text{sigmoid}(input_i) = \frac{1}{1 + \exp(-input_i)}

    其中， :math:`x_i` 是input的一个元素。

    参数：
        - **input** (Tensor) - 任意维度的Tensor，数据类型为float16、float32、float64、complex64或complex128。

    返回：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 的数据类型不是float16、float32、float64、complex64或complex128。
        - **TypeError** - `input` 不是Tensor。