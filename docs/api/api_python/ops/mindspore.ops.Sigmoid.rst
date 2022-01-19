mindspore.ops.Sigmoid
=====================

.. py:class:: mindspore.ops.Sigmoid()

    Sigmoid激活函数。

    逐元素计算Sgmoid激活函数。Sigmoid函数定义为：

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)}

    其中， :math:`x_i` 是输入Tensor的一个元素。

    **输入：**

    - **input_x** (Tensor) - 任意维度的Tensor，数据类型为float16或float32。

    **输出：**

    Tensor，数据类型和shape与 `input_x` 的相同。

    **异常：**

    - **TypeError** - `input_x` 的数据类型既不是float16也不是float32。
    - **TypeError** - `input_x` 不是Tensor。
