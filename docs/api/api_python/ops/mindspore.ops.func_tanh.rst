mindspore.ops.tanh
===================

.. py:function:: mindspore.ops.tanh(input)

    逐元素计算输入元素的双曲正切。Tanh函数定义为：

    .. math::
        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    其中 :math:`x_i` 是输入Tensor的元素。

    参数：
        - **input** (Tensor) - Tanh的输入，任意维度的Tensor，其数据类型为float16或float32。

    返回：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 的数据类型既不是float16也不是float32。
        - **TypeError** - `input` 不是Tensor。