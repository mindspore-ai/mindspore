mindspore.ops.csr_tanh
=======================

.. py:function:: mindspore.ops.csr_tanh(x)

    逐元素计算CSRTensor输入元素的双曲正切。Tanh函数定义为：

    .. math::
        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    其中 :math:`x_i` 是输入CSRTensor的元素。

    参数：
        - **x** (CSRTensor) - Tanh的输入，任意维度的CSRTensor，其数据类型为float16或float32。

    返回：
        CSRTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
        - **TypeError** - `x` 不是CSRTensor。