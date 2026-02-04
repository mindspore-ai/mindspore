mindspore.ops.coo_sigmoid
==========================

.. py:function:: mindspore.ops.coo_sigmoid(x)

    Sigmoid激活函数，对COOTensor逐元素计算Sigmoid激活函数。Sigmoid函数定义为：

    .. math::

        \text{coo_sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)}

    其中， :math:`x_i` 是x的一个元素。

    参数：
        - **x** (COOTensor) - 任意维度的COOTensor，数据类型为float16、float32、float64、complex64或complex128。

    返回：
        COOTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型不是float16、float32、float64、complex64或complex128。
        - **TypeError** - `x` 不是COOTensor。