mindspore.ops.sinc
==================

.. py:function:: mindspore.ops.sinc(x)

    按照以下公式逐元素计算输入Tensor的数学正弦函数。

    .. math::
        out_i = \begin{cases} \frac{sin(\pi x_i)}{x_i} & x_i\neq 0\\ 
        1 & x_i=0 \end{cases}

    参数：
        - **x** (Tensor) - `x` 的shape为 :math:`(x_1, x_2, ..., x_R)`。

    返回：
        Tensor，shape与 `x` 相同。
        当输入类型为[uint8, uint8, uint16, int16, uint32, int32, uint64, int64, bool]时，返回值类型为float32。
        否则，返回值类型与输入类型相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
