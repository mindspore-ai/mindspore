mindspore.ops.sinc
==================

.. py:function:: mindspore.ops.sinc(input)

    按照以下公式逐元素计算输入Tensor的数学正弦函数。

    .. math::
        out_i = \begin{cases} \frac{sin(\pi input_i)}{input_i} & input_i\neq 0\\ 
        1 & input_i=0 \end{cases}

    参数：
        - **input** (Tensor) - `input` 的shape为 :math:`(input_1, input_2, ..., input_R)`。

    返回：
        Tensor，shape与 `input` 相同。
        当输入类型为int或bool时，返回值类型为float32。
        否则，返回值类型与输入类型相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
