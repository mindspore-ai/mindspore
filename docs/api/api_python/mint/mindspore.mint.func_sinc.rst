mindspore.mint.sinc
=====================

.. py:function:: mindspore.mint.sinc(input)

    计算输入的归一化正弦值。

    .. math::
        out_i = \begin{cases} \frac{sin(\pi input_i)}{\pi input_i} & input_i\neq 0\\ 
        1 & input_i=0 \end{cases}

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，shape与 `input` 相同。
        当输入类型为int或bool时，返回值类型为float32。
        否则，返回值类型与输入类型相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。