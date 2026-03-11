mindspore.mint.sinc
=====================

.. py:function:: mindspore.mint.sinc(input)

    计算输入的归一化正弦值。

    .. math::
        out_i = \begin{cases} \frac{\sin(\pi input_i)}{\pi input_i} & input_i\neq 0\\
        1 & input_i=0 \end{cases}

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor