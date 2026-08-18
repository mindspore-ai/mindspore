mindspore.ops.logaddexp
=======================

.. py:function:: mindspore.ops.logaddexp(input, other)

    计算输入的指数和的对数。
    该函数可以很好地解决统计学中计算得到的事件概率过小（超过了正常浮点数表达的范围）的问题。

    .. math::

        out_i = \log(\exp(input_i) + \exp(other_i))

    参数：
        - **input** (Tensor) - 输入tensor。
        - **other** (Tensor) - 输入tensor。

    返回：
        Tensor
