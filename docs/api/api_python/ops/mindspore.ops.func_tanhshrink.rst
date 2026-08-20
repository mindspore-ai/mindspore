mindspore.ops.tanhshrink
=========================

.. py:function:: mindspore.ops.tanhshrink(input)

    逐元素计算输入tensor的Tanhshrink激活函数值。

    .. math::
        Tanhshrink(x) = x - \tanh(x)

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor