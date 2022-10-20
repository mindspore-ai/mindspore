mindspore.Tensor.expm1
======================

.. py:method:: mindspore.Tensor.expm1()

    逐元素计算输入Tensor的指数，然后减去1。

    .. math::
        out_i = e^{x_i} - 1

    返回：
        Tensor，shape与当前Tensor相同。

    异常：
        - **TypeError** - 如果当前Tensor的数据类型既不是float16也不是float32。
