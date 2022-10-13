mindspore.Tensor.atan
=====================

.. py:method:: mindspore.Tensor.atan()

    逐元素计算输入张量的反正切值。

    .. math::
        out_i = tan^{-1}(x_i)

    返回：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型既不是float16，也不是float32。