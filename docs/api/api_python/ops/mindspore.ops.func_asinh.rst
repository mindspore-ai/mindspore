mindspore.ops.asinh
====================

.. py:function:: mindspore.ops.asinh(x)

    计算输入元素的反双曲正弦。

    .. math::

        out_i = \sinh^{-1}(x_i)

    参数：
        - **x** (Tensor) - 需要计算反双曲正弦函数的输入。

    返回：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
