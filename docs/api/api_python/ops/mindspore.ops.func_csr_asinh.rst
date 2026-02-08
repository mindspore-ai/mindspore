mindspore.ops.csr_asinh
========================

.. py:function:: mindspore.ops.csr_asinh(x)

    计算CSRTensor输入元素的反双曲正弦。

    .. math::

        out_i = \sinh^{-1}(input_i)

    参数：
        - **x** (CSRTensor) - 需要计算反双曲正弦函数的输入，即 :math:`input_i` 。

    返回：
        CSRTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。
