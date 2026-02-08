mindspore.ops.csr_acosh
========================

.. py:function:: mindspore.ops.csr_acosh(x)

    逐元素计算输入CSRTensor的反双曲余弦。

    .. math::
        out_i = \cosh^{-1}(input_i)

    参数：
        - **x** (CSRTensor) - 需要计算反双曲余弦函数的输入CSRTensor，即 :math:`input_i`，其每个元素的取值范围必须在[1, inf]。

    返回：
        CSRTensor，数据类型与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是CSRTensor。
