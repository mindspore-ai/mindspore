mindspore.ops.coo_acosh
========================

.. py:function:: mindspore.ops.coo_acosh(x: COOTensor)

    逐元素计算输入COOTensor的反双曲余弦。

    .. math::
        out_i = \cosh^{-1}(input_i)

    .. warning::
        给定一个输入COOTensor `x` ，该函数计算每个元素的反双曲余弦。输入范围为[1, inf]。

    参数：
        - **x** (COOTensor) - 需要计算反双曲余弦函数的输入COOTensor，其秩的范围必须在[0, 7]。

    返回：
        COOTensor，数据类型与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是COOTensor。
