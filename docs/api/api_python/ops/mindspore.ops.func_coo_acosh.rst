mindspore.ops.coo_acosh
========================

.. py:function:: mindspore.ops.coo_acosh(x)

    逐元素计算输入COOTensor的反双曲余弦。

    .. math::
        y_i = \cosh^{-1}(x_i)

    .. warning::
        给定一个输入COOTensor `x` ，该函数计算每个元素的反双曲余弦。输入范围为[1, inf]。

    参数：
        - **x** (COOTensor) - 需要计算反双曲余弦函数的输入COOTensor。

    返回：
        COOTensor，数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是COOTensor。
