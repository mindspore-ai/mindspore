mindspore.ops.coo_abs
======================

.. py:function:: mindspore.ops.coo_abs(x: COOTensor)

    逐元素计算输入COOTensor的绝对值。

    .. math::
        out_i = |x_i|

    参数：
        - **x** (COOTensor) - 输入COOTensor。T其shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    返回：
        COOTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是COOTensor。
