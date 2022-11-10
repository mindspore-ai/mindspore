mindspore.ops.coo_acos
=======================

.. py:function:: mindspore.ops.coo_acos(x: COOTensor)

    逐元素计算输入COOTensor的反余弦。

    .. math::
        out_i = cos^{-1}(x_i)

    参数：
        - **x** (COOTensor) - COOTensor的shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。数据类型应该是以下类型之一：float16、float32、float64。

    返回：
        COOTensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是COOTensor。
        - **TypeError** - 如果 `x` 的数据类型不是float16、float32或float64。
