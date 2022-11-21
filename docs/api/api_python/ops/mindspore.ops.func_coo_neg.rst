mindspore.ops.coo_neg
======================

.. py:function:: mindspore.ops.coo_neg(x: COOTensor)

    计算输入COOTensor的相反数并返回。

    .. math::
        out_{i} = - x_{i}

    参数：
        - **x** (COOTensor) - Neg的输入，shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。秩应小于8。其数据类型为数值型。

    返回：
        COOTensor，shape和类型与输入相同。

    异常：
        - **TypeError** - `x` 不是COOTensor。