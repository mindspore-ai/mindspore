mindspore.ops.coo_neg
======================

.. py:function:: mindspore.ops.coo_neg(x)

    计算输入COOTensor的相反数并返回。

    .. math::
        out_{i} = - x_{i}

    参数：
        - **x** (COOTensor) - Neg的输入，数据类型为数值型。

    返回：
        COOTensor，shape和数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `x` 不是COOTensor。