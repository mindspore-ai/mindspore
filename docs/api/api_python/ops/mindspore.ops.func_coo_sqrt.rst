mindspore.ops.coo_sqrt
=======================

.. py:function:: mindspore.ops.coo_sqrt(x: COOTensor)

    逐元素返回当前COOTensor的平方根。

    .. math::
        out_{i} = \sqrt{x_{i}}

    参数：
        - **x** (COOTensor) - 输入COOTensor，数据类型为number.Number，其rank需要在[0, 7]范围内.

    返回：
        COOTensor，具有与 `x` 相同的shape。

    异常：
        - **TypeError** - 如果 `x` 不是COOTensor。

