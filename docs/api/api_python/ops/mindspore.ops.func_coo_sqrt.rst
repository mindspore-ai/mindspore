mindspore.ops.coo_sqrt
=======================

.. py:function:: mindspore.ops.coo_sqrt(x)

    逐元素计算COOTensor的平方根。

    .. math::
        out_{i} = \sqrt{x_{i}}

    参数：
        - **x** (COOTensor) - 输入COOTensor，数据类型为number.Number。

    返回：
        COOTensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是COOTensor。
