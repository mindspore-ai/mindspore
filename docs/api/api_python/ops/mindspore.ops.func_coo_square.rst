mindspore.ops.coo_square
=========================

.. py:function:: mindspore.ops.coo_square(x: COOTensor)

    逐元素返回COOTensor的平方。

    .. math::
        y_i = x_i ^ 2

    参数：
        - **x** (COOTensor) - 输入COOTensor的维度范围为[0,7]，类型为数值类型。

    返回：
        COOTensor，具有与当前COOTensor相同的数据类型和shape。

    异常：
        - **TypeError** - `x` 不是COOTensor。
