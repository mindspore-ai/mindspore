mindspore.ops.coo_square
=========================

.. py:function:: mindspore.ops.coo_square(x)

    逐元素返回COOTensor的平方。

    .. math::
        out_{i} = (x_{i})^2

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **x** (COOTensor) - 输入COOTensor的类型为数值类型。

    返回：
        COOTensor，具有与当前COOTensor相同的数据类型和shape。

    异常：
        - **TypeError** - `x` 不是COOTensor。
